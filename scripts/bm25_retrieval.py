#!/usr/bin/env python3
"""
BM25 retrieval script - exact mirror of FAISS retrieval
Loads persisted BM25Retriever + question embeddings → hierarchical JSON
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from llama_index.retrievers.bm25 import BM25Retriever

# reranker gpu
from FlagEmbedding import FlagReranker
import pyarrow.parquet as pq


# Config - MIRRORS your FAISS retrieval exactly
INDEX_DIR = Path('indexes')
BM25_INDEX_DIR = INDEX_DIR / 'bm25_patent_index'
ID_MAPPING_FILE = INDEX_DIR / 'bm25_patent_id_mapping.npy'
EMBEDDINGS_DIR = "data/question_generation/embeddings"
PARQUET_SHARDS_DIR = "/project/jevans/apto_data_engineering/data/arxiv/s2_arxiv_full_text_grobid/shards_partitioned"
OUTPUT_DIR = "data/question_generation/retrieval_results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "retrieval_results_bm25_hierarchical.json")
K = 100

EMBEDDING_FILES = [
    ("questions_remembering_embeddings.npy", "questions_remembering_metadata.csv", "remembering"),
    ("questions_understanding_embeddings.npy", "questions_understanding_metadata.csv", "understanding")
]

# Pipeline configuration
BM25_TOP_K = 30000  # Stage 1: BM25 keywords → top 30K
SEMANTIC_TOP_K = 1000  # Stage 2: Semantic reranking → top 1K
FINAL_TOP_K = 10  # Stage 3: LLM reranker → top 10

# Copy your exact functions from FAISS retrieval
def extract_paper_id(paragraph_id): return paragraph_id.split('_')[0]
def extract_paragraph_number(paragraph_id): 
    try: return int(paragraph_id.split('_p')[1])
    except: return None
def get_adjacent_paragraph_ids(paragraph_id): 
    paper_id = extract_paper_id(paragraph_id)
    para_num = extract_paragraph_number(paragraph_id)
    if para_num is None: return None, None
    return f"{paper_id}_p{para_num - 1}" if para_num > 0 else None, f"{paper_id}_p{para_num + 1}"

def load_paragraph_texts(paragraph_ids, load_adjacent=False):
    """EXACT copy from your FAISS script"""
    print(f"\nLoading {len(paragraph_ids)} paragraph texts...")
    
    paragraph_texts = {}
    needed_ids = set(paragraph_ids)
    if load_adjacent:
        expanded_ids = set(needed_ids)
        for pid in paragraph_ids:
            prev_id, next_id = get_adjacent_paragraph_ids(pid)
            if prev_id: expanded_ids.add(prev_id)
            if next_id: expanded_ids.add(next_id)
        needed_ids = expanded_ids
    
    shards_dir = Path(PARQUET_SHARDS_DIR)
    shard_dirs = sorted([d for d in shards_dir.iterdir() if d.is_dir() and d.name.startswith('shard=')])
    
    for shard_dir in tqdm(shard_dirs, desc="Searching shards"):
        parquet_files = list(shard_dir.glob("*.parquet"))
        for pq_file in parquet_files:
            try:
                table = pq.read_table(pq_file, columns=['paper_paragraph_id', 'paragraph_text'])
                df = table.to_pandas()
                mask = df['paper_paragraph_id'].isin(needed_ids)
                for _, row in df[mask].iterrows():
                    pid = row['paper_paragraph_id']
                    if pid not in paragraph_texts:
                        paragraph_texts[pid] = row['paragraph_text']
                if len(paragraph_texts) >= len(needed_ids): break
            except: continue
        if len(paragraph_texts) >= len(needed_ids): break
    
    return paragraph_texts

def main(limit_questions=None):
    print("="*80)
    print("Loading BM25 index...")
    
    # === LOAD BM25Retriever - EXACTLY like doc ===
    bm25_retriever = BM25Retriever.from_persist_dir(str(BM25_INDEX_DIR))
    bm25_retriever.similarity_top_k = K  # Set default top_k
    print(f"✓ BM25Retriever loaded: {BM25_INDEX_DIR}")

    print("Loading GPU reranker...")
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True, device='cuda')
    print("✓ GPU reranker loaded")

    
    # Load ID mapping (same as FAISS)
    id_mapping = np.load(str(ID_MAPPING_FILE), allow_pickle=True)
    print(f"✓ ID mapping loaded: {len(id_mapping):,} IDs")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patents = {}
    all_paragraph_ids = set()
    
    # Process each question type
    for emb_file, meta_file, bloom_level in EMBEDDING_FILES:
        print(f"\n{'='*40}\nProcessing: {bloom_level}\n{'='*40}")
        
        emb_path = os.path.join(EMBEDDINGS_DIR, emb_file)
        df = pd.read_csv(os.path.join(EMBEDDINGS_DIR, meta_file))
        
        if limit_questions:
            df = df.head(limit_questions)
        
        print(f"Processing {len(df)} questions...")
        
        # Load embeddings for semantic reranking (Stage 2)
        embeddings = np.load(emb_path).astype('float32')
        embeddings_subset = embeddings[:len(df)]
        
        for idx in tqdm(range(len(df)), desc="Questions"):
            row = df.iloc[idx]
            question = row['question']
            patent_id = str(row['patent_id'])
            question_embedding = embeddings_subset[idx]
            
            # ===== STAGE 1: BM25 Search on Keywords → Top 30K =====
            bm25_retriever.similarity_top_k = BM25_TOP_K
            nodes_stage1 = bm25_retriever.retrieve(row['keywords'])
            stage1_ids = [node.node_id for node in nodes_stage1]
            stage1_texts = np.array([node.text for node in nodes_stage1])
            
            # ===== STAGE 2: Semantic Reranking (numpy L2 distance) → Top 1K =====
            # Load embeddings for retrieved paragraphs (using a simple proxy: average word embeddings)
            # For efficiency, compute L2 distances directly
            stage1_node_embeddings = np.array([node.embedding if hasattr(node, 'embedding') else None for node in nodes_stage1])
            
            # If embeddings not available, compute simple text-based similarity
            if stage1_node_embeddings[0] is None:
                # Fallback: use simple keyword overlap as proxy for semantic similarity
                question_words = set(question.lower().split())
                semantic_scores = []
                for text in stage1_texts:
                    text_words = set(text.lower().split())
                    overlap = len(question_words & text_words) / max(len(question_words), len(text_words), 1)
                    semantic_scores.append(overlap)
                semantic_scores = np.array(semantic_scores)
            else:
                # Compute L2 distances between question embedding and retrieved paragraphs
                distances = np.linalg.norm(stage1_node_embeddings - question_embedding, axis=1)
                semantic_scores = -distances  # Negative distances for ranking (higher = better)
            
            # Get top SEMANTIC_TOP_K by semantic score
            semantic_indices = np.argsort(semantic_scores)[::-1][:SEMANTIC_TOP_K]
            stage2_ids = [stage1_ids[i] for i in semantic_indices]
            stage2_texts = stage1_texts[semantic_indices]
            stage2_nodes = [nodes_stage1[i] for i in semantic_indices]
            
            # ===== STAGE 3: LLM Reranker on Top 1K → Top 10 =====
            # Use full patent context (title + abstract + question) for reranking
            patent_context = f"Patent Title: {row['patent_title']}\nPatent Abstract: {row['patent_abstract']}\nQuestion: {question}"
            passage_texts = stage2_texts.tolist()
            scores = reranker.compute_score([[patent_context, passage] for passage in passage_texts])
            reranked_indices = np.argsort(scores)[::-1][:FINAL_TOP_K]
            top_k_paragraph_ids = [stage2_ids[i] for i in reranked_indices]
            
            # Rest is IDENTICAL to your FAISS script
            top_k_paper_ids = [extract_paper_id(pid) for pid in top_k_paragraph_ids]
            unique_papers = list(dict.fromkeys(top_k_paper_ids))
            
            if patent_id not in patents:
                patents[patent_id] = {
                    'cpc': row['cpc_class'],
                    'patent_title': row['patent_title'],
                    'patent_abstract': row['patent_abstract'],
                    'question_types': {'remembering': [], 'understanding': []}
                }
            
            papers_dict = defaultdict(list)
            for rank, para_id in enumerate(top_k_paragraph_ids[:10], 1):
                paper_id = extract_paper_id(para_id)
                all_paragraph_ids.add(para_id)
                papers_dict[paper_id].append({
                    'paragraph_id': para_id, 'rank': rank,
                    'has_paragraph': True, 'paragraph_text': None,
                    'contains_answer': None, 'answer': None
                })
            
            retrieved_papers = [{
                'paper_id': paper_id, 'metadata': {},
                'paragraphs': sorted(paragraphs, key=lambda x: x['rank'])
            } for paper_id, paragraphs in papers_dict.items()]
            
            retrieved_papers.sort(key=lambda x: min(p['rank'] for p in x['paragraphs']))
            
            question_entry = {
                'question': question,
                'retrieved_papers': retrieved_papers,
                'stats': {'unique_papers@100': len(unique_papers)}
            }
            
            patents[patent_id]['question_types'][bloom_level].append(question_entry)
    
    # Save (same as FAISS)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(patents, f, indent=2)
    print(f"\n✓ Saved BM25 results: {OUTPUT_JSON}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    main(limit_questions=args.limit)
