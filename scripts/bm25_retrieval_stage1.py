#!/usr/bin/env python3
"""
BM25 Retrieval Stage 1: CPU-only
- BM25 search on keywords → top 30K
- Semantic reranking (numpy L2 distance) → top 1K
- Save intermediate results to JSON (no reranker, no GPU needed)
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from llama_index.retrievers.bm25 import BM25Retriever

# Config
INDEX_DIR = Path('indexes')
BM25_INDEX_DIR = INDEX_DIR / 'bm25_patent_index'
ID_MAPPING_FILE = INDEX_DIR / 'bm25_patent_id_mapping.npy'
EMBEDDINGS_DIR = "data/question_generation/embeddings"
OUTPUT_DIR = "data/question_generation/retrieval_results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "retrieval_results_bm25_stage1.json")
K = 100

EMBEDDING_FILES = [
    ("questions_remembering_embeddings.npy", "questions_remembering_metadata.csv", "remembering"),
    ("questions_understanding_embeddings.npy", "questions_understanding_metadata.csv", "understanding")
]

# Pipeline configuration - Stage 1 only
BM25_TOP_K = 30000  # Stage 1: BM25 keywords → top 30K
SEMANTIC_TOP_K = 1000  # Stage 2: Semantic reranking → top 1K

def extract_paper_id(paragraph_id):
    return paragraph_id.split('_')[0]

def main(limit_questions=None):
    print("="*80)
    print("BM25 Retrieval Stage 1: CPU-only (BM25 + Semantic Reranking)")
    print("="*80)
    
    print("\nLoading BM25 index...")
    bm25_retriever = BM25Retriever.from_persist_dir(str(BM25_INDEX_DIR))
    bm25_retriever.similarity_top_k = K
    print(f"✓ BM25Retriever loaded: {BM25_INDEX_DIR}")
    
    # Load ID mapping
    id_mapping = np.load(str(ID_MAPPING_FILE), allow_pickle=True)
    print(f"✓ ID mapping loaded: {len(id_mapping):,} IDs")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patents = {}
    
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
            
            # Build hierarchical structure for stage 1 output
            # Include ALL top 1K paragraphs, not just top 10
            top_k_paragraph_ids = stage2_ids  # Keep all 1K for stage 2
            top_k_paper_ids = [extract_paper_id(pid) for pid in top_k_paragraph_ids]
            unique_papers = list(dict.fromkeys(top_k_paper_ids))
            
            if patent_id not in patents:
                patents[patent_id] = {
                    'cpc': row['cpc_class'],
                    'patent_title': row['patent_title'],
                    'patent_abstract': row['patent_abstract'],
                    'question_types': {'remembering': [], 'understanding': []}
                }
            
            # Store all 1K paragraphs (will be reranked in stage 2)
            papers_dict = defaultdict(list)
            for rank, para_id in enumerate(top_k_paragraph_ids, 1):
                paper_id = extract_paper_id(para_id)
                papers_dict[paper_id].append({
                    'paragraph_id': para_id,
                    'rank': rank,
                    'has_paragraph': False,  # Text not loaded in stage 1
                    'paragraph_text': None,
                    'contains_answer': None,
                    'answer': None,
                    'rerank_score': None  # Will be filled in stage 2
                })
            
            retrieved_papers = [{
                'paper_id': paper_id,
                'metadata': {},
                'paragraphs': sorted(paragraphs, key=lambda x: x['rank'])
            } for paper_id, paragraphs in papers_dict.items()]
            
            retrieved_papers.sort(key=lambda x: min(p['rank'] for p in x['paragraphs']))
            
            question_entry = {
                'question': question,
                'retrieved_papers': retrieved_papers,
                'stats': {
                    'unique_papers@1000': len(unique_papers),
                    'stage': 1,
                    'note': 'Top 1K candidates - waiting for stage 2 reranker'
                }
            }
            
            patents[patent_id]['question_types'][bloom_level].append(question_entry)
    
    # Save stage 1 results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(patents, f, indent=2)
    print(f"\n✓ Stage 1 complete: {OUTPUT_JSON}")
    print(f"  Ready for Stage 2 GPU reranking")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BM25 Retrieval Stage 1: CPU-only (BM25 + semantic reranking)')
    parser.add_argument('--limit', type=int, default=None, help='Limit to first N questions for testing')
    args = parser.parse_args()
    main(limit_questions=args.limit)
