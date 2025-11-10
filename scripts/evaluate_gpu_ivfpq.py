#!/usr/bin/env python3
"""
RAG Evaluation - GPU IVF-PQ Index (Compressed, Fast)

Evaluates retrieval performance using the compressed IVF-PQ FAISS index on GPU.
- Uses vLLM + Qwen embeddings (same as embed_texts.py)
- Loads IVF-PQ index to GPU for fast approximate search
- Evaluates Recall@K and MRR@K metrics
- No reranker (pure vector similarity retrieval)

Input: Questions CSV with columns: question, paper_id, bloom_level
Output: CSV with retrieval results and metrics
"""

import pandas as pd
import numpy as np
import faiss
import torch
from pathlib import Path
from tqdm import tqdm
from vllm import LLM
from transformers import AutoTokenizer

# Prevent vLLM connection issues
import vllm.envs as envs
envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

# Config
QUESTIONS_DIR = Path("data/question_generation")
FAISS_INDEX = Path("indexes/faiss_qwen06_ivfpq.index")
ID_MAPPING = Path("indexes/patent_id_mapping_ivf.npy")
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
OUTPUT_FILE = Path("data/question_generation/rag_eval_gpu_ivfpq.csv")
TOP_K = [1, 5, 10, 20, 50, 100]
USE_GPU = True
GPU_ID = 0

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction (for Qwen embeddings)"""
    return f'Instruct: {task_description}\nQuery: {query}'

def embed_texts_vllm(texts, model, tokenizer, max_tokens=32000, microbatch=1024):
    """
    Embed texts using vLLM with head+tail strategy for long texts.
    Same logic as embed_texts.py.
    """
    expanded, owners, weights = [], [], []
    n_each = max_tokens // 2

    for i, t in enumerate(texts):
        ids = tokenizer.encode(t, add_special_tokens=False)
        L = len(ids)
        if L <= max_tokens:
            expanded.append(t)
            owners.append(i)
            weights.append(L if L > 0 else 1)
        else:
            head_ids = ids[:n_each]
            tail_ids = ids[-n_each:]
            head = tokenizer.decode(head_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            tail = tokenizer.decode(tail_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            expanded.extend([head, tail])
            owners.extend([i, i])
            weights.extend([len(head_ids), len(tail_ids)])

    # Embed in microbatches
    outs = []
    for s in range(0, len(expanded), microbatch):
        outs.extend(model.embed(expanded[s:s+microbatch], use_tqdm=False))

    embs = np.array([o.outputs.embedding for o in outs], dtype=np.float32)

    # Pool back to one vector per original text (length-weighted mean)
    d = embs.shape[1]
    pooled = np.zeros((len(texts), d), dtype=np.float32)
    wsum = np.zeros(len(texts), dtype=np.float32)

    for emb, owner, w in zip(embs, owners, weights):
        pooled[owner] += emb * float(w)
        wsum[owner] += float(w)

    pooled /= np.clip(wsum[:, None], 1e-9, None)

    # Normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pooled = pooled / norms
    return pooled

def main():
    print("="*80)
    print("RAG Evaluation - GPU IVF-PQ Index (Compressed)")
    print("Understanding-Based Knowledge Flow Measurement")
    print("="*80)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: No GPU available, using CPU (will be slower)")
        use_gpu = False
    else:
        use_gpu = USE_GPU
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    
    # Load vLLM model for embeddings
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = LLM(model=MODEL_NAME, task="embed", trust_remote_code=True, enable_chunked_prefill=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    print("✓ Model loaded")
    
    # Load FAISS index
    print(f"\nLoading FAISS index: {FAISS_INDEX}")
    index_cpu = faiss.read_index(str(FAISS_INDEX))
    
    if use_gpu:
        print(f"Transferring index to GPU {GPU_ID}...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, GPU_ID, index_cpu)
        print(f"✓ Index on GPU ({index.ntotal:,} vectors)")
    else:
        index = index_cpu
        print(f"✓ Index on CPU ({index.ntotal:,} vectors)")
    
    # Load ID mapping
    print(f"Loading ID mapping: {ID_MAPPING}")
    id_mapping = np.load(ID_MAPPING, allow_pickle=True)
    print(f"✓ Loaded {len(id_mapping):,} paper_paragraph_ids")
    print(f"  Sample IDs: {id_mapping[:5]}")
    
    # Load questions
    print(f"\nLoading questions from: {QUESTIONS_DIR}")
    question_files = list(QUESTIONS_DIR.glob("questions_*.csv"))
    
    if not question_files:
        print(f"ERROR: No question files found in {QUESTIONS_DIR}")
        return
    
    print(f"Found {len(question_files)} question files:")
    for f in question_files:
        print(f"  - {f.name}")
    
    all_questions = []
    for f in question_files:
        df = pd.read_csv(f)
        # Infer bloom_level from filename (e.g., questions_remembering.csv -> remembering)
        bloom_level = f.stem.replace('questions_', '')
        df['bloom_level'] = bloom_level
        # Keep patent_id as is (this is the technology being questioned)
        if 'patent_id' in df.columns:
            df['patent_id'] = df['patent_id'].astype(str)
        all_questions.append(df)
    
    questions_df = pd.concat(all_questions, ignore_index=True)
    print(f"✓ Loaded {len(questions_df):,} questions")
    print(f"  Bloom levels: {questions_df['bloom_level'].unique().tolist()}")
    print(f"  Patents: {questions_df['patent_id'].nunique()} unique patents")
    
    # Embed questions
    print("\nEmbedding questions...")
    task = "Given a web search query, retrieve relevant passages that answer the query"
    instruct_questions = [get_detailed_instruct(task, q) for q in questions_df['question'].tolist()]
    
    question_embeddings = embed_texts_vllm(instruct_questions, model, tokenizer)
    print(f"✓ Embedded {len(question_embeddings):,} questions")
    
    # Retrieve for each question
    print(f"\nRetrieving top {max(TOP_K)} scientific paragraphs for each patent question...")
    max_k = max(TOP_K)
    distances, indices = index.search(question_embeddings, max_k)
    
    # Build results: patent question → retrieved scientific papers
    results = []
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing"):
        retrieved_indices = indices[idx]
        retrieved_paragraph_ids = [id_mapping[i] for i in retrieved_indices if i < len(id_mapping)]
        retrieved_dists = distances[idx]
        
        # Extract unique paper IDs from paragraph IDs (e.g., "12345_p3" -> "12345")
        retrieved_paper_ids = [pid.split('_')[0] if isinstance(pid, str) else str(pid) 
                              for pid in retrieved_paragraph_ids]
        
        # For top K values, track unique papers retrieved
        papers_at_k = {}
        for k in TOP_K:
            unique_papers = list(dict.fromkeys(retrieved_paper_ids[:k]))  # Preserve order, remove duplicates
            papers_at_k[f'unique_papers@{k}'] = len(unique_papers)
        
        result = {
            'patent_id': row['patent_id'],
            'question': row['question'],
            'bloom_level': row['bloom_level'],
            'patent_title': row.get('patent_title', ''),
            'cpc_class': row.get('cpc_class', ''),
            # Top 10 retrieved paragraphs (for inspection)
            'top_10_paragraph_ids': ';'.join(retrieved_paragraph_ids[:10]),
            'top_10_distances': ';'.join(map(lambda x: f'{x:.4f}', retrieved_dists[:10])),
            # Top 100 unique papers (for analysis)
            'top_100_papers': ';'.join(list(dict.fromkeys(retrieved_paper_ids[:100]))),
            **papers_at_k
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "="*80)
    print("RETRIEVAL SUMMARY")
    print("="*80)
    print(f"Total questions processed: {len(results_df):,}")
    print(f"Unique patents: {results_df['patent_id'].nunique()}")
    print(f"\nAverage unique papers retrieved per question:")
    for k in TOP_K:
        avg_papers = results_df[f'unique_papers@{k}'].mean()
        print(f"  Top {k:3d}: {avg_papers:.1f} unique papers")
    
    print(f"\nBy Bloom level:")
    for level in results_df['bloom_level'].unique():
        count = (results_df['bloom_level'] == level).sum()
        print(f"  {level:15s}: {count:4d} questions")
    
    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    
    print("\n" + "="*80)
    print("Next Steps for Analysis:")
    print("="*80)
    print("1. Examine top_10_paragraph_ids to see which scientific papers were retrieved")
    print("2. Aggregate by CPC class to build epistemic linkage matrix Ekl")
    print("3. Compare retrieval patterns across Bloom levels")
    print("4. Optional: Use LLM to evaluate if retrieved paragraphs answer the questions")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
