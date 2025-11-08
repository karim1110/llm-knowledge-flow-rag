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

def calculate_metrics(retrieved_ids, ground_truth_id, top_k_values):
    """Calculate Recall@K and MRR@K"""
    results = {}
    
    # Extract paper ID from paragraph ID (e.g., "12345_p3" -> "12345")
    gt_paper = ground_truth_id.split('_')[0] if isinstance(ground_truth_id, str) else str(ground_truth_id)
    
    for k in top_k_values:
        top_k_ids = retrieved_ids[:k]
        
        # Extract paper IDs from retrieved paragraph IDs
        retrieved_papers = [rid.split('_')[0] if isinstance(rid, str) else str(rid) for rid in top_k_ids]
        
        # Recall@K: Is ground truth in top K?
        results[f'recall@{k}'] = 1 if gt_paper in retrieved_papers else 0
        
        # MRR@K: Reciprocal rank of first match
        try:
            rank = retrieved_papers.index(gt_paper) + 1  # 1-indexed
            results[f'mrr@{k}'] = 1.0 / rank
        except ValueError:
            results[f'mrr@{k}'] = 0.0
    
    return results

def main():
    print("="*80)
    print("RAG Evaluation - GPU IVF-PQ Index (Compressed)")
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
        all_questions.append(df)
    
    questions_df = pd.concat(all_questions, ignore_index=True)
    print(f"✓ Loaded {len(questions_df):,} questions")
    
    # Embed questions
    print("\nEmbedding questions...")
    task = "Given a web search query, retrieve relevant passages that answer the query"
    instruct_questions = [get_detailed_instruct(task, q) for q in questions_df['question'].tolist()]
    
    question_embeddings = embed_texts_vllm(instruct_questions, model, tokenizer)
    print(f"✓ Embedded {len(question_embeddings):,} questions")
    
    # Retrieve for each question
    print(f"\nRetrieving top {max(TOP_K)} results for each question...")
    max_k = max(TOP_K)
    distances, indices = index.search(question_embeddings, max_k)
    
    # Map indices to paper_paragraph_ids
    results = []
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing"):
        retrieved_indices = indices[idx]
        retrieved_ids = [id_mapping[i] for i in retrieved_indices if i < len(id_mapping)]
        retrieved_dists = distances[idx]
        
        # Calculate metrics
        metrics = calculate_metrics(retrieved_ids, row['paper_id'], TOP_K)
        
        result = {
            'question': row['question'],
            'ground_truth_id': row['paper_id'],
            'bloom_level': row.get('bloom_level', 'unknown'),
            'top_10_retrieved': ';'.join(map(str, retrieved_ids[:10])),
            'top_10_distances': ';'.join(map(str, retrieved_dists[:10])),
            **metrics
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Calculate aggregate metrics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    for k in TOP_K:
        recall = results_df[f'recall@{k}'].mean()
        mrr = results_df[f'mrr@{k}'].mean()
        print(f"Recall@{k:3d}: {recall:.4f}  |  MRR@{k:3d}: {mrr:.4f}")
    
    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    
    print("\nEvaluation complete!")

if __name__ == '__main__':
    main()
