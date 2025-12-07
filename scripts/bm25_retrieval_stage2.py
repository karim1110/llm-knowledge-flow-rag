#!/usr/bin/env python3
"""
BM25 Retrieval Stage 2: GPU reranker
- Loads stage 1 output (top 1K per question)
- Uses FlagReranker with full patent context → top 10
- Saves final hierarchical JSON
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import FlagReranker

# Config
INPUT_DIR = "data/question_generation/retrieval_results"
INPUT_JSON = os.path.join(INPUT_DIR, "retrieval_results_bm25_stage1.json")
OUTPUT_JSON = os.path.join(INPUT_DIR, "retrieval_results_bm25_hierarchical.json")
FINAL_TOP_K = 10

def extract_paper_id(paragraph_id):
    return paragraph_id.split('_')[0]

def main():
    print("="*80)
    print("BM25 Retrieval Stage 2: GPU Reranker (top 1K → top 10)")
    print("="*80)
    
    # Load stage 1 results
    print(f"\nLoading stage 1 results from: {INPUT_JSON}")
    with open(INPUT_JSON, 'r') as f:
        patents = json.load(f)
    print(f"✓ Loaded {len(patents)} patents")
    
    # Load GPU reranker
    print("\nLoading GPU reranker...")
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True, device='cuda')
    print("✓ GPU reranker loaded")
    
    # Process all patents and questions
    total_questions = 0
    for patent_id, patent_data in tqdm(patents.items(), desc="Patents"):
        for bloom_level in ['remembering', 'understanding']:
            questions = patent_data['question_types'][bloom_level]
            
            for q_idx, question_entry in enumerate(questions):
                total_questions += 1
                question = question_entry['question']
                patent_title = patent_data['patent_title']
                patent_abstract = patent_data['patent_abstract']
                
                # Build full context for reranking
                patent_context = f"Patent Title: {patent_title}\nPatent Abstract: {patent_abstract}\nQuestion: {question}"
                
                # Collect all paragraphs from retrieved papers (top 1K from stage 1)
                all_para_ids = []
                all_para_texts = []
                
                for paper in question_entry['retrieved_papers']:
                    for para in paper['paragraphs']:
                        all_para_ids.append(para['paragraph_id'])
                        # In stage 1, we don't have paragraph texts, so use empty string
                        all_para_texts.append("")
                
                if len(all_para_ids) == 0:
                    continue
                
                # === STAGE 2: LLM Reranker on Top 1K → Top 10 ===
                # Create reranking queries
                rerank_pairs = [[patent_context, para_id] for para_id in all_para_ids]
                scores = reranker.compute_score(rerank_pairs)
                scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
                
                # Get top 10 by reranker score
                top_indices = np.argsort(scores)[::-1][:FINAL_TOP_K]
                top_para_ids = [all_para_ids[i] for i in top_indices]
                top_scores = scores[list(top_indices)]
                
                # Rebuild papers_dict with only top 10
                papers_dict = {}
                for rank, (para_id, score) in enumerate(zip(top_para_ids, top_scores), 1):
                    paper_id = extract_paper_id(para_id)
                    
                    if paper_id not in papers_dict:
                        papers_dict[paper_id] = []
                    
                    papers_dict[paper_id].append({
                        'paragraph_id': para_id,
                        'rank': rank,
                        'has_paragraph': False,
                        'paragraph_text': None,
                        'contains_answer': None,
                        'answer': None,
                        'rerank_score': float(score)
                    })
                
                # Convert to list format
                retrieved_papers = [{
                    'paper_id': paper_id,
                    'metadata': {},
                    'paragraphs': sorted(paragraphs, key=lambda x: x['rank'])
                } for paper_id, paragraphs in papers_dict.items()]
                
                retrieved_papers.sort(key=lambda x: min(p['rank'] for p in x['paragraphs']))
                
                # Update question entry
                question_entry['retrieved_papers'] = retrieved_papers
                question_entry['stats'] = {
                    'final_papers': len(papers_dict),
                    'final_top_k': FINAL_TOP_K,
                    'stage': 2,
                    'note': 'Top 10 after GPU reranking with full patent context'
                }
    
    # Save final results
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(patents, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✓ Stage 2 Complete!")
    print(f"{'='*80}")
    print(f"Processed: {total_questions} questions")
    print(f"Output: {OUTPUT_JSON}")
    print(f"\nReady for: load_paragraph_texts.py → verify_paragraphs.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BM25 Retrieval Stage 2: GPU reranking (top 1K → top 10)')
    args = parser.parse_args()
    main()
