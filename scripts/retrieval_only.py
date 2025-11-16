"""
Retrieval only script for GPU job (no model loading, no internet required)
Loads pre-computed question embeddings and performs FAISS search
Outputs hierarchical JSON structure organized by patent -> question type -> questions -> papers
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from collections import defaultdict

# Configuration
INDEX_FILE = "indexes/faiss_qwen06_ivfpq.index"
ID_MAPPING_FILE = "indexes/patent_id_mapping_ivf.npy"
EMBEDDINGS_DIR = "data/question_generation/embeddings"
PARQUET_SHARDS_DIR = "/project/jevans/apto_data_engineering/data/arxiv/s2_arxiv_full_text_grobid/shards_partitioned"
OUTPUT_DIR = "data/question_generation/retrieval_results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "retrieval_results_hierarchical.json")
K = 100  # Top K results

# Embedding files
EMBEDDING_FILES = [
    ("questions_remembering_embeddings.npy", "questions_remembering_metadata.csv", "remembering"),
    ("questions_understanding_embeddings.npy", "questions_understanding_metadata.csv", "understanding")
]

def extract_paper_id(paragraph_id):
    """Extract paper ID from paragraph ID (e.g., '2510268_p5' -> '2510268')"""
    return paragraph_id.split('_')[0]

def extract_paragraph_number(paragraph_id):
    """Extract paragraph number from ID (e.g., '2510268_p5' -> 5)"""
    try:
        return int(paragraph_id.split('_p')[1])
    except (IndexError, ValueError):
        return None

def get_adjacent_paragraph_ids(paragraph_id):
    """Get IDs of previous and next paragraphs if they exist."""
    paper_id = extract_paper_id(paragraph_id)
    para_num = extract_paragraph_number(paragraph_id)
    
    if para_num is None:
        return None, None
    
    prev_id = f"{paper_id}_p{para_num - 1}" if para_num > 0 else None
    next_id = f"{paper_id}_p{para_num + 1}"
    
    return prev_id, next_id

def merge_with_context(paragraph_id, paragraph_text, all_paragraphs, min_length=200):
    """
    Merge paragraph with adjacent paragraphs if it's too short.
    
    Args:
        paragraph_id: The paragraph ID
        paragraph_text: The main paragraph text
        all_paragraphs: Dict of all loaded paragraphs
        min_length: Minimum character length threshold
    
    Returns:
        Merged text with context if needed, otherwise original text
    """
    # If paragraph is long enough, return as-is
    if len(paragraph_text) >= min_length:
        return paragraph_text
    
    # Get adjacent paragraph IDs
    prev_id, next_id = get_adjacent_paragraph_ids(paragraph_id)
    
    # Build merged text
    parts = []
    
    # Add previous paragraph if exists
    if prev_id and prev_id in all_paragraphs:
        prev_text = all_paragraphs[prev_id]
        if prev_text:  # Only add if not empty
            parts.append(f"[Previous paragraph]: {prev_text}")
    
    # Add main paragraph
    parts.append(paragraph_text)
    
    # Add next paragraph if exists
    if next_id and next_id in all_paragraphs:
        next_text = all_paragraphs[next_id]
        if next_text:  # Only add if not empty
            parts.append(f"[Next paragraph]: {next_text}")
    
    return "\n\n".join(parts)

def load_paragraph_texts(paragraph_ids, load_adjacent=False):
    """
    Load paragraph texts from Parquet shards.
    
    Args:
        paragraph_ids: List of paragraph IDs to load
        load_adjacent: If True, also load adjacent paragraphs for context merging
    """
    from pathlib import Path
    import pyarrow.parquet as pq
    
    print(f"\nLoading {len(paragraph_ids)} paragraph texts...")
    if load_adjacent:
        print("Will also load adjacent paragraphs for context merging...")
    
    paragraph_texts = {}
    needed_ids = set(paragraph_ids)
    
    # If loading adjacent, expand the needed IDs
    if load_adjacent:
        expanded_ids = set(needed_ids)
        for pid in paragraph_ids:
            prev_id, next_id = get_adjacent_paragraph_ids(pid)
            if prev_id:
                expanded_ids.add(prev_id)
            if next_id:
                expanded_ids.add(next_id)
        print(f"Expanded to {len(expanded_ids)} IDs (including adjacent paragraphs)")
        needed_ids = expanded_ids
    
    # Find all shard directories
    shards_dir = Path(PARQUET_SHARDS_DIR)
    shard_dirs = sorted([d for d in shards_dir.iterdir() if d.is_dir() and d.name.startswith('shard=')])
    
    print(f"Found {len(shard_dirs)} parquet shards")
    
    # Search through shards
    for shard_dir in tqdm(shard_dirs, desc="Searching parquet shards"):
        # Find parquet files in this shard
        parquet_files = list(shard_dir.glob("*.parquet"))
        
        for pq_file in parquet_files:
            try:
                # Read parquet file
                table = pq.read_table(pq_file, columns=['paper_paragraph_id', 'paragraph_text'])
                df = table.to_pandas()
                
                # Filter to only needed IDs
                mask = df['paper_paragraph_id'].isin(needed_ids)
                matching_rows = df[mask]
                
                # Add to results
                for _, row in matching_rows.iterrows():
                    pid = row['paper_paragraph_id']
                    if pid not in paragraph_texts:
                        paragraph_texts[pid] = row['paragraph_text']
                
                # Stop early if found all
                if len(paragraph_texts) == len(needed_ids):
                    break
            except Exception as e:
                # Skip corrupted or incompatible files
                continue
        
        if len(paragraph_texts) == len(needed_ids):
            break
    
    print(f"Found {len(paragraph_texts)} / {len(needed_ids)} paragraph texts")
    
    # If loading adjacent, merge context for short paragraphs
    if load_adjacent:
        original_count = len(paragraph_ids)
        merged_count = 0
        for pid in paragraph_ids:
            if pid in paragraph_texts:
                original_text = paragraph_texts[pid]
                merged_text = merge_with_context(pid, original_text, paragraph_texts)
                if merged_text != original_text:
                    paragraph_texts[pid] = merged_text
                    merged_count += 1
        print(f"Merged context for {merged_count}/{original_count} short paragraphs")
    
    return paragraph_texts

def main(limit_questions=None):
    print("="*80)
    print("Loading FAISS index...")
    if limit_questions:
        print(f"Will process only first {limit_questions} questions for testing")
    print("="*80)
    
    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)
    print(f"Index loaded: {index.ntotal} vectors")
    
    # Move to GPU if available
    if faiss.get_num_gpus() > 0:
        print(f"Moving index to GPU (found {faiss.get_num_gpus()} GPUs)")
        res = faiss.StandardGpuResources()
        # Use Float16 lookup tables to reduce shared memory requirements
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
    
    # Load ID mapping
    print(f"\nLoading ID mapping from {ID_MAPPING_FILE}")
    id_mapping = np.load(ID_MAPPING_FILE, allow_pickle=True)
    print(f"Loaded {len(id_mapping)} IDs")
    print(f"First 5 IDs: {id_mapping[:5]}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize hierarchical structure
    patents = {}
    all_paragraph_ids = set()  # Track all paragraph IDs for text loading
    
    # Process each embedding file
    for emb_file, meta_file, bloom_level in EMBEDDING_FILES:
        print(f"\n{'='*80}")
        print(f"Processing: {bloom_level}")
        print(f"{'='*80}")
        
        # Load embeddings
        emb_path = os.path.join(EMBEDDINGS_DIR, emb_file)
        embeddings = np.load(emb_path)
        print(f"Loaded embeddings: {embeddings.shape}")
        
        # Load metadata
        meta_path = os.path.join(EMBEDDINGS_DIR, meta_file)
        df = pd.read_csv(meta_path)
        
        # Apply limit if specified
        if limit_questions:
            df = df.head(limit_questions)
            print(f"Limited to first {len(df)} questions for testing")
        else:
            print(f"Loaded metadata: {len(df)} questions")
        
        # Get embeddings for the questions we're processing
        embeddings_subset = embeddings[:len(df)]
        
        # Perform retrieval
        print(f"Searching for top {K} results per question...")
        distances, indices = index.search(embeddings_subset, K)
        
        # Process results
        print("Processing results...")
        for idx in tqdm(range(len(df)), desc="Questions"):
            row = df.iloc[idx]
            patent_id = str(row['patent_id'])
            question = row['question']
            
            # Get top K paragraph IDs
            top_k_indices = indices[idx]
            top_k_paragraph_ids = [id_mapping[i] for i in top_k_indices]
            
            # Extract unique paper IDs (preserving order)
            top_k_paper_ids = [extract_paper_id(pid) for pid in top_k_paragraph_ids]
            unique_papers = list(dict.fromkeys(top_k_paper_ids))
            
            # Calculate unique papers at different K values
            unique_at_10 = len(set([extract_paper_id(id_mapping[i]) for i in indices[idx][:10]]))
            unique_at_50 = len(set([extract_paper_id(id_mapping[i]) for i in indices[idx][:50]]))
            unique_at_100 = len(set([extract_paper_id(id_mapping[i]) for i in indices[idx][:100]]))
            
            # === Build Hierarchical Structure ===
            
            # Initialize patent entry if not exists
            if patent_id not in patents:
                patents[patent_id] = {
                    'cpc': row['cpc_class'],
                    'grant_year': None,  # Not available in current dataset
                    'patent_title': row['patent_title'],
                    'patent_abstract': row['patent_abstract'],
                    'question_types': {
                        'remembering': [],
                        'understanding': []
                    }
                }
            
            # Group top 10 paragraphs by paper
            papers_dict = defaultdict(list)
            for rank, para_id in enumerate(top_k_paragraph_ids[:10], start=1):
                paper_id = extract_paper_id(para_id)
                all_paragraph_ids.add(para_id)  # Track for text loading later
                papers_dict[paper_id].append({
                    'paragraph_id': para_id,
                    'rank': rank,
                    'has_paragraph': True,
                    'paragraph_text': None,  # Will be filled in next step
                    'contains_answer': None,  # Will be filled by verification script
                    'answer': None,  # Will be filled by verification script  
                    'paragraph_type': None  # Placeholder for future classification
                })
            
            # Convert to list of papers with metadata
            retrieved_papers = []
            for paper_id, paragraphs in papers_dict.items():
                retrieved_papers.append({
                    'paper_id': paper_id,
                    'metadata': {},  # Placeholder for paper metadata (title, year, etc.)
                    'paragraphs': sorted(paragraphs, key=lambda x: x['rank'])
                })
            
            # Sort papers by earliest paragraph rank
            retrieved_papers.sort(key=lambda x: min(p['rank'] for p in x['paragraphs']))
            
            # Add question to appropriate type
            question_entry = {
                'question': question,
                'retrieved_papers': retrieved_papers,
                'stats': {
                    'unique_papers@10': unique_at_10,
                    'unique_papers@50': unique_at_50,
                    'unique_papers@100': unique_at_100
                }
            }
            
            patents[patent_id]['question_types'][bloom_level].append(question_entry)
    
    # Skip loading paragraph texts - do this in separate job
    # Just initialize paragraph_text fields as None
    print(f"\n{'='*80}")
    print("Initializing paragraph text fields (load texts separately)...")
    print(f"{'='*80}")
    for patent_data in patents.values():
        for qtype in ['remembering', 'understanding']:
            for question in patent_data['question_types'][qtype]:
                for paper in question['retrieved_papers']:
                    for para in paper['paragraphs']:
                        para['paragraph_text'] = None
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    # Save hierarchical JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(patents, f, indent=2)
    print(f"✓ Saved hierarchical JSON to: {OUTPUT_JSON}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    total_questions = sum(
        len(p['question_types']['remembering']) + len(p['question_types']['understanding'])
        for p in patents.values()
    )
    
    print(f"Total patents: {len(patents)}")
    print(f"Total questions: {total_questions}")
    
    # Sample structure
    if patents:
        sample_patent_id = list(patents.keys())[0]
        sample_patent = patents[sample_patent_id]
        
        print(f"\nSample structure for patent {sample_patent_id}:")
        print(f"  CPC: {sample_patent['cpc']}")
        print(f"  Remembering questions: {len(sample_patent['question_types']['remembering'])}")
        print(f"  Understanding questions: {len(sample_patent['question_types']['understanding'])}")
        
        if sample_patent['question_types']['remembering']:
            q = sample_patent['question_types']['remembering'][0]
            print(f"\n  Sample question: {q['question'][:80]}...")
            print(f"  Retrieved papers: {len(q['retrieved_papers'])}")
            if q['retrieved_papers']:
                paper = q['retrieved_papers'][0]
                print(f"    Top paper: {paper['paper_id']}")
                print(f"    Paragraphs: {len(paper['paragraphs'])}")
    
    print(f"\n{'='*80}")
    print("Retrieval completed successfully!")
    print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FAISS retrieval (paragraph texts loaded separately)')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit to first N questions for testing (default: process all)')
    args = parser.parse_args()
    
    main(limit_questions=args.limit)
