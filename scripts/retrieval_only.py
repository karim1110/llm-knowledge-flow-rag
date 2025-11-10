"""
Retrieval only script for GPU job (no model loading, no internet required)
Loads pre-computed question embeddings and performs FAISS search
"""
import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm

# Configuration
INDEX_FILE = "indexes/faiss_qwen06_ivfpq.index"
ID_MAPPING_FILE = "indexes/patent_id_mapping_ivf.npy"
EMBEDDINGS_DIR = "data/question_generation/embeddings"
OUTPUT_DIR = "data/question_generation/retrieval_results"
K = 100  # Top K results

# Embedding files
EMBEDDING_FILES = [
    ("questions_remembering_embeddings.npy", "questions_remembering_metadata.csv", "remembering"),
    ("questions_understanding_embeddings.npy", "questions_understanding_metadata.csv", "understanding")
]

def extract_paper_id(paragraph_id):
    """Extract paper ID from paragraph ID (e.g., '2510268_p5' -> '2510268')"""
    return paragraph_id.split('_')[0]

def main():
    print("="*80)
    print("Loading FAISS index...")
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
    
    # Process each embedding file
    all_results = []
    
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
        print(f"Loaded metadata: {len(df)} questions")
        
        # Perform retrieval
        print(f"Searching for top {K} results per question...")
        distances, indices = index.search(embeddings, K)
        
        # Process results
        print("Processing results...")
        for idx in tqdm(range(len(df)), desc="Questions"):
            patent_id = df.iloc[idx]['patent_id']
            question = df.iloc[idx]['question']
            
            # Get top K paragraph IDs
            top_k_indices = indices[idx]
            top_k_paragraph_ids = [id_mapping[i] for i in top_k_indices]
            
            # Extract unique paper IDs
            top_k_paper_ids = [extract_paper_id(pid) for pid in top_k_paragraph_ids]
            unique_papers = list(dict.fromkeys(top_k_paper_ids))  # Preserve order
            
            # Calculate unique papers at different K values
            unique_at_10 = len(set([extract_paper_id(id_mapping[i]) for i in indices[idx][:10]]))
            unique_at_50 = len(set([extract_paper_id(id_mapping[i]) for i in indices[idx][:50]]))
            unique_at_100 = len(set([extract_paper_id(id_mapping[i]) for i in indices[idx][:100]]))
            
            all_results.append({
                'patent_id': patent_id,
                'question': question,
                'bloom_level': bloom_level,
                'top_10_paragraph_ids': '|'.join(top_k_paragraph_ids[:10]),
                'top_100_papers': '|'.join(unique_papers),
                'unique_papers@10': unique_at_10,
                'unique_papers@50': unique_at_50,
                'unique_papers@100': unique_at_100
            })
    
    # Save all results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(all_results)
    output_file = os.path.join(OUTPUT_DIR, "retrieval_results.csv")
    results_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(results_df)} results to: {output_file}")
    print(f"\nColumns: {list(results_df.columns)}")
    print(f"\nSample result:")
    print(results_df.iloc[0])
    print(f"\n{'='*80}")
    print("Retrieval completed successfully!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
