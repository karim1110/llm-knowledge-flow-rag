"""
Retrieval only script for GPU job (no model loading, no internet required)
Loads pre-computed question embeddings and performs FAISS search
Outputs hierarchical JSON structure organized by patent -> question type -> questions -> papers
"""
import os
import json
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from collections import defaultdict

# Configuration
INDEX_FILE = "indexes/faiss_qwen06_ivfpq.index"
ID_MAPPING_FILE = "indexes/patent_id_mapping_ivf.npy"
EMBEDDINGS_DIR = "data/question_generation/embeddings"
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
    
    # Initialize hierarchical structure
    patents = {}
    
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
        print(f"Loaded metadata: {len(df)} questions")
        
        # Perform retrieval
        print(f"Searching for top {K} results per question...")
        distances, indices = index.search(embeddings, K)
        
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
                papers_dict[paper_id].append({
                    'paragraph_id': para_id,
                    'rank': rank,
                    'has_paragraph': True,
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
    main()
