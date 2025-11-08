#!/usr/bin/env python3
"""
FAISS index builder for Qwen embeddings - CPU version for large datasets.

This script:
- Loads embeddings from multiple qwen_06_embds_* directories
- Uses Flat index for maximum accuracy (built on CPU to avoid GPU memory limits)
- Preserves patent IDs for retrieval
- Handles corrupted .npz files gracefully

Note: Building on CPU to avoid GPU memory limits. A flat index for millions of 
1024-dim vectors requires ~50GB+ GPU memory. The saved index can still be loaded 
to GPU for fast searches later.

Input: /project/jevans/apto_data_engineering/personalFolders/nadav/arxiv_emb/qwen_06_embds_*
Output: indexes/faiss_qwen06_flat.index + patent_id_mapping.npy
"""

from pathlib import Path
import numpy as np
import time
import re
import sys

# Config
EMB_BASE_DIR = Path('/project/jevans/apto_data_engineering/personalFolders/nadav/arxiv_emb')
INDEX_DIR = Path('indexes')
OUTPUT_NAME = 'faiss_qwen06_flat_cpu.index'
ID_MAPPING_FILE = 'patent_id_mapping_cpu.npy'
USE_GPU = False  # Build on CPU to avoid memory limits
NUM_THREADS = 32  # Use all available CPU cores
BATCH_SIZE = 100_000

def find_embedding_directories(base_dir):
    """Find all qwen_06_embds_* directories"""
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('qwen_06_embds')])
    return dirs

def find_embedding_files(directories):
    """Find all chunk_*.npz files across all directories"""
    all_files = []
    for directory in directories:
        files = sorted(directory.glob('chunk_*.npz'))
        all_files.extend(files)
    return all_files

def load_chunk_with_ids(path):
    """Load embeddings and IDs from npz file, handling corrupted files"""
    try:
        z = np.load(path, allow_pickle=True)
        embeddings = z['embeddings'].astype('float32')
        ids = z['ids']
        return embeddings, ids, True
    except Exception as e:
        print(f'  WARNING: Failed to load {path.name}: {e}')
        return None, None, False

def parse_patent_id(id_str):
    """Keep full paper_paragraph_id (e.g., '2510268_p0') as string"""
    return str(id_str)

def main():
    import faiss
    
    print('Looking for embedding directories under', EMB_BASE_DIR)
    dirs = find_embedding_directories(EMB_BASE_DIR)
    if not dirs:
        raise SystemExit(f'No qwen_06_embds_* directories found in {EMB_BASE_DIR}')
    
    print(f'Found {len(dirs)} embedding directories')
    print(f'Directories: {[d.name for d in dirs[:5]]}... (showing first 5)')
    
    print('\nLooking for chunk files...')
    files = find_embedding_files(dirs)
    if not files:
        raise SystemExit(f'No chunk files found in embedding directories')
    
    print(f'Found {len(files)} chunk files total')
    
    # Probe dimension
    print('\nProbing embedding dimension...')
    emb0, _, success = load_chunk_with_ids(files[0])
    if not success:
        raise SystemExit('Failed to load first file to probe dimension')
    
    d = emb0.shape[1]
    print(f'Embedding dimension: {d}')
    
    # Create Flat index (no compression)
    print(f'\nCreating IndexFlatL2 with dimension {d}')
    index_base = faiss.IndexFlatL2(d)
    
    # Wrap with IDMap to preserve IDs
    index = faiss.IndexIDMap(index_base)
    
    # Build on CPU to avoid GPU memory limits
    print(f'Building index on CPU with {NUM_THREADS} threads')
    print('(Flat index for millions of vectors would exceed GPU memory)')
    print('Note: The saved index can be loaded to GPU for searches later')
    faiss.omp_set_num_threads(NUM_THREADS)
    index_worker = index
    
    # Add vectors with IDs
    all_patent_ids = []
    faiss_ids = []
    total = 0
    skipped = 0
    t0 = time.time()
    
    print(f'\nAdding embeddings to index...')
    for idx, f in enumerate(files):
        if idx % 100 == 0:
            print(f'Processing file {idx+1}/{len(files)}: {f.name}')
        
        X, ids, success = load_chunk_with_ids(f)
        if not success:
            skipped += 1
            continue
        
        n = len(X)
        
        # Process in batches
        for i0 in range(0, n, BATCH_SIZE):
            i1 = min(n, i0 + BATCH_SIZE)
            batch_emb = X[i0:i1]
            batch_ids = ids[i0:i1]
            
            # Extract patent IDs
            patent_ids = [parse_patent_id(id_str) for id_str in batch_ids]
            
            # Use sequential FAISS IDs
            batch_faiss_ids = np.arange(total, total + len(batch_emb), dtype=np.int64)
            
            index_worker.add_with_ids(batch_emb, batch_faiss_ids)
            
            all_patent_ids.extend(patent_ids)
            faiss_ids.extend(batch_faiss_ids.tolist())
            total += len(batch_emb)
        
        if idx < 5 or idx % 100 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f'  Added {n:,} vectors (total: {total:,}, rate: {rate:.0f} vecs/sec)')
    
    t1 = time.time()
    print(f'\nAdded {total:,} vectors in {t1-t0:.1f}s')
    print(f'Skipped {skipped} corrupted files')
    
    # Save index
    print('\nSaving index to disk...')
    index_out = index_worker
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / OUTPUT_NAME
    faiss.write_index(index_out, str(index_path))
    print(f'Index saved: {index_path}')
    print(f'Index size: {index_path.stat().st_size / (1024**3):.2f} GB')
    
    # Save ID mapping: faiss_id -> paper_paragraph_id (as strings)
    mapping_path = INDEX_DIR / ID_MAPPING_FILE
    np.save(mapping_path, np.array(all_patent_ids, dtype=object))
    print(f'ID mapping saved: {mapping_path} ({len(all_patent_ids):,} IDs)')
    
    # Verification
    print('\n=== Verification ===')
    print(f'  Index total: {index_out.ntotal:,}')
    print(f'  Mapping length: {len(all_patent_ids):,}')
    print(f'  First 5 paper_paragraph_ids: {all_patent_ids[:5]}')
    print(f'  Last 5 paper_paragraph_ids: {all_patent_ids[-5:]}')
    print(f'  ID type: {type(all_patent_ids[0])}')  # Should be <class 'str'>    # Test search
    print('\n=== Test Search ===')
    test_vec = np.random.randn(1, d).astype('float32')
    D, I = index_out.search(test_vec, 5)
    print(f'  Query vector shape: {test_vec.shape}')
    print(f'  Top 5 distances: {D[0]}')
    print(f'  Top 5 FAISS IDs: {I[0]}')
    print(f'  Top 5 paper_paragraph_ids: {[all_patent_ids[i] for i in I[0]]}')
    
    print('\n✓ Index building completed successfully!')
    print(f'✓ Paragraph-level IDs preserved (format: "paperID_pN")')

if __name__ == '__main__':
    main()
