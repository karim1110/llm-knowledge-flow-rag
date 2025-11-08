#!/usr/bin/env python3
"""
FAISS IVF index builder for Qwen embeddings - Optimized for H100 GPU.

This script uses IVF (Inverted File) indexing with less aggressive compression
for a good balance between speed and accuracy on H100.

Input: /project/jevans/apto_data_engineering/personalFolders/nadav/arxiv_emb/qwen_06_embds_*
Output: indexes/faiss_qwen06_ivf.index + patent_id_mapping.npy
"""

from pathlib import Path
import numpy as np
import time
import re
import sys

# Config
EMB_BASE_DIR = Path('/project/jevans/apto_data_engineering/personalFolders/nadav/arxiv_emb')
INDEX_DIR = Path('indexes')
OUTPUT_NAME = 'faiss_qwen06_ivfpq.index'
ID_MAPPING_FILE = 'patent_id_mapping_ivf.npy'
USE_GPU = True  # Use GPU with IVF-PQ compression (works on both A100 and H100)
GPU_ID = 0

# IVF-PQ parameters - compressed for GPU memory efficiency
NLIST = 8192  # Number of clusters
TRAIN_SAMPLE = 1_000_000  # Training samples
NPROBE = 64  # Search clusters (increase for better recall)
M = 64  # Number of PQ subquantizers (must divide dimension: 1024/64=16)
NBITS = 8  # Bits per subquantizer (8 = 256 centroids)
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
    
    # Collect training samples
    print(f'\nCollecting {TRAIN_SAMPLE:,} training samples...')
    rnd = np.random.default_rng(1234)
    train_samples = []
    sample_count = 0
    
    for f in files:
        if sample_count >= TRAIN_SAMPLE:
            break
        X, _, success = load_chunk_with_ids(f)
        if not success:
            continue
        remaining = TRAIN_SAMPLE - sample_count
        take = min(remaining, len(X))
        if take < len(X):
            idx = rnd.choice(len(X), size=take, replace=False)
            train_samples.append(X[idx])
        else:
            train_samples.append(X)
        sample_count += take
        
        if len(train_samples) % 10 == 0:
            print(f'  Collected {sample_count:,}/{TRAIN_SAMPLE:,} samples')
    
    train_data = np.vstack(train_samples)
    print(f'Training data shape: {train_data.shape}')
    
    # Build IVF-PQ index (compressed for GPU memory)
    print(f'\nCreating IVF{NLIST},PQ{M}x{NBITS} index (compressed)...')
    quantizer = faiss.IndexFlatL2(d)
    index_base = faiss.IndexIVFPQ(quantizer, d, NLIST, M, NBITS)
    
    print('Training index...')
    t_train_start = time.time()
    index_base.train(train_data)
    t_train_end = time.time()
    print(f'Training completed in {t_train_end - t_train_start:.1f}s')
    
    index_base.nprobe = NPROBE
    print(f'Set nprobe to {NPROBE}')
    
    # Wrap with IDMap to preserve IDs
    index = faiss.IndexIDMap(index_base)
    
    # GPU transfer
    use_gpu = USE_GPU and faiss.get_num_gpus() > 0
    if use_gpu:
        print(f'\nUsing GPU {GPU_ID}')
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index_worker = faiss.index_cpu_to_gpu(res, GPU_ID, index, co)
    else:
        print('\nGPU not available, using CPU')
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
    
    # Move back to CPU and save
    print('\nSaving index to disk...')
    if use_gpu:
        index_out = faiss.index_gpu_to_cpu(index_worker)
    else:
        index_out = index_worker
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / OUTPUT_NAME
    faiss.write_index(index_out, str(index_path))
    print(f'Index saved: {index_path}')
    print(f'Index size: {index_path.stat().st_size / (1024**3):.2f} GB')
    
    # Save ID mapping: faiss_id -> paper_paragraph_id
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
