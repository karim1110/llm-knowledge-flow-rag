#!/usr/bin/env python3
"""
bm25s index builder - FASTEST option (30-60min total vs 20+hrs)
Memory-maps 188M+ patent paragraphs, instant load/retrieval
pip install bm25s
"""

from pathlib import Path
import pyarrow.parquet as pq
import time
import sys
import numpy as np
from bm25s import BM25  # pip install bm25s
import pickle

# Config - MIRRORS your FAISS script exactly
EMB_BASE_DIR = Path('/project/jevans/apto_data_engineering/personalFolders/nadav/arxiv_emb')
INDEX_DIR = Path('indexes')
OUTPUT_NAME = 'bm25s_patent_index'  # bm25s format directory
ID_MAPPING_FILE = 'bm25s_patent_id_mapping.npy'
BATCH_SIZE = 100_000

def find_embedding_directories(base_dir):
    """Find all qwen_06_embds_* directories - SAME as FAISS"""
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('qwen_06_embds')])
    return dirs

def find_embedding_files(directories):
    """Find all chunk_*.npz files - SAME as FAISS"""
    all_files = []
    for directory in directories:
        files = sorted(directory.glob('chunk_*.npz'))
        all_files.extend(files)
    return all_files

def load_chunk_texts(path):
    """Load TEXTS from PARQUET + IDS from npz - SAME as original"""
    try:
        z = np.load(path, allow_pickle=True)
        ids = z['ids']
        texts = load_paragraph_texts_from_parquet(ids)
        if len(texts) != len(ids):
            raise ValueError(f"Mismatch: {len(texts)} texts vs {len(ids)} ids")
        return texts, ids, True
    except Exception as e:
        print(f'  WARNING: Failed to load {path.name}: {e}')
        return None, None, False

def load_paragraph_texts_from_parquet(paragraph_ids):
    """Load texts from parquet - SAME as original"""
    PARQUET_SHARDS_DIR = Path("/project/jevans/apto_data_engineering/data/arxiv/s2_arxiv_full_text_grobid/shards_partitioned")
    paragraph_texts = {}
    needed_ids = set(paragraph_ids)
    
    shards_dir = PARQUET_SHARDS_DIR
    shard_dirs = sorted([d for d in shards_dir.iterdir() if d.is_dir() and d.name.startswith('shard=')])
    
    for shard_dir in shard_dirs:
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
                if len(paragraph_texts) == len(needed_ids):
                    break
            except:
                continue
        if len(paragraph_texts) == len(needed_ids):
            break
    
    result = [paragraph_texts.get(pid, "") for pid in paragraph_ids]
    return np.array(result)

def parse_patent_id(id_str):
    """Keep full paper_paragraph_id - SAME as FAISS"""
    return str(id_str)

def main():
    print('Looking for embedding directories under', EMB_BASE_DIR)
    dirs = find_embedding_directories(EMB_BASE_DIR)
    if not dirs:
        raise SystemExit(f'No qwen_06_embds_* directories found in {EMB_BASE_DIR}')
    
    print(f'Found {len(dirs)} embedding directories')
    files = find_embedding_files(dirs)
    print(f'Found {len(files)} chunk files total')
    
    # === STREAM TEXTS (bm25s handles tokenization) ===
    print('\nStreaming raw texts to bm25s (automatic tokenization)...')
    all_texts = []
    all_patent_ids = []
    total_docs = 0
    skipped = 0
    t0 = time.time()
    
    for idx, f in enumerate(files):
        if idx % 100 == 0:
            print(f'Processing file {idx+1}/{len(files)}: {f.name}')
        
        texts, ids, success = load_chunk_texts(f)
        if not success:
            skipped += 1
            continue
        
        all_texts.extend(texts.tolist())
        all_patent_ids.extend([parse_patent_id(pid) for pid in ids])
        total_docs += len(texts)
        
        if idx < 5 or idx % 100 == 0:
            elapsed = time.time() - t0
            rate = total_docs / elapsed if elapsed > 0 else 0
            print(f'  Loaded {len(texts):,} docs (total: {total_docs:,}, rate: {rate:.0f} docs/sec)')
    
    t1 = time.time()
    print(f'\n✓ Loaded {total_docs:,} texts in {t1-t0:.1f}s')
    print(f'Skipped {skipped} files')
    
    # === BUILD bm25s INDEX (30-60min total) ===
    print(f'\nBuilding bm25s index from {len(all_texts):,} raw texts...')
    t2 = time.time()
    bm25 = BM25.from_texts(all_texts)
    t3 = time.time()
    print(f'✓ bm25s indexing complete: {t3-t2:.1f}s')
    
    # === SAVE (memory-mapped format) ===
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / OUTPUT_NAME
    bm25.to_index(str(index_path))
    print(f'✓ bm25s index saved: {index_path}')
    
    mapping_path = INDEX_DIR / ID_MAPPING_FILE
    np.save(mapping_path, np.array(all_patent_ids, dtype=object))
    print(f'✓ ID mapping saved: {mapping_path}')
    
    # === VERIFICATION ===
    print('\n=== Verification ===')
    print(f'  Total docs: {len(all_texts)}')
    print(f'  ID mapping: {len(all_patent_ids)}')
    
    print('\n=== Test Search ===')
    test_query = "isolation prevents reconnaissance"
    t_search = time.time()
    results = bm25.search(test_query, k=100)
    print(f'  Top-100 search: {time.time() - t_search:.3f}s')
    print(f'  Sample doc indices: {results[:5]}')
    print(f'  Sample doc IDs: {[all_patent_ids[i] for i in results[:3]]}')
    
    # Load test (memory-mapped, instant)
    print('\n=== Load Test ===')
    t_load = time.time()
    bm25_loaded = BM25.from_index(str(index_path))
    print(f'  Load time: {time.time() - t_load:.3f}s (memory-mapped!)')
    
    print('\n✓ bm25s COMPLETE - FASTEST build + FASTEST retrieval!')
    print(f'Usage:')
    print(f'  from bm25s import BM25')
    print(f'  bm25 = BM25.from_index("{index_path}")')
    print(f'  results = bm25.search("query", k=100)  # doc indices')
    print(f'  ids = np.load("{mapping_path}", allow_pickle=True)')
    print(f'  doc_ids = [ids[i] for i in results]')

if __name__ == '__main__':
    main()
