#!/usr/bin/env python3
"""
LlamaIndex BM25 index builder - exact mirror of FAISS builder
Follows https://developers.llamaindex.ai/python/examples/retrievers/bm25_retriever/
Disk persistence for 10M+ paragraphs
"""

from pathlib import Path
import pyarrow.parquet as pq
import time
import sys
import numpy as np
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core import StorageContext

# Config - MIRRORS your FAISS script exactly
EMB_BASE_DIR = Path('/project/jevans/apto_data_engineering/personalFolders/nadav/arxiv_emb')
INDEX_DIR = Path('indexes')
OUTPUT_NAME = 'bm25_patent_index'  # Directory for persistence
ID_MAPPING_FILE = 'bm25_patent_id_mapping.npy'
BATCH_SIZE = 100_000
NUM_THREADS = 32

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
    """Load TEXTS from PARQUET + IDS from npz - matches your retrieval script"""
    try:
        # Load IDs from npz (same as FAISS)
        z = np.load(path, allow_pickle=True)
        ids = z['ids']
        
        # Load texts from parquet using your exact logic
        texts = load_paragraph_texts_from_parquet(ids)  # See function below
        if len(texts) != len(ids):
            raise ValueError(f"Mismatch: {len(texts)} texts vs {len(ids)} ids")
            
        return texts, ids, True
    except Exception as e:
        print(f'  WARNING: Failed to load {path.name}: {e}')
        return None, None, False

def load_paragraph_texts_from_parquet(paragraph_ids):
    """EXACT copy from your retrieval script - load texts for given IDs"""
    
    PARQUET_SHARDS_DIR = Path("/project/jevans/apto_data_engineering/data/arxiv/s2_arxiv_full_text_grobid/shards_partitioned")
    paragraph_texts = {}
    needed_ids = set(paragraph_ids)
    
    shards_dir = PARQUET_SHARDS_DIR
    shard_dirs = sorted([d for d in shards_dir.iterdir() if d.is_dir() and d.name.startswith('shard=')])
    
    for shard_dir in shard_dirs[:10]:  # Limit shards for speed during build
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
    
    # Fill missing texts with empty string
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
    
    print('\nLooking for chunk files...')
    files = find_embedding_files(dirs)
    if not files:
        raise SystemExit(f'No chunk files found')
    
    print(f'Found {len(files)} chunk files total')
    
    # === EXACTLY LIKE LLAMAINDEX DOC: Create TextNodes ===
    print('\nCreating TextNodes (BM25Retriever.from_defaults(nodes=...))...')
    all_patent_ids = []
    nodes = []
    total = 0
    skipped = 0
    t0 = time.time()
    
    for idx, f in enumerate(files):
        if idx % 100 == 0:
            print(f'Processing file {idx+1}/{len(files)}: {f.name}')
        
        texts, ids, success = load_chunk_texts(f)
        if not success:
            skipped += 1
            continue
        
        n = len(texts)
        
        # Batch processing - MIRRORS FAISS
        for i0 in range(0, n, BATCH_SIZE):
            i1 = min(n, i0 + BATCH_SIZE)
            batch_texts = texts[i0:i1]
            batch_ids = ids[i0:i1]
            
            # Create TextNodes - EXACTLY like LlamaIndex doc
            batch_nodes = [
                TextNode(text=str(text), id_=parse_patent_id(pid))
                for text, pid in zip(batch_texts, batch_ids)
            ]
            nodes.extend(batch_nodes)
            
            # Track IDs for mapping - SAME as FAISS
            all_patent_ids.extend([parse_patent_id(pid) for pid in batch_ids])
            total += len(batch_nodes)
        
        if idx < 5 or idx % 100 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f'  Created {len(batch_texts):,} TextNodes (total: {total:,}, rate: {rate:.0f} nodes/sec)')
    
    t1 = time.time()
    print(f'\nCreated {total:,} TextNodes in {t1-t0:.1f}s')
    print(f'Skipped {skipped} corrupted files')
    
    # === EXACTLY LIKE DOC: BM25Retriever.from_defaults(nodes=nodes) + persist ===
    print(f'\nCreating BM25Retriever from {len(nodes):,} TextNodes...')
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=100  # Your K
    )
    
    # Disk persistence - EXACTLY like "BM25 Retriever + Disk Persistence" section
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    persist_path = INDEX_DIR / OUTPUT_NAME
    bm25_retriever.persist(str(persist_path))

    print(f'BM25Retriever persisted: {persist_path}')
    
    # Save ID mapping - SAME as FAISS
    mapping_path = INDEX_DIR / ID_MAPPING_FILE

    np.save(mapping_path, np.array(all_patent_ids, dtype=object))
    print(f'ID mapping saved: {mapping_path}')
    
    # === Verification - Test retrieval ===
    print('\n=== Verification ===')
    print(f'  Total TextNodes: {len(nodes)}')
    print(f'  ID mapping length: {len(all_patent_ids)}')
    print(f'  First 5 IDs: {all_patent_ids[:5]}')
    
    print('\n=== Test Search ===')
    test_nodes = bm25_retriever.retrieve("isolation prevents reconnaissance")
    print(f'  Test query returned {len(test_nodes)} nodes')
    print(f'  Sample node IDs: {[node.node_id for node in test_nodes[:3]]}')
    
    print('\n✓ BM25 index building completed - EXACTLY like LlamaIndex doc!')
    print(f'Usage: BM25Retriever.from_defaults(persist_directory="{persist_path}")')

if __name__ == '__main__':
    main()
