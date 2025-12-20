import pyarrow.parquet as pq
import json
import pandas as pd
import os

def extract_metadata(ids_path, shards_path, arxiv_metadata_path, output_path):
    with open(ids_path, 'r') as f:
        target_corpus_ids = set(json.load(f))
    
    print(f"Target Corpus IDs: {len(target_corpus_ids)}")
    
    # Step 1: Map corpusid to arxiv id using shards
    corpus_to_arxiv = {}
    dataset_shards = pq.ParquetDataset(shards_path)
    print(f"Scanning {len(dataset_shards.fragments)} shards for arXiv ID mapping...")
    
    for i, fragment in enumerate(dataset_shards.fragments):
        table = fragment.to_table(columns=['corpusid', 'arxiv'])
        df = table.to_pandas()
        
        # Filter for target corpus IDs
        # corpusid in parquet is int64, target_corpus_ids are strings
        df['corpusid_str'] = df['corpusid'].astype(str)
        matches = df[df['corpusid_str'].isin(target_corpus_ids)]
        
        for _, row in matches.iterrows():
            if row['arxiv']:
                corpus_to_arxiv[row['corpusid_str']] = row['arxiv']
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(dataset_shards.fragments)} shards. Found {len(corpus_to_arxiv)} arXiv IDs.")

    print(f"Found {len(corpus_to_arxiv)} arXiv IDs for {len(target_corpus_ids)} corpus IDs.")
    
    # Step 2: Get metadata from arxiv_metadata.parquet using arxiv ids
    target_arxiv_ids = set(corpus_to_arxiv.values())
    arxiv_to_metadata = {}
    
    dataset_arxiv = pq.ParquetDataset(arxiv_metadata_path)
    print(f"Scanning {len(dataset_arxiv.fragments)} arxiv metadata fragments...")
    
    for i, fragment in enumerate(dataset_arxiv.fragments):
        table = fragment.to_table(columns=['id', 'categories', 'update_date'])
        df = table.to_pandas()
        
        matches = df[df['id'].isin(target_arxiv_ids)]
        
        for _, row in matches.iterrows():
            year = str(row['update_date'])[:4] if row['update_date'] else None
            primary_cat = row['categories'].split(' ')[0] if row['categories'] else None
            
            arxiv_to_metadata[row['id']] = {
                'year': year,
                'category': primary_cat,
                'all_categories': row['categories']
            }
        
        if i % 5 == 0:
            print(f"Processed {i}/{len(dataset_arxiv.fragments)} fragments. Found {len(arxiv_to_metadata)} metadata matches.")

    # Combine
    final_metadata_map = {}
    for corpus_id in target_corpus_ids:
        arxiv_id = corpus_to_arxiv.get(corpus_id)
        if arxiv_id and arxiv_id in arxiv_to_metadata:
            final_metadata_map[corpus_id] = arxiv_to_metadata[arxiv_id]
            final_metadata_map[corpus_id]['arxiv_id'] = arxiv_id
        else:
            final_metadata_map[corpus_id] = {
                'year': None,
                'category': None,
                'all_categories': None,
                'arxiv_id': arxiv_id
            }

    print(f"Finished. Found metadata for {len([v for v in final_metadata_map.values() if v['year']])} papers.")
    
    with open(output_path, 'w') as f:
        json.dump(final_metadata_map, f)

if __name__ == "__main__":
    extract_metadata(
        'data/unique_paper_ids.json',
        '/project/jevans/apto_data_engineering/data/arxiv/s2_arxiv_full_text_grobid/shards_partitioned',
        '/project/jevans/apto_data_engineering/data/arxiv/arxiv_metadata.parquet',
        'data/paper_metadata_map.json'
    )
