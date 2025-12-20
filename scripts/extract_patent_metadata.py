import pandas as pd
import glob
import json
import os

def extract_patent_metadata(patents_dir, target_ids, output_path):
    print(f"Extracting metadata for {len(target_ids)} patents...")
    
    patent_files = glob.glob(os.path.join(patents_dir, "*.parquet"))
    metadata_map = {}
    
    # Convert target_ids to integers for matching with parquet
    target_ids_int = set()
    for tid in target_ids:
        try:
            target_ids_int.add(int(tid))
        except ValueError:
            pass
    
    for i, f in enumerate(patent_files):
        print(f"Processing {i+1}/{len(patent_files)}: {os.path.basename(f)}")
        df = pd.read_parquet(f, columns=['patent_id', 'patentGrant_year', 'cpc_class', 'cpc_section'])
        
        # Filter for target IDs
        matches = df[df['patent_id'].isin(target_ids_int)]
        
        for _, row in matches.iterrows():
            pid_str = str(row['patent_id'])
            metadata_map[pid_str] = {
                'grant_year': int(row['patentGrant_year']) if pd.notnull(row['patentGrant_year']) else None,
                'cpc_class': row['cpc_class'],
                'cpc_section': row['cpc_section']
            }
        
        if len(metadata_map) >= len(target_ids):
            # Optimization: stop if we found all
            # But target_ids might come from multiple sources, so maybe not
            pass

    print(f"Found metadata for {len(metadata_map)} patents.")
    with open(output_path, 'w') as f:
        json.dump(metadata_map, f)

def get_patent_ids(file_path):
    import subprocess
    print(f"Extracting IDs from {file_path} using grep...")
    cmd = f"grep -oP '^  \"\\d+\"' {file_path} | tr -d ' \" '"
    result = subprocess.check_output(cmd, shell=True).decode('utf-8')
    ids = set(result.strip().split('\n'))
    print(f"Found {len(ids)} IDs.")
    return ids

def main():
    # Get all unique patent IDs from the merged files
    target_ids = set()
    files = [
        'data/question_generation/retrieval_results/retrieval_results_understanding_merged_all_patents_verified_all_shards.json',
        'data/question_generation/retrieval_results/retrieval_results_remembering_merged_all_patents_verified_all_shards.json'
    ]
    
    for f_path in files:
        if os.path.exists(f_path):
            target_ids.update(get_patent_ids(f_path))
    
    print(f"Total unique patents: {len(target_ids)}")
    
    extract_patent_metadata(
        'data/patent_sample/',
        target_ids,
        'data/patent_metadata_map.json'
    )

if __name__ == "__main__":
    main()
