#!/usr/bin/env python3
import json
import glob
import os
import argparse

def merge_shards(input_base, num_shards):
    """Merge sharded verification results back into one file."""
    merged_data = {}
    total_shards_found = 0
    
    print(f"Merging {num_shards} shards for {input_base}...")
    
    for i in range(num_shards):
        shard_file = input_base.replace('.json', f'_verified_shard_{i}.json')
        if not os.path.exists(shard_file):
            # Try checkpoint if final is not there
            shard_file = input_base.replace('.json', f'_verified_shard_{i}_checkpoint.json')
            
        if os.path.exists(shard_file):
            print(f"Loading shard {i}: {shard_file}")
            with open(shard_file, 'r') as f:
                shard_data = json.load(f)
                merged_data.update(shard_data)
            total_shards_found += 1
        else:
            print(f"WARNING: Shard {i} not found!")

    output_file = input_base.replace('.json', '_verified_all_shards.json')
    print(f"Saving merged data ({len(merged_data)} patents) to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Done! Found {total_shards_found}/{num_shards} shards.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Original input JSON path')
    parser.add_argument('--shards', type=int, default=10, help='Number of shards')
    args = parser.parse_args()
    
    merge_shards(args.input, args.shards)
