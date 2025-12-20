"""
Load paragraph texts from parquet using Spark (much faster than PyArrow)
Reads retrieval results JSON and fills in paragraph texts
Separate job to avoid doing this on login node
"""
import os
import json
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Configuration
PARQUET_SHARDS_DIR = "/project/jevans/apto_data_engineering/data/arxiv/s2_arxiv_full_text_grobid/shards_partitioned"
INPUT_JSON = "data/question_generation/retrieval_results/retrieval_results_hierarchical.json"
OUTPUT_JSON = None  # Will be set based on merge_context flag

def extract_paragraph_number(paragraph_id):
    """Extract paragraph number from ID (e.g., '2510268_p5' -> 5)"""
    try:
        return int(paragraph_id.split('_p')[1])
    except (IndexError, ValueError):
        return None

def get_adjacent_paragraph_ids(paragraph_id):
    """Get IDs of previous and next paragraphs."""
    parts = paragraph_id.split('_p')
    if len(parts) != 2:
        return None, None
    
    paper_id = parts[0]
    try:
        para_num = int(parts[1])
    except ValueError:
        return None, None
    
    prev_id = f"{paper_id}_p{para_num - 1}" if para_num > 0 else None
    next_id = f"{paper_id}_p{para_num + 1}"
    
    return prev_id, next_id

def merge_with_context(paragraph_id, paragraph_text, all_paragraphs, min_length=200):
    """Merge paragraph with adjacent paragraphs if it's too short."""
    if not paragraph_text or len(paragraph_text) >= min_length:
        return paragraph_text
    
    prev_id, next_id = get_adjacent_paragraph_ids(paragraph_id)
    parts = []
    
    if prev_id and prev_id in all_paragraphs:
        prev_text = all_paragraphs[prev_id]
        if prev_text:
            parts.append(f"[Previous paragraph]: {prev_text}")
    
    parts.append(paragraph_text)
    
    if next_id and next_id in all_paragraphs:
        next_text = all_paragraphs[next_id]
        if next_text:
            parts.append(f"[Next paragraph]: {next_text}")
    
    return "\n\n".join(parts)

def main(merge_context=False, input_json=None, output_json=None):
    if input_json is None:
        input_json = INPUT_JSON

    # Set default output based on merge_context flag
    if output_json is None:
        if merge_context:
            output_json = input_json.replace('.json', '_merged.json')
        else:
            output_json = input_json.replace('.json', '_baseline.json')
    
    print("="*80)
    print("Loading Paragraph Texts with Spark")
    if merge_context:
        print("Context merging ENABLED")
        print(f"Output: {output_json}")
    else:
        print("Baseline (no merging)")
        print(f"Output: {output_json}")
    print("="*80)
    
    # Load retrieval results
    print(f"\nLoading retrieval results from {input_json}...")
    with open(input_json, 'r') as f:
        patents = json.load(f)
    
    # Collect all paragraph IDs
    all_paragraph_ids = set()
    for patent_data in patents.values():
        for qtype in ['remembering', 'understanding']:
            for question in patent_data['question_types'][qtype]:
                for paper in question['retrieved_papers']:
                    for para in paper['paragraphs']:
                        all_paragraph_ids.add(para['paragraph_id'])
    
    print(f"Found {len(all_paragraph_ids)} unique paragraph IDs")
    
    # If merging context, expand to include adjacent paragraphs
    if merge_context:
        expanded_ids = set(all_paragraph_ids)
        for pid in list(all_paragraph_ids):
            prev_id, next_id = get_adjacent_paragraph_ids(pid)
            if prev_id:
                expanded_ids.add(prev_id)
            if next_id:
                expanded_ids.add(next_id)
        print(f"Expanded to {len(expanded_ids)} IDs (including adjacent)")
        needed_ids = list(expanded_ids)
    else:
        needed_ids = list(all_paragraph_ids)
    
    # Initialize Spark
    print("\nInitializing Spark...")
    master = os.environ.get("MASTER", "local[*]")
    spark = SparkSession.builder \
        .appName("LoadParagraphTexts") \
        .master(master) \
        .config("spark.driver.memory", "100g") \
        .config("spark.executor.memory", "20g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    try:
        # Read all parquet files
        print(f"Reading parquet files from {PARQUET_SHARDS_DIR}...")
        df = spark.read.parquet(f"{PARQUET_SHARDS_DIR}/*/")
        
        # Filter to needed IDs
        print(f"Filtering to {len(needed_ids)} paragraph IDs...")
        df_filtered = df.filter(col("paper_paragraph_id").isin(needed_ids))
        
        # Collect results
        print("Collecting paragraph texts...")
        results = df_filtered.select("paper_paragraph_id", "paragraph_text").collect()
        
        # Convert to dict
        paragraph_texts = {row.paper_paragraph_id: row.paragraph_text for row in results}
        print(f"Loaded {len(paragraph_texts)} / {len(needed_ids)} paragraphs")
        
        # Merge context if needed
        if merge_context:
            print("\nMerging context for short paragraphs...")
            merged_count = 0
            for pid in all_paragraph_ids:
                if pid in paragraph_texts:
                    original_text = paragraph_texts[pid]
                    merged_text = merge_with_context(pid, original_text, paragraph_texts)
                    if merged_text != original_text:
                        paragraph_texts[pid] = merged_text
                        merged_count += 1
            print(f"Merged context for {merged_count}/{len(all_paragraph_ids)} paragraphs")
        
        # Fill in paragraph texts
        print("\nFilling paragraph texts into results...")
        for patent_data in patents.values():
            for qtype in ['remembering', 'understanding']:
                for question in patent_data['question_types'][qtype]:
                    for paper in question['retrieved_papers']:
                        for para in paper['paragraphs']:
                            para['paragraph_text'] = paragraph_texts.get(para['paragraph_id'], None)
        
        # Save updated results
        print(f"\nSaving updated results to {output_json}...")
        with open(output_json, 'w') as f:
            json.dump(patents, f, indent=2)
        
        print("\n" + "="*80)
        print("✓ Paragraph texts loaded successfully!")
        print("="*80)
        
    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load paragraph texts using Spark')
    parser.add_argument('--merge-context', action='store_true',
                      help='Merge adjacent paragraphs for short texts')
    parser.add_argument('--input', default=INPUT_JSON,
                      help='Input JSON file with retrieval results')
    parser.add_argument('--output', default=OUTPUT_JSON,
                      help='Output JSON file')
    args = parser.parse_args()
    
    main(merge_context=args.merge_context, 
         input_json=args.input, 
         output_json=args.output)
