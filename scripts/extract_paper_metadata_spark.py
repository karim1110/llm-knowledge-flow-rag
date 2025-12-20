import json
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("ExtractPaperMetadata") \
        .config("spark.driver.memory", "100g") \
        .config("spark.executor.memory", "100g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "20g") \
        .getOrCreate()

    # Paths
    ids_path = 'data/unique_paper_ids.json'
    shards_path = '/project/jevans/apto_data_engineering/data/arxiv/s2_arxiv_full_text_grobid/shards_partitioned'
    arxiv_metadata_path = '/project/jevans/apto_data_engineering/data/arxiv/arxiv_metadata.parquet'
    output_path = 'data/paper_metadata_map.json'

    print(f"Loading target IDs from {ids_path}...")
    with open(ids_path, 'r') as f:
        target_ids = json.load(f)
    
    # Convert target IDs to a DataFrame for joining
    # Note: corpusid in parquet is long, so we convert target_ids to long
    target_ids_df = spark.createDataFrame([(int(i),) for i in target_ids], ["target_corpusid"])
    
    print(f"Loading shards from {shards_path}...")
    shards_df = spark.read.parquet(shards_path).select("corpusid", "arxiv")
    
    # Join to get arXiv IDs for our target corpus IDs
    # Using broadcast join since target_ids_df is relatively small (~387k rows)
    mapped_df = shards_df.join(broadcast(target_ids_df), shards_df.corpusid == target_ids_df.target_corpusid) \
        .select("corpusid", "arxiv") \
        .filter(col("arxiv").isNotNull()) \
        .distinct()
    
    print(f"Loading arXiv metadata from {arxiv_metadata_path}...")
    metadata_df = spark.read.parquet(arxiv_metadata_path).select("id", "categories", "update_date")
    
    # Join with metadata
    # arxiv in mapped_df matches id in metadata_df
    final_df = mapped_df.join(metadata_df, mapped_df.arxiv == metadata_df.id, "left") \
        .select(
            col("corpusid").cast("string").alias("corpusid"),
            col("arxiv").alias("arxiv_id"),
            col("categories").alias("all_categories"),
            col("update_date")
        )
    
    print("Saving results to temporary parquet...")
    temp_output = "data/temp_metadata_results"
    final_df.write.mode("overwrite").parquet(temp_output)
    
    print("Reading results back with pandas...")
    # Read all parts and combine
    import pandas as pd
    import glob
    parts = glob.glob(os.path.join(temp_output, "*.parquet"))
    dfs = [pd.read_parquet(p) for p in parts]
    final_pandas_df = pd.concat(dfs)
    
    metadata_map = {}
    for _, row in final_pandas_df.iterrows():
        year = str(row.update_date)[:4] if row.update_date else None
        primary_cat = row.all_categories.split(' ')[0] if row.all_categories else None
        
        metadata_map[row.corpusid] = {
            'year': year,
            'category': primary_cat,
            'all_categories': row.all_categories,
            'arxiv_id': row.arxiv_id
        }
    
    # Add entries for IDs that weren't found
    found_ids = set(metadata_map.keys())
    for corpus_id in target_ids:
        if corpus_id not in found_ids:
            metadata_map[corpus_id] = {
                'year': None,
                'category': None,
                'all_categories': None,
                'arxiv_id': None
            }

    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(metadata_map, f)
    
    print(f"Done! Found metadata for {len([v for v in metadata_map.values() if v['year']])} papers.")
    spark.stop()

if __name__ == "__main__":
    main()
