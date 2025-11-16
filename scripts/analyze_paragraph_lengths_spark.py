"""
Analyze paragraph length distribution using PySpark for faster processing
Reads all paragraphs and computes statistics on character lengths
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, col

# Configuration
PARQUET_SHARDS_DIR = "/project/jevans/apto_data_engineering/data/arxiv/s2_arxiv_full_text_grobid/shards_partitioned"
OUTPUT_DIR = "data/question_generation/retrieval_results"

def analyze_paragraph_lengths_spark():
    """Analyze paragraph lengths using PySpark."""
    print("Starting paragraph length analysis with PySpark...")
    
    # Get Spark master from environment (set by sbatch script)
    import os
    master = os.environ.get("MASTER", "local[*]")
    
    # Initialize Spark with appropriate memory settings
    spark = SparkSession.builder \
        .appName("ParagraphLengthAnalysis") \
        .master(master) \
        .config("spark.driver.memory", "200g") \
        .config("spark.executor.memory", "40g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.maxResultSize", "8g") \
        .getOrCreate()
    
    try:
        # Read all parquet files
        print(f"Reading parquet files from {PARQUET_SHARDS_DIR}...")
        df = spark.read.parquet(f"{PARQUET_SHARDS_DIR}/*/")
        
        # Select only paragraph_text column and filter nulls
        df = df.select("paragraph_text").filter(col("paragraph_text").isNotNull())
        
        # Add length column
        df = df.withColumn("length", length(col("paragraph_text")))
        
        # Cache for multiple operations
        df.cache()
        
        # Get total count
        total_count = df.count()
        print(f"Total paragraphs: {total_count:,}")
        
        # Compute statistics
        print("Computing statistics...")
        stats = df.select("length").summary("count", "mean", "stddev", "min", "max")
        stats.show()
        
        # Get percentiles
        percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        quantiles = df.stat.approxQuantile("length", percentiles, 0.01)
        
        # Collect lengths for detailed analysis (sample if too large)
        print("Collecting length data...")
        if total_count > 10_000_000:
            # Sample 10M paragraphs for plotting
            print(f"Sampling 10M paragraphs from {total_count:,} total...")
            lengths_df = df.sample(False, 10_000_000 / total_count).select("length")
        else:
            lengths_df = df.select("length")
        
        lengths = np.array([row.length for row in lengths_df.collect()])
        
        # Count short paragraphs
        print("\nCounting short paragraphs...")
        short_counts = {}
        for threshold in [10, 25, 50, 100, 200]:
            count = df.filter(col("length") < threshold).count()
            pct = 100 * count / total_count
            short_counts[threshold] = (count, pct)
            print(f"Paragraphs < {threshold} chars: {count:,} ({pct:.2f}%)")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PARAGRAPH LENGTH STATISTICS (PySpark)")
        print(f"{'='*60}")
        print(f"Total paragraphs: {total_count:,}")
        print(f"\nLength Statistics (characters):")
        stats_dict = {row.summary: {k: row[k] for k in stats.columns if k != 'summary'} 
                     for row in stats.collect()}
        print(f"  Mean:     {stats_dict['mean']['length']}")
        print(f"  Std Dev:  {stats_dict['stddev']['length']}")
        print(f"  Min:      {stats_dict['min']['length']}")
        print(f"  Max:      {stats_dict['max']['length']}")
        print(f"\nPercentiles:")
        for i, p in enumerate([1, 5, 10, 25, 50, 75, 90, 95, 99]):
            print(f"  {p:2d}th:     {quantiles[i]:.2f}")
        
        # Save statistics
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stats_file = os.path.join(OUTPUT_DIR, "paragraph_length_stats_spark.txt")
        
        with open(stats_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"PARAGRAPH LENGTH STATISTICS (PySpark)\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total paragraphs: {total_count:,}\n")
            f.write(f"\nLength Statistics (characters):\n")
            f.write(f"  Mean:     {stats_dict['mean']['length']}\n")
            f.write(f"  Std Dev:  {stats_dict['stddev']['length']}\n")
            f.write(f"  Min:      {stats_dict['min']['length']}\n")
            f.write(f"  Max:      {stats_dict['max']['length']}\n")
            f.write(f"\nPercentiles:\n")
            for i, p in enumerate([1, 5, 10, 25, 50, 75, 90, 95, 99]):
                f.write(f"  {p:2d}th:     {quantiles[i]:.2f}\n")
            f.write(f"\nShort Paragraph Counts:\n")
            for threshold in [10, 25, 50, 100, 200]:
                count, pct = short_counts[threshold]
                f.write(f"  < {threshold:3d} chars: {count:,} ({pct:.2f}%)\n")
        
        print(f"\nStatistics saved to: {stats_file}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Histogram of all lengths (capped at 2000 for better visualization)
        ax1 = axes[0, 0]
        lengths_capped = lengths[lengths <= 2000]
        ax1.hist(lengths_capped, bins=100, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Paragraph Length (characters)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Distribution of Paragraph Lengths (≤2000 chars)\n(Sampled: n={len(lengths):,}, Total dataset: {total_count:,})', fontsize=12)
        ax1.axvline(lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {lengths.mean():.0f}')
        ax1.axvline(np.median(lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.0f}')
        ax1.legend(fontsize=10)
        ax1.set_xlim(0, 2000)
        
        # 2. Histogram zoomed to < 1000 chars
        ax2 = axes[0, 1]
        lengths_subset = lengths[lengths < 1000]
        ax2.hist(lengths_subset, bins=100, edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel('Paragraph Length (characters)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title(f'Distribution (< 1000 chars)\n(n={len(lengths_subset):,} of {len(lengths):,} sampled)', fontsize=12)
        ax2.axvline(lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {lengths.mean():.0f}')
        ax2.axvline(np.median(lengths), color='green', linestyle='--', linewidth=2, label=f'Overall Median: {np.median(lengths):.0f}')
        ax2.legend(fontsize=10)
        
        # 3. Box plot (capped for better visibility)
        ax3 = axes[1, 0]
        lengths_for_box = lengths[lengths <= 2000]  # Cap outliers for better visualization
        bp = ax3.boxplot(lengths_for_box, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax3.set_ylabel('Paragraph Length (characters)', fontsize=11)
        ax3.set_title(f'Box Plot of Paragraph Lengths (≤2000 chars)\n(Excludes {(lengths > 2000).sum():,} extreme outliers)', fontsize=12)
        ax3.set_xticklabels([f'All Paragraphs\n(n={len(lengths_for_box):,})'])
        ax3.set_ylim(0, 2000)
        
        # 4. Cumulative distribution (log scale X for better detail)
        ax4 = axes[1, 1]
        sorted_lengths = np.sort(lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
        ax4.plot(sorted_lengths, cumulative, linewidth=2)
        ax4.set_xlabel('Paragraph Length (characters, log scale)', fontsize=11)
        ax4.set_ylabel('Cumulative Percentage (%)', fontsize=11)
        ax4.set_title('Cumulative Distribution of Paragraph Lengths', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        # Add reference lines with better positioning
        for threshold in [50, 100, 200, 500, 1000]:
            pct = 100 * (lengths < threshold).sum() / len(lengths)
            ax4.axvline(threshold, color='red', linestyle='--', alpha=0.3)
            ax4.text(threshold, 50, f'{threshold}\n({pct:.1f}%)', 
                    rotation=90, va='center', ha='right', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(OUTPUT_DIR, "paragraph_length_distribution_spark.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        
        # Save lengths array
        lengths_file = os.path.join(OUTPUT_DIR, "paragraph_lengths_spark.npy")
        np.save(lengths_file, lengths)
        print(f"Lengths array saved to: {lengths_file}")
        
        print("\nAnalysis complete!")
        
    finally:
        spark.stop()
    
    return lengths

if __name__ == "__main__":
    lengths = analyze_paragraph_lengths_spark()
