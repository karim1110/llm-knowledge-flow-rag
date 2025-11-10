"""
Analyze retrieval results and build epistemic linkage insights

This script demonstrates how to use the retrieval results to:
1. Examine understanding-based dependencies
2. Aggregate by technology/science fields (future work)
3. Compare different Bloom levels
"""

import pandas as pd
import numpy as np

# Load results
results = pd.read_csv("data/question_generation/retrieval_results/retrieval_results.csv")

print("="*80)
print("RETRIEVAL RESULTS ANALYSIS")
print("="*80)

# Basic statistics
print(f"\nTotal questions: {len(results)}")
print(f"\nQuestions by Bloom level:")
print(results['bloom_level'].value_counts())

print(f"\nUnique papers retrieved per question (average):")
print(f"  Top-10:  {results['unique_papers@10'].mean():.1f} papers")
print(f"  Top-50:  {results['unique_papers@50'].mean():.1f} papers")
print(f"  Top-100: {results['unique_papers@100'].mean():.1f} papers")

# Example: Examine one question's results
print("\n" + "="*80)
print("EXAMPLE QUESTION")
print("="*80)

example = results.iloc[0]
print(f"\nPatent ID: {example['patent_id']}")
print(f"Question: {example['question']}")
print(f"Bloom Level: {example['bloom_level']}")
print(f"\nTop 10 retrieved paragraphs:")
for i, para_id in enumerate(example['top_10_paragraph_ids'].split('|')[:10], 1):
    paper_id = para_id.split('_')[0]
    para_num = para_id.split('_')[1]
    print(f"  {i}. Paper {paper_id}, paragraph {para_num}")

print(f"\nUnique papers in top-100: {example['unique_papers@100']}")

# Understanding-based dependence calculation (simplified)
print("\n" + "="*80)
print("UNDERSTANDING-BASED DEPENDENCIES")
print("="*80)
print("\nFor each patent, we can calculate U_ps (dependence on paper s):")
print("  U_ps = Σⱼ αⱼ w_pjs")
print("  where:")
print("    - j iterates over questions about patent p")
print("    - w_pjs is the retrieval score (inverse rank)")
print("    - αⱼ is the Bloom level weight")
print("\nCurrent data has top-100 papers per question.")
print("Retrieval scores can be inferred from ranking.")

# Group by patent
print("\n" + "="*80)
print("PATENTS ANALYZED")
print("="*80)
patents = results['patent_id'].unique()
print(f"\nTotal unique patents: {len(patents)}")
print(f"Patents: {', '.join(map(str, patents[:5]))}...")

# Bloom level comparison
print("\n" + "="*80)
print("BLOOM LEVEL COMPARISON")
print("="*80)
for level in results['bloom_level'].unique():
    level_data = results[results['bloom_level'] == level]
    print(f"\n{level.upper()} questions:")
    print(f"  Count: {len(level_data)}")
    print(f"  Avg unique papers @100: {level_data['unique_papers@100'].mean():.1f}")

# Future: Epistemic Linkage Matrix E_kl
print("\n" + "="*80)
print("NEXT STEPS: Building E_kl Matrix")
print("="*80)
print("""
To build the epistemic linkage matrix E_kl:

1. Add CPC class mapping for patents (technology field k)
2. Add arXiv category mapping for papers (science field l)
3. Aggregate retrieval scores:
   E_kl = Σ_p∈k Σ_s∈l U_ps
   
4. Compare with citation-based C_kl:
   - Which scientific fields are understood vs cited?
   - What knowledge flows are invisible to citations?

5. Visualize as heatmap:
   - Rows: Technology fields (CPC classes)
   - Columns: Science fields (arXiv categories)
   - Values: Understanding-based linkage strength
""")

print("\n" + "="*80)
print("DATA EXPORT")
print("="*80)
print("\nTo export for further analysis:")
print("  - CSV format: Ready for Python/R/Excel")
print("  - Top papers per patent: Parse 'top_100_papers' column")
print("  - Paragraph IDs: Parse 'top_10_paragraph_ids' for fine-grained analysis")

# Save summary statistics
summary = results.groupby('bloom_level').agg({
    'patent_id': 'count',
    'unique_papers@10': 'mean',
    'unique_papers@50': 'mean',
    'unique_papers@100': 'mean'
}).round(1)

summary.columns = ['num_questions', 'avg_papers@10', 'avg_papers@50', 'avg_papers@100']
print("\nSummary by Bloom Level:")
print(summary)

summary.to_csv("data/question_generation/retrieval_results/summary_by_bloom_level.csv")
print("\nSaved summary to: data/question_generation/retrieval_results/summary_by_bloom_level.csv")
