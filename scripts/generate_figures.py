import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        return json.load(f)

def get_stats(data, label):
    rows = []
    for patent_id, patent_data in data.items():
        patent_year = patent_data.get('grant_year')
        patent_cpc = patent_data.get('cpc_class')
        patent_section = patent_data.get('cpc_section')
        
        q_types = patent_data.get('question_types', {})
        for q_type, questions in q_types.items():
            if not questions:
                continue
            
            for q_obj in questions:
                q_text = q_obj.get('question')
                for paper in q_obj.get('retrieved_papers', []):
                    paper_id = paper['paper_id']
                    paper_meta = paper.get('metadata', {})
                    paper_year = paper_meta.get('year')
                    paper_cat = paper_meta.get('category')
                    
                    # Check if any paragraph contains the answer
                    is_verified = any(p.get('contains_answer') == 'Y' for p in paper.get('paragraphs', []))
                    
                    if is_verified:
                        rows.append({
                            'patent_id': patent_id,
                            'patent_year': patent_year,
                            'patent_cpc': patent_cpc,
                            'patent_section': patent_section,
                            'paper_id': paper_id,
                            'paper_year': paper_year,
                            'paper_cat': paper_cat,
                            'q_type': q_type,
                            'label': label
                        })
    return pd.DataFrame(rows)

def main():
    os.makedirs('figures', exist_ok=True)
    
    df_und = get_stats(load_data('data/question_generation/retrieval_results/retrieval_results_understanding_enriched.json'), 'understanding')
    df_rem = get_stats(load_data('data/question_generation/retrieval_results/retrieval_results_remembering_enriched.json'), 'remembering')
    
    df = pd.concat([df_und, df_rem])
    df.to_csv('data/verified_links_stats.csv', index=False)
    print(f"Saved {len(df)} verified links to data/verified_links_stats.csv")

    # Save E_kl matrix (aggregated)
    df_u = df[df['q_type'] == 'understanding'].copy()
    df_u['paper_field'] = df_u['paper_cat'].apply(lambda x: x.split('.')[0] if x else 'Unknown')
    e_kl = df_u.groupby(['patent_section', 'paper_field']).size().unstack(fill_value=0)
    e_kl.to_csv('data/e_kl_matrix.csv')
    print("Saved data/e_kl_matrix.csv")

    # Figure 1: Heatmap of Patent Section vs Science Field (Understanding)
    plt.figure(figsize=(14, 10))
    df_u = df[df['q_type'] == 'understanding'].copy()
    # Simplify paper category to broad field
    df_u['paper_field'] = df_u['paper_cat'].apply(lambda x: x.split('.')[0] if x else 'Unknown')
    
    # Filter out Unknown
    df_u = df_u[df_u['paper_field'] != 'Unknown']
    
    # Filter for top fields/sections to keep heatmap readable
    top_fields = df_u['paper_field'].value_counts().head(15).index
    top_sections = df_u['patent_section'].value_counts().head(10).index
    
    df_filtered = df_u[df_u['paper_field'].isin(top_fields) & df_u['patent_section'].isin(top_sections)]
    
    # Create pivot table with counts
    pivot = df_filtered.pivot_table(index='patent_section', columns='paper_field', values='patent_id', aggfunc='count', fill_value=0)
    
    # Normalize by row (Patent Section) to show dependence probability
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
    
    sns.heatmap(pivot_norm, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Figure 1: Knowledge Flow (Understanding-based)\nProbability of Science Field given Patent Section')
    plt.ylabel('Patent Section (CPC)')
    plt.xlabel('Scientific Field (ArXiv)')
    plt.tight_layout()
    plt.savefig('figures/figure1_heatmap.png')
    print("Saved figures/figure1_heatmap.png")

    # Figure 2: Temporal Lag (Patent Year - Paper Year)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    df_temp = df.dropna(subset=['patent_year', 'paper_year']).copy()
    df_temp['patent_year'] = df_temp['patent_year'].astype(int)
    df_temp['paper_year'] = df_temp['paper_year'].astype(int)
    df_temp['lag'] = df_temp['patent_year'] - df_temp['paper_year']
    
    # Subplot 1: Histogram
    sns.histplot(data=df_temp, x='lag', hue='q_type', element='step', bins=range(-5, 30), ax=ax1)
    ax1.set_title('Distribution of Temporal Lag')
    ax1.set_xlabel('Years (Patent Grant Year - Paper Year)')
    ax1.set_ylabel('Number of Verified Links')
    
    # Subplot 2: Evolution of Mean Lag over Time
    # Group by patent year and q_type, calculate mean lag
    lag_evolution = df_temp.groupby(['patent_year', 'q_type'])['lag'].mean().reset_index()
    # Filter for years with enough data (e.g., 2000-2024)
    lag_evolution = lag_evolution[(lag_evolution['patent_year'] >= 2000) & (lag_evolution['patent_year'] <= 2024)]
    
    sns.lineplot(data=lag_evolution, x='patent_year', y='lag', hue='q_type', marker='o', ax=ax2)
    ax2.set_title('Evolution of Mean Epistemic Lag over Time')
    ax2.set_xlabel('Patent Grant Year')
    ax2.set_ylabel('Mean Lag (Years)')
    
    plt.tight_layout()
    plt.savefig('figures/figure2_temporal_lag.png')
    print("Saved figures/figure2_temporal_lag.png")

    # Figure 3: Understanding vs Remembering (by Patent Section)
    plt.figure(figsize=(12, 6))
    section_counts = df.groupby(['patent_section', 'q_type']).size().unstack(fill_value=0)
    section_counts_norm = section_counts.div(section_counts.sum(axis=1), axis=0)
    section_counts_norm.plot(kind='bar', stacked=True)
    plt.title('Figure 3: Proportion of Understanding vs Remembering Links by Patent Section')
    plt.ylabel('Proportion')
    plt.legend(title='Question Type')
    plt.tight_layout()
    plt.savefig('figures/figure3_comparison.png')
    print("Saved figures/figure3_comparison.png")

if __name__ == "__main__":
    main()
