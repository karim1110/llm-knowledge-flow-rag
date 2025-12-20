import json
import os

def enrich_json(input_path, paper_metadata, patent_metadata, output_path):
    print(f"Enriching {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    enriched_papers = 0
    enriched_patents = 0
    total_papers = 0
    total_patents = len(data)
    
    for patent_id, patent_data in data.items():
        # Enrich patent metadata
        if patent_id in patent_metadata:
            patent_data['grant_year'] = patent_metadata[patent_id].get('grant_year')
            patent_data['cpc_class'] = patent_metadata[patent_id].get('cpc_class')
            patent_data['cpc_section'] = patent_metadata[patent_id].get('cpc_section')
            enriched_patents += 1
            
        q_types = patent_data.get('question_types', {})
        for q_type in ['understanding', 'remembering']:
            for q_obj in q_types.get(q_type, []):
                for paper in q_obj.get('retrieved_papers', []):
                    total_papers += 1
                    paper_id = paper['paper_id']
                    if paper_id in paper_metadata:
                        paper['metadata'] = paper_metadata[paper_id]
                        if paper_metadata[paper_id].get('year'):
                            enriched_papers += 1
    
    print(f"Enriched {enriched_patents}/{total_patents} patents.")
    print(f"Enriched {enriched_papers}/{total_papers} paper instances.")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    paper_metadata_path = 'data/paper_metadata_map.json'
    patent_metadata_path = 'data/patent_metadata_map.json'
    
    if not os.path.exists(paper_metadata_path):
        print(f"Warning: {paper_metadata_path} not found.")
        paper_metadata = {}
    else:
        print(f"Loading paper metadata from {paper_metadata_path}...")
        with open(paper_metadata_path, 'r') as f:
            paper_metadata = json.load(f)
            
    if not os.path.exists(patent_metadata_path):
        print(f"Warning: {patent_metadata_path} not found.")
        patent_metadata = {}
    else:
        print(f"Loading patent metadata from {patent_metadata_path}...")
        with open(patent_metadata_path, 'r') as f:
            patent_metadata = json.load(f)
    
    files_to_enrich = [
        ('data/question_generation/retrieval_results/retrieval_results_understanding_merged_all_patents_verified_all_shards.json', 
         'data/question_generation/retrieval_results/retrieval_results_understanding_enriched.json'),
        ('data/question_generation/retrieval_results/retrieval_results_remembering_merged_all_patents_verified_all_shards.json',
         'data/question_generation/retrieval_results/retrieval_results_remembering_enriched.json')
    ]
    
    for input_p, output_p in files_to_enrich:
        if os.path.exists(input_p):
            enrich_json(input_p, paper_metadata, patent_metadata, output_p)
        else:
            print(f"Warning: {input_p} not found.")

if __name__ == "__main__":
    main()
