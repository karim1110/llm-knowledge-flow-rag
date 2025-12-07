#!/usr/bin/env python3
"""
Tavily-based retrieval for patent questions.
Uses Tavily API to retrieve relevant papers/paragraphs from the web/arXiv.
Outputs hierarchical JSON compatible with verify_paragraphs.py
"""

import os
import json
import argparse
from dotenv import load_dotenv
from tavily import TavilyClient
from tqdm import tqdm

# Load environment
load_dotenv()

# Config
QUESTIONS_CSV_REMEMBERING = "data/question_generation/questions_remembering.csv"
QUESTIONS_CSV_UNDERSTANDING = "data/question_generation/questions_understanding.csv"
OUTPUT_DIR = "data/question_generation/retrieval_results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "retrieval_results_tavily.json")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_RESULTS_PER_QUESTION = 10


def tavily_retrieve_paragraphs(client, question, patent_title, patent_abstract, max_results=10):
    """
    Retrieve paragraphs using Tavily API.
    Returns list of papers with paragraphs.
    """
    # Build search query (keep it under 400 chars)
    query = f"Question:{question} (context:{patent_title})"
    
    try:
        resp = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=True,
            include_domains=["arxiv.org"]  # Focus on arXiv for scientific papers
        )
    except Exception as e:
        print(f"  Tavily API error: {e}")
        return []
    
    papers = []
    for idx, result in enumerate(resp.get("results", [])):
        # Extract content - prefer raw_content over snippet
        content = result.get("raw_content") or result.get("content") or result.get("snippet", "")
        
        if not content or len(content.strip()) < 50:
            continue
        
        # Build paper structure
        paper_id = f"tavily_{idx}"
        paper = {
            "paper_id": paper_id,
            "metadata": {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "score": result.get("score", 0.0)
            },
            "paragraphs": [
                {
                    "paragraph_id": f"{paper_id}_p0",
                    "rank": idx + 1,
                    "has_paragraph": True,
                    "paragraph_text": content,
                    "contains_answer": None,
                    "answer": None
                }
            ]
        }
        papers.append(paper)
    
    return papers


def load_questions():
    """Load questions from CSVs and build hierarchical structure."""
    import csv
    
    patents = {}
    
    # Load remembering questions
    with open(QUESTIONS_CSV_REMEMBERING, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patent_id = row['patent_id']
            if patent_id not in patents:
                patents[patent_id] = {
                    'cpc': row['cpc_class'],
                    'patent_title': row['patent_title'],
                    'patent_abstract': row['patent_abstract'],
                    'question_types': {
                        'remembering': [],
                        'understanding': []
                    }
                }
            
            patents[patent_id]['question_types']['remembering'].append({
                'question': row['question'],
                'retrieved_papers': []
            })
    
    # Load understanding questions
    with open(QUESTIONS_CSV_UNDERSTANDING, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patent_id = row['patent_id']
            if patent_id not in patents:
                patents[patent_id] = {
                    'cpc': row['cpc_class'],
                    'patent_title': row['patent_title'],
                    'patent_abstract': row['patent_abstract'],
                    'question_types': {
                        'remembering': [],
                        'understanding': []
                    }
                }
            
            patents[patent_id]['question_types']['understanding'].append({
                'question': row['question'],
                'retrieved_papers': []
            })
    
    return patents


def main(limit=None):
    print("="*80)
    print("Tavily Retrieval Pipeline")
    print("="*80)
    
    if not TAVILY_API_KEY:
        print("Error: TAVILY_API_KEY not found in .env file")
        return
    
    print(f"API Key: {TAVILY_API_KEY[:20]}...")
    print(f"Max results per question: {MAX_RESULTS_PER_QUESTION}")
    
    # Initialize Tavily client
    client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # Load questions
    print("\nLoading questions...")
    patents = load_questions()
    print(f"Loaded {len(patents)} patents")
    
    # Count total questions
    total_questions = sum(
        len(p['question_types']['remembering']) + len(p['question_types']['understanding'])
        for p in patents.values()
    )
    print(f"Total questions: {total_questions}")
    
    if limit:
        print(f"Limiting to first {limit} questions")
    
    # Process each patent and question
    question_count = 0
    
    for patent_id, patent_data in tqdm(patents.items(), desc="Patents"):
        patent_title = patent_data['patent_title']
        patent_abstract = patent_data['patent_abstract']
        
        for qtype in ['remembering', 'understanding']:
            for question_entry in tqdm(patent_data['question_types'][qtype], 
                                      desc=f"  {qtype}", leave=False):
                question = question_entry['question']
                
                # Retrieve papers using Tavily
                papers = tavily_retrieve_paragraphs(
                    client, 
                    question, 
                    patent_title, 
                    patent_abstract,
                    max_results=MAX_RESULTS_PER_QUESTION
                )
                
                question_entry['retrieved_papers'] = papers
                question_entry['stats'] = {
                    'retrieved_papers': len(papers),
                    'retrieval_method': 'tavily',
                    'max_results': MAX_RESULTS_PER_QUESTION
                }
                
                question_count += 1
                
                if limit and question_count >= limit:
                    break
            
            if limit and question_count >= limit:
                break
        
        if limit and question_count >= limit:
            break
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = OUTPUT_JSON
    if limit:
        output_path = OUTPUT_JSON.replace('.json', f'_{limit}.json')
    
    with open(output_path, 'w') as f:
        json.dump(patents, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✓ Retrieval Complete!")
    print(f"{'='*80}")
    print(f"Processed: {question_count} questions")
    print(f"Output: {output_path}")
    print(f"\nNext step: Run verification")
    print(f"python scripts/verify_paragraphs.py --input {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tavily-based retrieval for patent questions')
    parser.add_argument('--limit', type=int, help='Limit to first N questions (for testing)')
    parser.add_argument('--max-results', type=int, default=MAX_RESULTS_PER_QUESTION,
                       help='Max results per question (default: 10)')
    
    args = parser.parse_args()
    
    if args.max_results:
        MAX_RESULTS_PER_QUESTION = args.max_results
    
    main(limit=args.limit)
