#!/usr/bin/env python3
"""
Interactive viewer for hierarchical retrieval results JSON.

Usage:
    python view_hierarchical_results.py
    python view_hierarchical_results.py --patent 9262630
    python view_hierarchical_results.py --patent 9262630 --question-type remembering
"""

import json
import argparse
from pathlib import Path

JSON_FILE = "data/question_generation/retrieval_results/retrieval_results_hierarchical.json"

def load_results():
    """Load the hierarchical JSON results."""
    with open(JSON_FILE, 'r') as f:
        return json.load(f)

def display_summary(data):
    """Display summary of all patents."""
    print("\n=== RETRIEVAL RESULTS SUMMARY ===\n")
    print(f"Total patents: {len(data)}\n")
    
    for patent_id, patent_data in data.items():
        n_remembering = len(patent_data['question_types']['remembering'])
        n_understanding = len(patent_data['question_types']['understanding'])
        
        print(f"Patent {patent_id}")
        print(f"  CPC: {patent_data['cpc']}")
        print(f"  Title: {patent_data['patent_title'][:80]}...")
        print(f"  Questions: {n_remembering} remembering, {n_understanding} understanding")
        print()

def display_patent(data, patent_id, question_type=None):
    """Display detailed view of a specific patent."""
    if patent_id not in data:
        print(f"Error: Patent {patent_id} not found")
        print(f"Available patents: {', '.join(data.keys())}")
        return
    
    patent_data = data[patent_id]
    
    print(f"\n=== PATENT {patent_id} ===")
    print(f"CPC: {patent_data['cpc']}")
    print(f"Grant Year: {patent_data['grant_year']}")
    print(f"Title: {patent_data['patent_title']}")
    print(f"\nAbstract: {patent_data['patent_abstract']}\n")
    
    # Determine which question types to show
    if question_type:
        question_types = [question_type.lower()]
    else:
        question_types = ['remembering', 'understanding']
    
    for qtype in question_types:
        questions = patent_data['question_types'][qtype]
        
        print(f"\n--- {qtype.upper()} QUESTIONS ({len(questions)}) ---\n")
        
        for i, q in enumerate(questions, 1):
            print(f"Q{i}: {q['question']}")
            print(f"     Retrieved {len(q['retrieved_papers'])} papers:")
            
            for j, paper in enumerate(q['retrieved_papers'][:5], 1):  # Show top 5
                para_list = ', '.join([p['paragraph_id'] for p in paper['paragraphs']])
                print(f"       {j}. Paper {paper['paper_id']}: {para_list}")
            
            if len(q['retrieved_papers']) > 5:
                print(f"       ... and {len(q['retrieved_papers']) - 5} more papers")
            print()

def display_question_details(data, patent_id, question_index, question_type):
    """Display full details for a specific question."""
    if patent_id not in data:
        print(f"Error: Patent {patent_id} not found")
        return
    
    patent_data = data[patent_id]
    questions = patent_data['question_types'][question_type.lower()]
    
    if question_index < 1 or question_index > len(questions):
        print(f"Error: Question index {question_index} out of range (1-{len(questions)})")
        return
    
    question = questions[question_index - 1]
    
    print(f"\n=== QUESTION DETAILS ===")
    print(f"Patent: {patent_id}")
    print(f"Type: {question_type}")
    print(f"Question: {question['question']}\n")
    
    print(f"Retrieved {len(question['retrieved_papers'])} papers:\n")
    
    for i, paper in enumerate(question['retrieved_papers'], 1):
        print(f"{i}. Paper ID: {paper['paper_id']}")
        print(f"   Paragraphs ({len(paper['paragraphs'])}):")
        for para in paper['paragraphs']:
            print(f"     - {para['paragraph_id']} (rank {para['rank']})")
        print()

def main():
    parser = argparse.ArgumentParser(description='View hierarchical retrieval results')
    parser.add_argument('--patent', '-p', help='Patent ID to view in detail')
    parser.add_argument('--question-type', '-t', choices=['remembering', 'understanding'],
                        help='Filter by question type')
    parser.add_argument('--question', '-q', type=int,
                        help='Question number within patent (1-indexed)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_results()
    
    # Dispatch based on arguments
    if args.question and args.patent and args.question_type:
        display_question_details(data, args.patent, args.question, args.question_type)
    elif args.patent:
        display_patent(data, args.patent, args.question_type)
    else:
        display_summary(data)

if __name__ == "__main__":
    main()
