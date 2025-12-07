#!/usr/bin/env python3
"""
Get verification statistics from a verified results JSON file.
Shows per-question and per-paragraph success metrics.
"""

import json
import glob
import argparse
import os


def get_stats(verified_file):
    """Calculate and print verification statistics."""
    
    if not os.path.exists(verified_file):
        print(f"Error: File not found: {verified_file}")
        return
    
    print(f"Loading: {verified_file}\n")
    
    with open(verified_file, 'r') as f:
        data = json.load(f)
    
    # Count stats
    total_questions = 0
    questions_with_papers = 0
    questions_with_yes = 0
    total_paragraphs = 0
    paragraphs_with_yes = 0
    
    for patent_id, patent_data in data.items():
        for qtype in ['remembering', 'understanding']:
            for question_entry in patent_data['question_types'][qtype]:
                total_questions += 1
                papers = question_entry.get('retrieved_papers', [])
                
                if papers:  # Has papers
                    questions_with_papers += 1
                    question_has_yes = False
                    
                    for paper in papers:
                        for para in paper.get('paragraphs', []):
                            total_paragraphs += 1
                            if para.get('contains_answer') == 'Y':
                                paragraphs_with_yes += 1
                                question_has_yes = True
                    
                    if question_has_yes:
                        questions_with_yes += 1
    
    print(f"QUESTIONS:")
    print(f"  Total: {total_questions}")
    print(f"  With retrieved papers: {questions_with_papers}")
    print(f"  With at least one 'Y': {questions_with_yes}")
    if questions_with_papers > 0:
        print(f"  Success rate (of those with papers): {100*questions_with_yes/questions_with_papers:.1f}%")
    print(f"  Overall success rate: {100*questions_with_yes/total_questions:.1f}%")
    print(f"\nPARAGRAPHS:")
    print(f"  Total: {total_paragraphs}")
    print(f"  With 'Y' answer: {paragraphs_with_yes}")
    if total_paragraphs > 0:
        print(f"  Success rate: {100*paragraphs_with_yes/total_paragraphs:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get verification statistics')
    parser.add_argument('--input', type=str, help='Path to verified JSON file')
    parser.add_argument('--latest', action='store_true', help='Use latest verified file')
    
    args = parser.parse_args()
    
    if args.latest:
        # Find latest verified file
        verified_files = glob.glob("data/question_generation/retrieval_results/*_verified*.json")
        if not verified_files:
            print("No verified files found")
            exit(1)
        verified_files.sort(key=os.path.getmtime, reverse=True)
        verified_file = verified_files[0]
        print(f"Latest verified file: {verified_file}\n")
    elif args.input:
        verified_file = args.input
    else:
        # Try to find latest tavily verified
        verified_files = glob.glob("data/question_generation/retrieval_results/retrieval_results_tavily_*_verified*.json")
        if not verified_files:
            print("No Tavily verified files found. Use --input or --latest")
            exit(1)
        verified_files.sort(key=os.path.getmtime, reverse=True)
        verified_file = verified_files[0]
        print(f"Latest Tavily verified file: {verified_file}\n")
    
    get_stats(verified_file)
