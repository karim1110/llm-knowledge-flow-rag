#!/usr/bin/env python3
"""
Merge retrieval results from multiple methods (FAISS + Tavily).
For each question, takes the first 'Y' answer found across methods.
If FAISS has an answer, uses it. Otherwise, uses Tavily's answer if available.
"""

import json
import argparse


def merge_results(faiss_file, tavily_file, output_file=None):
    """
    Merge FAISS and Tavily results.
    Priority: FAISS first, then Tavily
    """
    
    print(f"Loading FAISS results from: {faiss_file}")
    with open(faiss_file, 'r') as f:
        faiss_data = json.load(f)
    
    print(f"Loading Tavily results from: {tavily_file}")
    with open(tavily_file, 'r') as f:
        tavily_data = json.load(f)
    
    # Start with FAISS as base
    merged_data = json.loads(json.dumps(faiss_data))  # Deep copy
    
    # Track stats
    total_questions = 0
    answered_by_faiss_only = 0
    answered_by_tavily_only = 0
    answered_by_both = 0
    not_answered = 0
    questions_with_answer = 0
    
    # For each question in merged data, check if we have an answer from FAISS
    # If not, try to get it from Tavily
    for patent_id, patent_data in merged_data.items():
        if patent_id not in tavily_data:
            continue
        
        tavily_patent = tavily_data[patent_id]
        
        for qtype in ['remembering', 'understanding']:
            for i, question_entry in enumerate(patent_data['question_types'].get(qtype, [])):
                total_questions += 1
                
                # Check if FAISS has an answer for this question
                has_faiss_answer = False
                for paper in question_entry.get('retrieved_papers', []):
                    for para in paper.get('paragraphs', []):
                        if para.get('contains_answer') == 'Y':
                            has_faiss_answer = True
                            answered_by_faiss_only += 1
                            questions_with_answer += 1
                            break
                    if has_faiss_answer:
                        break
                
                # If FAISS doesn't have answer, try Tavily
                if not has_faiss_answer:
                    tavily_question = tavily_patent['question_types'].get(qtype, [])
                    if i < len(tavily_question):
                        tavily_entry = tavily_question[i]
                        
                        for paper in tavily_entry.get('retrieved_papers', []):
                            for para in paper.get('paragraphs', []):
                                if para.get('contains_answer') == 'Y':
                                    # Found answer in Tavily, add first one to merged
                                    # But we need to be careful - check if we already have papers
                                    if not question_entry.get('retrieved_papers'):
                                        question_entry['retrieved_papers'] = []
                                    
                                    # Add the Tavily paper with the answer
                                    question_entry['retrieved_papers'].append(paper)
                                    answered_by_tavily_only += 1
                                    questions_with_answer += 1
                                    break
                            if len([p for p in (question_entry.get('retrieved_papers', []) or []) 
                                   for para in p.get('paragraphs', []) 
                                   if para.get('contains_answer') == 'Y']) > 0:
                                break
                
                # Track unanswered
                has_answer = len([p for p in (question_entry.get('retrieved_papers', []) or []) 
                                 for para in p.get('paragraphs', []) 
                                 if para.get('contains_answer') == 'Y']) > 0
                if not has_answer:
                    not_answered += 1
    
    # Save merged results
    if output_file is None:
        output_file = "data/question_generation/retrieval_results/retrieval_results_merged_faiss_tavily.json"
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    # Print stats
    print("\n" + "="*70)
    print("MERGE RESULTS - FAISS + Tavily")
    print("="*70)
    print(f"\nTotal questions: {total_questions}")
    print(f"Questions with answer:")
    print(f"  From FAISS only: {answered_by_faiss_only}")
    print(f"  From Tavily only: {answered_by_tavily_only}")
    print(f"  Total answered: {questions_with_answer}")
    print(f"  Not answered: {not_answered}")
    print(f"\n✓ Success rate: {100*questions_with_answer/total_questions:.1f}% ({questions_with_answer}/{total_questions})")
    print(f"\nMerged results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge FAISS and Tavily retrieval results')
    parser.add_argument('--faiss', type=str, 
                       default='data/question_generation/retrieval_results/retrieval_results_hierarchical_merged_verified.json',
                       help='Path to FAISS verified results')
    parser.add_argument('--tavily', type=str,
                       default='data/question_generation/retrieval_results/retrieval_results_tavily_100_verified.json',
                       help='Path to Tavily verified results')
    parser.add_argument('--output', type=str,
                       help='Path to save merged results (default: retrieval_results_merged_faiss_tavily.json)')
    
    args = parser.parse_args()
    
    merge_results(args.faiss, args.tavily, args.output)
