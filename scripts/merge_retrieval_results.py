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
    questions_compared = 0  # Only questions where both methods retrieved papers
    answered_by_faiss_only = 0
    answered_by_tavily_only = 0
    answered_by_both = 0
    not_answered = 0
    questions_with_answer = 0
    
    # Track questions answered by both
    both_questions = []
    
    # For each question in merged data, check if we have an answer from FAISS
    # If not, try to get it from Tavily
    for patent_id, patent_data in merged_data.items():
        if patent_id not in tavily_data:
            continue
        
        tavily_patent = tavily_data[patent_id]
        
        for qtype in ['remembering', 'understanding']:
            for i, question_entry in enumerate(patent_data['question_types'].get(qtype, [])):
                total_questions += 1
                
                # Check if Tavily has retrieved papers for this question
                tavily_has_papers = False
                tavily_question = tavily_patent['question_types'].get(qtype, [])
                if i < len(tavily_question):
                    tavily_entry = tavily_question[i]
                    if tavily_entry.get('retrieved_papers') and len(tavily_entry['retrieved_papers']) > 0:
                        tavily_has_papers = True
                
                # Only compare questions where Tavily actually retrieved papers
                if not tavily_has_papers:
                    continue
                
                questions_compared += 1
                
                # Check if FAISS has an answer for this question
                has_faiss_answer = False
                for paper in question_entry.get('retrieved_papers', []):
                    for para in paper.get('paragraphs', []):
                        if para.get('contains_answer') == 'Y':
                            has_faiss_answer = True
                            break
                    if has_faiss_answer:
                        break
                
                # Check if Tavily has an answer (we already know it has papers from check above)
                has_tavily_answer = False
                tavily_entry = tavily_question[i]
                
                for paper in tavily_entry.get('retrieved_papers', []):
                    for para in paper.get('paragraphs', []):
                        if para.get('contains_answer') == 'Y':
                            has_tavily_answer = True
                            # Found answer in Tavily, add first one to merged
                            if not has_faiss_answer:
                                if not question_entry.get('retrieved_papers'):
                                    question_entry['retrieved_papers'] = []
                                # Add the Tavily paper with the answer
                                question_entry['retrieved_papers'].append(paper)
                            break
                    if has_tavily_answer:
                        break
                
                # Track which methods answered the question
                if has_faiss_answer and has_tavily_answer:
                    answered_by_both += 1
                    questions_with_answer += 1
                    both_questions.append((patent_id, qtype, i, question_entry.get('question', 'N/A')))
                elif has_faiss_answer:
                    answered_by_faiss_only += 1
                    questions_with_answer += 1
                elif has_tavily_answer:
                    answered_by_tavily_only += 1
                    questions_with_answer += 1
                else:
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
    print(f"\nTotal questions in files: {total_questions}")
    print(f"Questions compared (where Tavily retrieved papers): {questions_compared}")
    print(f"\nOf the {questions_compared} questions compared:")
    print(f"  FAISS only answered: {answered_by_faiss_only}")
    print(f"  Tavily only answered: {answered_by_tavily_only}")
    print(f"  BOTH methods answered: {answered_by_both}")
    print(f"  Neither answered: {not_answered}")
    print(f"  Total with answer: {questions_with_answer}")
    print(f"\n✓ Combined success rate: {100*questions_with_answer/questions_compared:.1f}% ({questions_with_answer}/{questions_compared})")
    print(f"  FAISS alone: {100*(answered_by_faiss_only+answered_by_both)/questions_compared:.1f}% ({answered_by_faiss_only+answered_by_both}/{questions_compared})")
    print(f"  Tavily alone: {100*(answered_by_tavily_only+answered_by_both)/questions_compared:.1f}% ({answered_by_tavily_only+answered_by_both}/{questions_compared})")
    
    # Show questions answered by both
    if both_questions:
        print(f"\n" + "-"*70)
        print(f"Questions answered by BOTH FAISS and Tavily ({len(both_questions)} total):")
        print("-"*70)
        for patent_id, qtype, idx, question_text in both_questions[:20]:  # Show first 20
            print(f"\nPatent: {patent_id} | Type: {qtype} | Q{idx+1}")
            print(f"  {question_text[:150]}{'...' if len(question_text) > 150 else ''}")
        if len(both_questions) > 20:
            print(f"\n... and {len(both_questions) - 20} more questions answered by both")
    
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
