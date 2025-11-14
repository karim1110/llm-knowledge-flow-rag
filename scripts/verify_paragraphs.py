#!/usr/bin/env python3
"""
Verify retrieved paragraphs using GPT-4 via OpenRouter API.

For each paragraph in the retrieval results:
1. Ask: Does this paragraph contain the answer? (Y/N)
2. If Y: Extract the answer using only knowledge from the paragraph

Updates the hierarchical JSON in-place with verification results.
Paragraph texts are already in the JSON from the retrieval step.
"""

import json
import os
import time
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
RESULTS_JSON = "data/question_generation/retrieval_results/retrieval_results_hierarchical.json"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini-2024-07-18"  # Fast and cheap for this task
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def verify_paragraph_with_gpt4(question, paragraph_text):
    """
    Use GPT-4 to verify if paragraph contains the answer.
    
    Returns:
        dict: {
            'contains_answer': 'Y' or 'N',
            'answer': str or None (answer extracted from paragraph if Y)
        }
    """
    
    prompt = f"""You are evaluating whether a paragraph contains the answer to a specific question.

QUESTION: {question}

PARAGRAPH:
{paragraph_text}

Task:
1. Does this paragraph contain information that answers the question? Respond with ONLY "Y" or "N".
2. If Y: Extract the answer using ONLY the knowledge present in this paragraph. Be concise and direct.

Format your response as:
CONTAINS_ANSWER: [Y or N]
ANSWER: [your answer if Y, or "N/A" if N]
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # Low temperature for consistent responses
        "max_tokens": 300
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Parse response
        lines = content.split('\n')
        contains_answer = 'N'
        answer = None
        
        for line in lines:
            if line.startswith('CONTAINS_ANSWER:'):
                contains_answer = line.split(':', 1)[1].strip().upper()
                if contains_answer not in ['Y', 'N']:
                    contains_answer = 'Y' if 'yes' in contains_answer.lower() else 'N'
            elif line.startswith('ANSWER:'):
                answer_text = line.split(':', 1)[1].strip()
                if answer_text != 'N/A':
                    answer = answer_text
        
        return {
            'contains_answer': contains_answer,
            'answer': answer
        }
    
    except Exception as e:
        print(f"API Error: {e}")
        return {
            'contains_answer': 'N',
            'answer': None
        }

def process_results(dry_run=False):
    """
    Process all paragraphs in the hierarchical JSON and verify them.
    Paragraph texts are already in the JSON.
    
    Args:
        dry_run: If True, only process first 5 paragraphs as a test
    """
    
    print("Loading hierarchical results...")
    with open(RESULTS_JSON, 'r') as f:
        data = json.load(f)
    
    # Count total paragraphs
    total_paragraphs = 0
    for patent in data.values():
        for qtype in ['remembering', 'understanding']:
            for question in patent['question_types'][qtype]:
                for paper in question['retrieved_papers']:
                    total_paragraphs += len(paper['paragraphs'])
    
    print(f"Total paragraphs to process: {total_paragraphs}")
    
    if dry_run:
        print("DRY RUN: Processing only first 5 paragraphs")
    
    processed = 0
    verified = 0
    contains_answer_count = 0
    
    # Process each paragraph
    with tqdm(total=min(5, total_paragraphs) if dry_run else total_paragraphs, desc="Verifying paragraphs") as pbar:
        for patent_id, patent_data in data.items():
            for qtype in ['remembering', 'understanding']:
                for question_entry in patent_data['question_types'][qtype]:
                    question = question_entry['question']
                    
                    for paper in question_entry['retrieved_papers']:
                        for para in paper['paragraphs']:
                            if dry_run and processed >= 5:
                                break
                            
                            paragraph_id = para['paragraph_id']
                            
                            # Get paragraph text from JSON (already loaded during retrieval)
                            para_text = para.get('paragraph_text')
                            
                            if para_text is None:
                                para['contains_answer'] = 'N'
                                para['answer'] = None
                                para['error'] = 'Paragraph text not found'
                            else:
                                # Verify with GPT-4
                                result = verify_paragraph_with_gpt4(question, para_text)
                                
                                para['contains_answer'] = result['contains_answer']
                                para['answer'] = result['answer']
                                
                                if result['contains_answer'] == 'Y':
                                    contains_answer_count += 1
                                
                                verified += 1
                                
                                # Rate limiting
                                time.sleep(0.5)  # 2 requests/second
                            
                            processed += 1
                            pbar.update(1)
                        
                        if dry_run and processed >= 5:
                            break
                    
                    if dry_run and processed >= 5:
                        break
                
                if dry_run and processed >= 5:
                    break
            
            if dry_run and processed >= 5:
                break
    
    # Save updated results
    output_file = RESULTS_JSON if not dry_run else RESULTS_JSON.replace('.json', '_verified_test.json')
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed: {processed} paragraphs")
    print(f"Verified with API: {verified} paragraphs")
    print(f"Contains answer (Y): {contains_answer_count} ({100*contains_answer_count/max(verified,1):.1f}%)")
    print(f"Saved to: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify paragraphs with GPT-4')
    parser.add_argument('--dry-run', action='store_true', help='Test with first 5 paragraphs only')
    
    args = parser.parse_args()
    
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        return
    
    print(f"Using model: {MODEL}")
    print(f"API Key: {OPENROUTER_API_KEY[:20]}...")
    
    process_results(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
