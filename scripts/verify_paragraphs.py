#!/usr/bin/env python3
"""
Verify retrieved paragraphs using GPT-4 via OpenRouter API.

For each paragraph in the retrieval results:
1. Check if paragraph text is already loaded (loaded during retrieval)
2. Ask GPT-4: Does this paragraph contain the answer? (Y/N)
3. If Y: Extract the answer using only knowledge from the paragraph

Updates the hierarchical JSON in-place with verification results.
"""

import json
import os
import time
import datetime
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
RESULTS_JSON = "data/question_generation/retrieval_results/retrieval_results_hierarchical.json"
ERROR_LOG = "data/question_generation/retrieval_results/verification_errors.log"
TIMING_LOG = "data/question_generation/retrieval_results/verification_timing.log"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-5-nano-2025-08-07"  # GPT-5 nano (latest)
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def log_error(paragraph_id, error_msg):
    """Log API errors to file."""
    with open(ERROR_LOG, 'a') as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"[{timestamp}] Paragraph: {paragraph_id} | Error: {error_msg}\n")

def verify_paragraph_with_gpt4(question, paragraph_text, patent_title, patent_abstract, debug=False):
    """
    Use GPT-5 nano to verify if paragraph contains the answer.
    
    Returns:
        dict: {
            'contains_answer': 'Y' or 'N',
            'answer': str or None (answer extracted from paragraph if Y)
        }
    """
    
    prompt = f"""You are evaluating whether a scientific paragraph contains the answer to a question about a patent.

PATENT CONTEXT:
Title: {patent_title}
Abstract: {patent_abstract}

QUESTION: {question}

SCIENTIFIC PARAGRAPH:
{paragraph_text}

Task:
1. Does this scientific paragraph contain information that directly answers the question about the patent? 
   - Consider whether the paragraph provides relevant technical knowledge that could answer the question
   - Be strict: only answer "Y" if the paragraph contains substantive information addressing the question
2. If Y: Extract the answer using ONLY the knowledge present in this paragraph. Be concise and technical.

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
        "max_tokens": 1000  # Higher for reasoning models that need tokens for thinking + output
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # GPT-5 nano is a reasoning model - check both content and reasoning fields
        content = result['choices'][0]['message'].get('content') or ''
        reasoning = result['choices'][0]['message'].get('reasoning') or ''
        content = content.strip()
        reasoning = reasoning.strip()
        
        # Use whichever field has content (reasoning models put output in reasoning field)
        text_to_parse = reasoning if reasoning and not content else content
        
        if debug:
            print(f"\n=== DEBUG ===")
            print(f"Content: {content[:200] if content else '(empty)'}")
            print(f"Reasoning: {reasoning[:200] if reasoning else '(empty)'}")
            print(f"Text to parse: {text_to_parse[:200] if text_to_parse else '(empty)'}")
        
        # If content is empty, the model didn't produce structured output
        if not content:
            return {
                'contains_answer': 'N',
                'answer': None,
                'error': 'Model produced no structured output (reasoning only)'
            }
        
        # Parse response
        lines = text_to_parse.split('\n')
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
        
        if debug:
            print(f"Parsed contains_answer: {contains_answer}")
            print(f"Parsed answer: {answer}")
        
        return {
            'contains_answer': contains_answer,
            'answer': answer
        }
    
    except Exception as e:
        error_msg = str(e)
        return {
            'contains_answer': 'N',
            'answer': None,
            'error': error_msg
        }

def process_results(input_file=None, dry_run=False, limit=None, sample_mode='sequential', debug=False):
    """
    Process all paragraphs in the hierarchical JSON and verify them.
    Assumes paragraph texts are already loaded during retrieval.
    
    Args:
        input_file: Path to input JSON file (defaults to RESULTS_JSON)
        dry_run: If True, only process first 3 paragraphs as a test
        limit: If set, only process first N paragraphs
        sample_mode: 'sequential' (default) or 'distributed' (sample evenly across patents/questions)
        debug: If True, print debug information
    """
    
    if input_file is None:
        input_file = RESULTS_JSON
    
    # Clear previous logs
    open(ERROR_LOG, 'w').close()
    
    start_time = time.time()
    
    print(f"Loading hierarchical results from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Count total paragraphs and check if texts are loaded
    total_paragraphs = 0
    missing_texts = 0
    all_paragraphs = []  # Store (patent_id, patent_data, qtype, question_entry, paper, para) tuples
    
    for patent_id, patent_data in data.items():
        for qtype in ['remembering', 'understanding']:
            for question_entry in patent_data['question_types'][qtype]:
                for paper in question_entry['retrieved_papers']:
                    for para in paper['paragraphs']:
                        all_paragraphs.append((patent_id, patent_data, qtype, question_entry, paper, para))
                        total_paragraphs += 1
                        if para.get('paragraph_text') is None:
                            missing_texts += 1
    
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Total patents: {len(data)}")
    print(f"Missing paragraph texts: {missing_texts}")
    
    if missing_texts > 0:
        print(f"\nWARNING: {missing_texts} paragraphs are missing text!")
        print("Run the retrieval script first to load paragraph texts.")
    
    if dry_run:
        print("\nDRY RUN: Processing only first 3 paragraphs")
        limit = 3
    elif limit:
        print(f"\nLIMIT: Processing only first {limit} paragraphs")
        if sample_mode == 'distributed':
            print("SAMPLING MODE: Distributed (evenly across patents/questions)")
            # Sample evenly: take every Nth paragraph to distribute across patents/questions
            step = max(1, total_paragraphs // limit)
            all_paragraphs = all_paragraphs[::step][:limit]
            print(f"Sampling every {step}th paragraph to get {len(all_paragraphs)} samples")
        else:
            print("SAMPLING MODE: Sequential (first N paragraphs)")
            all_paragraphs = all_paragraphs[:limit]
    
    processed = 0
    verified = 0
    contains_answer_count = 0
    api_errors = 0
    
    # Process paragraphs from the selected sample
    with tqdm(total=len(all_paragraphs), desc="Verifying paragraphs") as pbar:
        for patent_id, patent_data, qtype, question_entry, paper, para in all_paragraphs:
            patent_title = patent_data['patent_title']
            patent_abstract = patent_data['patent_abstract']
            question = question_entry['question']
            paragraph_id = para['paragraph_id']
            
            # Get paragraph text (should already be loaded)
            para_text = para.get('paragraph_text')
            
            if para_text is None:
                para['contains_answer'] = 'N'
                para['answer'] = None
                para['error'] = 'Paragraph text not found'
                log_error(paragraph_id, 'Paragraph text not found')
            else:
                # Verify with GPT-5 nano
                if debug:
                    print(f"\n>>> Processing paragraph {processed+1}: {paragraph_id}")
                    print(f"    Patent: {patent_id}, Question type: {qtype}")
                
                result = verify_paragraph_with_gpt4(
                    question, 
                    para_text,
                    patent_title,
                    patent_abstract,
                    debug=debug
                )
                
                para['contains_answer'] = result['contains_answer']
                para['answer'] = result['answer']
                
                if 'error' in result:
                    para['error'] = result['error']
                    log_error(paragraph_id, result['error'])
                    api_errors += 1
                
                if result['contains_answer'] == 'Y':
                    contains_answer_count += 1
                
                verified += 1
                
                # No rate limiting needed - API responses are slow enough (~10s each)
            
            processed += 1
            pbar.update(1)
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save updated results
    if limit and limit < total_paragraphs:
        output_file = input_file.replace('.json', f'_verified_{limit}.json')
    else:
        output_file = input_file.replace('.json', '_verified.json')
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Log timing information
    with open(TIMING_LOG, 'w') as f:
        f.write(f"Verification Run: {datetime.datetime.now().isoformat()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total paragraphs: {processed}\n")
        f.write(f"Verified with API: {verified}\n")
        f.write(f"API errors: {api_errors}\n")
        f.write(f"Contains answer (Y): {contains_answer_count} ({100*contains_answer_count/max(verified,1):.1f}%)\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write(f"Avg time per paragraph: {total_time/max(verified,1):.2f}s\n")
        f.write(f"Output file: {output_file}\n")
    
    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed: {processed} paragraphs")
    print(f"Verified with API: {verified} paragraphs")
    print(f"API errors: {api_errors}")
    print(f"Contains answer (Y): {contains_answer_count} ({100*contains_answer_count/max(verified,1):.1f}%)")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Avg time per paragraph: {total_time/max(verified,1):.2f}s")
    print(f"Saved to: {output_file}")
    if api_errors > 0:
        print(f"Error log: {ERROR_LOG}")
    print(f"Timing log: {TIMING_LOG}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify paragraphs with GPT-5')
    parser.add_argument('--input', type=str, help='Input JSON file (defaults to RESULTS_JSON)')
    parser.add_argument('--dry-run', action='store_true', help='Test with first 3 paragraphs only')
    parser.add_argument('--limit', type=int, help='Process only first N paragraphs')
    parser.add_argument('--distributed', action='store_true', help='Sample evenly across patents/questions (use with --limit)')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        return
    
    print(f"Using model: {MODEL}")
    print(f"API Key: {OPENROUTER_API_KEY[:20]}...")
    
    sample_mode = 'distributed' if args.distributed else 'sequential'
    process_results(input_file=args.input, dry_run=args.dry_run, limit=args.limit, 
                   sample_mode=sample_mode, debug=args.debug)

if __name__ == "__main__":
    main()
