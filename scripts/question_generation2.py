#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import os
import re
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path

# Config
SAMPLE_SIZE = 10  # Start small - increase to 100, 1000, etc.
OUTPUT_DIR = "data/question_generation"
PATENT_SAMPLE_DIR = "data/patent_sample"

openrouter_key = os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

system_prompt = """You are an expert in patent comprehension. Your task is to generate structured questions that assess a reader's background knowledge necessary to understand a given patent abstract. The questions should focus on foundational concepts, principles, and applications relevant to the patent's domain without explicitly referencing the patent itself.

The questions should follow Bloom's Taxonomy and be categorized into three levels:
1. Remembering: Questions that assess the reader's ability to recall key technical concepts, definitions, and fundamental principles.
2. Understanding: Questions that assess the reader's ability to explain how different elements of similar technologies function and interact.
3. Applying: Questions that evaluate the reader's ability to apply their knowledge by solving problems, making predictions, or considering real-world applications.

Do not directly reference the patent abstract in any question."""

promptQ1 = """Patent Title: {patent_title}
Patent Abstract: {patent_summary}

Generate {qnum} questions that test a reader's ability to recall fundamental concepts, key terms, and basic components necessary to understand this patent domain.

Output Format (JSON):
{{"1": "question1", "2": "question2", "3": "question3"}}"""

promptQ2 = """Patent Title: {patent_title}
Patent Abstract: {patent_summary}

Generate {qnum} questions that assess comprehension by requiring explanation of how different elements in this technology domain work together.

Output Format (JSON):
{{"1": "question1", "2": "question2", "3": "question3"}}"""

def call_llm(prompt):
    completion = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=5000
    )
    response = completion.choices[0].message.content.strip()
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            print(f"JSON parse error, skipping: {response[:200]}")
            return None
    return None

def generate_questions(df, prompt_template, output_file):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {Path(output_file).stem}"):
        title = row.get('patent_title', '')
        abstract = row.get('patent_abstract_x', row.get('patent_abstract_y', ''))
        
        if pd.isna(abstract) or pd.isna(title):
            continue
        
        max_retries = 3
        questions = None
        for attempt in range(max_retries):
            prompt = prompt_template.format(patent_title=title, patent_summary=abstract, qnum=3)
            questions = call_llm(prompt)
            if questions and len(questions) == 3:
                break
        
        if questions and len(questions) == 3:
            for q_id, question in questions.items():
                results.append({
                    'patent_id': row['patent_id'],
                    'patent_title': title,
                    'patent_abstract': abstract,
                    'cpc_class': row.get('cpc_class', ''),
                    'is_computer_science_patent': row.get('is_computer_science_patent', False),
                    'question_id': q_id,
                    'question': question
                })
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved {len(results)} questions to {output_file}")

# Load CS patents
cs_files = sorted(Path(PATENT_SAMPLE_DIR).glob("cs_class_*.parquet"))
print(f"Found {len(cs_files)} CS patent files")

dfs = []
for f in cs_files:
    df = pd.read_parquet(f)
    dfs.append(df)

all_patents = pd.concat(dfs, ignore_index=True)
print(f"Total CS patents: {len(all_patents):,}")

# Sample
sample_df = all_patents.sample(n=min(SAMPLE_SIZE, len(all_patents)), random_state=42)
print(f"Sampled {len(sample_df)} patents for question generation")

# Generate both question types
os.makedirs(OUTPUT_DIR, exist_ok=True)
generate_questions(sample_df, promptQ1, f"{OUTPUT_DIR}/questions_remembering.csv")
generate_questions(sample_df, promptQ2, f"{OUTPUT_DIR}/questions_understanding.csv")

print("Done!")
