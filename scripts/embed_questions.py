"""
Embed questions using sentence-transformers (simpler than vLLM for small batches)
Saves embeddings to disk for retrieval job
"""
import os
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
LOCAL_MODEL = "/project/jevans/nadav/qaApproach/models/Qwen3-Embedding-0.6B"
QUESTION_DIR = "data/question_generation"
OUTPUT_DIR = "data/question_generation/embeddings"

# Question files
QUESTION_FILES = [
    "questions_remembering.csv",
    "questions_understanding.csv"
]

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading embedding model: {LOCAL_MODEL}", flush=True)
    model = SentenceTransformer(LOCAL_MODEL, trust_remote_code=True)
    print(f"Model loaded successfully!", flush=True)
    
    for question_file in QUESTION_FILES:
        print(f"\n{'='*80}", flush=True)
        print(f"Processing: {question_file}", flush=True)
        print(f"{'='*80}", flush=True)
        
        # Load questions
        question_path = os.path.join(QUESTION_DIR, question_file)
        df = pd.read_csv(question_path)
        print(f"Loaded {len(df)} questions", flush=True)
        
        # Get questions as list
        questions = df['question'].tolist()
        
        # Embed questions with "query" prompt for better retrieval
        # Qwen3 models benefit from using prompt_name="query" for search queries
        print("Embedding questions with 'query' prompt...", flush=True)
        embeddings = model.encode(
            questions, 
            prompt_name="query",
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        print(f"Embeddings shape: {embeddings.shape}", flush=True)
        
        # Save embeddings
        base_name = os.path.splitext(question_file)[0]
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_embeddings.npy")
        np.save(output_file, embeddings)
        print(f"Saved embeddings to: {output_file}", flush=True)
        
        # Also save the dataframe for reference
        df_output = os.path.join(OUTPUT_DIR, f"{base_name}_metadata.csv")
        df.to_csv(df_output, index=False)
        print(f"Saved metadata to: {df_output}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("All questions embedded successfully!", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(f"{'='*80}", flush=True)

if __name__ == "__main__":
    main()
