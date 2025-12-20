"""
Embed questions using sentence-transformers (simpler than vLLM for small batches)
Saves embeddings to disk for retrieval job
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
LOCAL_MODEL = "/project/jevans/nadav/qaApproach/models/Qwen3-Embedding-0.6B"
QUESTION_DIR = "data/question_generation"
OUTPUT_DIR = "data/question_generation/embeddings"

# Question files
QUESTION_FILES = [
    # "questions_remembering.csv",
    # "questions_understanding.csv",
    "questions_understanding_2.csv"  # uncomment when ready
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
        
        # Get questions as list (use keywords if available)
        #if 'keywords' in df.columns:
        #    questions = df['keywords'].fillna('').tolist()  # remove keywords
        #else:
        questions = df['question'].dropna().tolist()  # Skip NaN values
        
        # Embed questions with "query" prompt for better retrieval
        # Qwen3 models benefit from using prompt_name="query" for search queries
        print("Embedding questions with 'query' prompt...", flush=True)
        
        # Checkpoint every 10k questions
        checkpoint_size = 10000
        base_name = os.path.splitext(question_file)[0]
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"{base_name}_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check for existing checkpoints and metadata to resume from
        import glob as glob_module
        metadata_file = os.path.join(checkpoint_dir, "metadata.json")
        start_question_idx = 0
        existing_checkpoints = []
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                start_question_idx = metadata.get("total_questions_processed", 0)
                print(f"Found checkpoint metadata. Previously processed {start_question_idx} questions.", flush=True)
                # Only load checkpoints up to the previously processed count
                total_batches = (start_question_idx + checkpoint_size - 1) // checkpoint_size
                existing_checkpoints = [os.path.join(checkpoint_dir, f"batch_{i}.npy") for i in range(total_batches)]
        else:
            existing_checkpoints = sorted(glob_module.glob(os.path.join(checkpoint_dir, f"batch_*.npy")))
            if existing_checkpoints:
                start_question_idx = len(existing_checkpoints) * checkpoint_size
                print(f"Found {len(existing_checkpoints)} checkpoint batches. Resuming from question {start_question_idx}...", flush=True)
        
        all_embeddings = []
        # Load existing checkpoints (only up to what was previously completed)
        for checkpoint_file in existing_checkpoints:
            if os.path.exists(checkpoint_file):
                batch_emb = np.load(checkpoint_file)
                all_embeddings.append(batch_emb)
        
        # Embed remaining questions from start_question_idx onwards
        for i in range(start_question_idx, len(questions), checkpoint_size):
            batch_idx = i // checkpoint_size
            batch = questions[i:i+checkpoint_size]
            print(f"Embedding batch {batch_idx + 1} ({i}-{min(i+checkpoint_size, len(questions))} of {len(questions)})...", flush=True)
            batch_embeddings = model.encode(
                batch, 
                prompt_name="query",
                show_progress_bar=True, 
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)
            
            # Save checkpoint for this batch
            checkpoint_file = os.path.join(checkpoint_dir, f"batch_{batch_idx}.npy")
            np.save(checkpoint_file, batch_embeddings)
            print(f"Batch {batch_idx + 1} complete - checkpoint saved", flush=True)
        
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        print(f"Embeddings shape: {embeddings.shape}", flush=True)
        
        # Save final embeddings
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_embeddings.npy")
        np.save(output_file, embeddings)
        print(f"Saved embeddings to: {output_file}", flush=True)
        
        # Save metadata for next run
        with open(metadata_file, 'w') as f:
            json.dump({
                "total_questions_processed": len(questions),
                "embeddings_shape": embeddings.shape,
                "output_file": output_file
            }, f, indent=2)
        print(f"Saved checkpoint metadata", flush=True)
        
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
