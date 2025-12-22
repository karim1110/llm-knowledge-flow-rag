# Tracing Understanding: Measuring Knowledge Flow from Science to Technology Using Large Language Models

### Running Long Scripts with tmux

For long-running scripts on interactive nodes, use tmux: `tmux new-session -d -s name -c /path "micromamba activate rag-py310 && python script.py"`. Check without attaching: `tmux capture-pane -t name -p`. Attach/detach: `tmux attach -t name` then `Ctrl+B, D`.

### Data Flow
```
FAISS Index (pre-built) → Patents → Questions (Bloom's taxonomy) → Embeddings → Retrieval → Evaluation
```

### Components

1. **FAISS Index** (Pre-built)
   - GPU IVF-PQ compressed index: 188M arXiv paragraph embeddings
   - Index file: `indexes/faiss_qwen06_ivfpq.index` (15GB)
   - Mapping: `indexes/patent_id_mapping_ivf.npy` (paragraph IDs)
   - Configuration: 8192 clusters, PQ64x8, Float16 lookup tables

2. **Question Generation** (`scripts/question_generation2.py`) (works on sbatch job without gpu)
   - Generates questions about patents using Bloom's taxonomy levels (Remembering, Understanding)
   - Output: `data/question_generation/questions_*.csv`

3. **Question Embedding** (`scripts/embed_questions.py`) (sbatch job)
   - Embeds questions using Qwen3-Embedding-0.6B model
   - Output: `data/question_generation/embeddings/*.npy`

4. **Retrieval** (`scripts/retrieval_only.py`) (sbatch job)
   - Searches FAISS index for top-100 most relevant scientific paragraphs per question
   - Output: `data/question_generation/retrieval_results/retrieval_results_hierarchical.json`

5. **Paragraph Text Loading** (`scripts/load_paragraph_texts.py`) (sbatch job)
   - Fills in actual paragraph texts from IDs using Spark and arXiv parquet shards
   - Supports optional context merging for short paragraphs (`--merge-context`)
   - Output: `retrieval_results_hierarchical_baseline.json` or `retrieval_results_hierarchical_merged.json`

6. **Evaluation** (`scripts/verify_paragraphs.py`) 
   - jobs/verify_paragraphs_remembering_array.sbatch
   - Evaluates retrieval results using LLM verification
   - Prints per-question stats (number of 'Y' answers, at least one 'Y', summary)