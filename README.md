# Tracing Understanding: Measuring Knowledge Flow from Science to Technology Using Large Language Models

### Data Flow
```
Patents → Questions (Bloom's taxonomy) → Embeddings → FAISS Retrieval → Scientific Papers
```

### Components

1. **Question Generation** (`scripts/question_generation2.py`)
   - Generates questions about patents using Bloom's taxonomy levels (Remember, Understand, Apply, Analyze, Evaluate, Create)
   - Currently implemented: Remembering and Understanding levels
   - Output: `data/question_generation/questions_*.csv`

2. **Question Embedding** (`scripts/embed_questions.py`)
   - Embeds questions using Qwen3-Embedding-0.6B model
   - Uses sentence-transformers for efficient small-batch processing
   - Output: `data/question_generation/embeddings/*.npy`

3. **FAISS Index** (Built separately)
   - GPU IVF-PQ compressed index: 188M arXiv paragraph embeddings
   - Index file: `indexes/faiss_qwen06_ivfpq.index` (15GB)
   - Mapping: `indexes/patent_id_mapping_ivf.npy` (paragraph IDs)
   - Configuration: 8192 clusters, PQ64x8, Float16 lookup tables

4. **Retrieval** (`scripts/retrieval_only.py`)
   - Searches FAISS index for top-100 most relevant scientific paragraphs per question
   - Outputs hierarchical JSON organized by patent → question type → questions → papers → paragraphs
   - Output: `data/question_generation/retrieval_results/retrieval_results_hierarchical.json`

5. **Results Viewer** (`scripts/view_hierarchical_results.py`)
   - Interactive exploration of retrieval results
   - View by patent, question type, or specific questions
   - Usage: `python3 scripts/view_hierarchical_results.py [--patent ID] [--question-type TYPE]`