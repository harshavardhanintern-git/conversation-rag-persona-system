# Conversation RAG + Persona Extraction System (Final Evaluator Build)

A production-quality system for analyzing conversation histories using a high-signal retrieval pipeline and evidence-driven persona extraction. This build focuses on strict filtering to eliminate noise and ensure all insights are anchored in meaningful conversation evidence.

## 🚀 Key Features

### 1. Advanced Topic Optimization
- **Segmentation**: Chronological message streams are segmented using thematic shift detection.
- **Hierarchical Merging**: Adjacent topics are merged if their semantic vectors share > 65% similarity.
- **Noise Reduction**: Topics with fewer than 5 messages are merged into the closest adjacent topic when possible, otherwise removed.
- **Optimization Goal**: Target < 10,000 topics for efficient retrieval at scale.

### 2. High-Signal Persona Extraction (Strict Filtering)
- **Hard Filters**: 
  - Messages shorter than **25 characters** are ignored.
  - Greetings, generic filler phrases ("Hi", "How are you", "Good"), and identity-only messages are stripped.
- **Evidence Threshold**: Personality traits require a **minimum of 2 independent evidence points** to be included.
- **Categorization**: Focuses exclusively on high-signal attributes:
  - **Ambitious & Goal-Oriented**: Professional aspirations and skill development.
  - **Life Events**: Major transitions like relocation or educational milestones.
  - **Emotional Articulation**: Authentic expressions of sentiment grounded in life context.

### 3. Precision Retrieval & Answer Pipeline
- **Retrieval Guardrails**: 
  - Similarity score must exceed **0.30**.
  - All retrieved evidence is filtered for length and contextuality before being presented.
- **Structured Answering**:
  - Every answer begins with an explicit anchor: *"Based on conversation evidence..."*.
  - Direct evidence lines are quoted to ensure traceability.
  - Avoids generic adjectives; uses analytical, evidence-backed summaries.

## 🛠️ Execution Steps

1. **Environment Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install streamlit pandas scikit-learn sentence-transformers faiss-cpu joblib tqdm
   ```

2. **Index Generation**:
   Rebuild retrieval artifacts with strict filtering enabled.
   ```bash
   python src/build_index.py --csv data/conversations.csv
   ```

3. **Launch Application**:
   ```bash
   streamlit run app.py
   ```

## 📐 Logic & Design Decisions
- **Threshold Logic**: The 0.45 similarity threshold for initial segmentation, combined with the 0.65 merge threshold, ensures that conversation segments represent distinct "mini-episodes" rather than random fragments.
- **Rule-Based Filtering**: Explicitly removes conversational noise (greetings/filler) before the indexing phase to prevent the system from "hallucinating" persona traits based on trivial interactions.
- **Traceability**: Every insight includes a Msg ID or direct quote, ensuring that evaluators can verify findings against the raw dataset.
