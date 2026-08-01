# PaperIQ – AI Powered Research Insight Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python"/>
</p>

> Transform unstructured academic papers into structured knowledge, insights, and research opportunities.

---

##  Features

| Module | Description |
|--------|-------------|
|  PDF Processing | Extracts and cleans text from PDF research papers |
|  Keyword Extraction | TF-IDF + RAKE + KeyBERT multi-method extraction |
|  Domain Classification | Auto-categorizes into 10 research domains |
|  Summarization | Extractive summarization per section |
|  Gap Detection | Identifies limitations, gaps, and future directions |
|  Citation Analysis | Reference extraction, year distribution, top authors |
|  Semantic Similarity | Sentence-Transformers powered similarity engine |
|  Multi-Paper Comparison | Side-by-side comparison table with heatmap |
|  Topic Modeling | LDA-based topic discovery |
|  Quality Scoring | Composite score across 5 dimensions |
|  Idea Generator | Research extensions and project suggestions |
|  Semantic Search | Query across uploaded papers with NLP |

---

##  Project Structure

```
paper/
├── backend/
│   ├── main.py                    # FastAPI application
│   └── modules/
│       ├── pdf_processor.py       # PDF text extraction
│       ├── text_preprocessor.py   # NLP preprocessing
│       ├── section_identifier.py  # Section detection
│       ├── keyword_extractor.py   # TF-IDF/RAKE/KeyBERT
│       ├── domain_classifier.py   # Research domain classification
│       ├── summarizer.py          # Extractive summarization
│       ├── gap_detector.py        # Research gap detection
│       ├── citation_analyzer.py   # Citation analysis
│       ├── similarity_engine.py   # Semantic similarity
│       ├── trend_analyzer.py      # LDA topic modeling
│       ├── quality_scorer.py      # Paper quality scoring
│       ├── idea_generator.py      # Research idea generation
│       └── paper_comparator.py    # Multi-paper comparison
├── frontend/
│   └── app.py                     # Streamlit dashboard
├── requirements.txt
├── install.ps1                    # Dependency installer
├── start_backend.ps1              # Backend startup script
└── start_frontend.ps1             # Frontend startup script
```



## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/analyze` | Full single paper analysis |
| `POST` | `/api/compare` | Multi-paper comparison |
| `POST` | `/api/similarity` | Pairwise similarity matrix |
| `POST` | `/api/search` | Semantic search across papers |

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Plotly |
| Backend | FastAPI + Uvicorn |
| PDF Processing | PyMuPDF + pdfplumber |
| NLP | NLTK + SpaCy |
| Keywords | KeyBERT + RAKE + TF-IDF |
| Similarity | Sentence-Transformers (all-MiniLM-L6-v2) |
| Topic Modeling | scikit-learn LDA |
| Vector Search | FAISS (optional) |

---

## Analysis Output

For each paper, PaperIQ produces:

- Structured section extraction (Abstract, Intro, Methods, Results, Conclusion)
- Multi-method keyword extraction with scoring
- Research domain classification with confidence
- Section-level and overall summaries
- Citation count, year distribution, top authors
- Research gap and limitation identification
- Composite quality score (0–100) with letter grade
- Research ideas and project suggestions

---



## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ROUGE | Summarization quality |
| Cosine Similarity | Semantic similarity |
| Domain Accuracy | Classification correctness |
| Quality Score | Heuristic composite (0–100) |

---

*PaperIQ – Accelerate your research. Discover insights faster.*
