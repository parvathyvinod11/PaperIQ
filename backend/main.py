"""
PaperIQ – FastAPI Backend
Main application entry point exposing REST APIs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import uvicorn
import datetime
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from routers.auth import router as auth_router
from routers.history import router as history_router
from database import history_collection
from auth_utils import get_current_user_optional

from modules.pdf_processor import PDFProcessor
from modules.text_preprocessor import TextPreprocessor
from modules.section_identifier import SectionIdentifier
from modules.keyword_extractor import KeywordExtractor
from modules.domain_classifier import DomainClassifier
from modules.summarizer import Summarizer
from modules.gap_detector import GapDetector
from modules.citation_analyzer import CitationAnalyzer
from modules.similarity_engine import SimilarityEngine
from modules.trend_analyzer import TrendAnalyzer
from modules.quality_scorer import QualityScorer
from modules.idea_generator import IdeaGenerator
from modules.paper_comparator import PaperComparator

# ──────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="PaperIQ API",
    description="AI-powered research paper analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(history_router)

security = HTTPBearer(auto_error=False)

# ──────────────────────────────────────────────────────────
# Module instances (singletons)
# ──────────────────────────────────────────────────────────

pdf_processor = PDFProcessor()
preprocessor = TextPreprocessor()
section_id = SectionIdentifier()
kw_extractor = KeywordExtractor()
domain_clf = DomainClassifier()
summarizer = Summarizer()
gap_detector = GapDetector()
citation_analyzer = CitationAnalyzer()
similarity_engine = SimilarityEngine()
trend_analyzer = TrendAnalyzer()
quality_scorer = QualityScorer()
idea_generator = IdeaGenerator()
comparator = PaperComparator()

# ──────────────────────────────────────────────────────────
# Helper: full pipeline for a single PDF
# ──────────────────────────────────────────────────────────

def analyze_paper_bytes(pdf_bytes: bytes, title: str = "") -> dict:
    """Run the full analysis pipeline on a PDF."""
    # 1. PDF Processing
    pdf_result = pdf_processor.process(pdf_bytes)
    text = pdf_result["cleaned_text"]
    metadata = pdf_result["metadata"]
    if not title:
        title = metadata.get("title") or "Untitled Paper"

    # 2. Preprocessing
    preproc = preprocessor.preprocess(text)

    # 3. Section Identification
    sections = section_id.extract_sections(text)
    section_stats = section_id.get_section_stats(sections)

    # 4. Keywords & Entities
    keywords = kw_extractor.extract_all(preproc["processed_text"])

    # 5. Domain Classification
    domain = domain_clf.classify(text)

    # 6. Summarization
    summaries = summarizer.summarize_sections(sections)
    contributions = summarizer.get_key_contributions(text)

    # 7. Research Gap Detection
    gaps = gap_detector.detect_gaps(text, sections)

    # 8. Citation Analysis
    citations = citation_analyzer.analyze(text)

    # 9. Topic Trends
    topics = trend_analyzer.extract_single_paper_topics(text)
    kw_freq = trend_analyzer.keyword_frequency(text, top_n=20)

    # 10. Quality Score
    quality = quality_scorer.compute_score(text, sections, citations, keywords)

    # 11. Research Ideas
    ideas = idea_generator.generate_ideas(
        sections,
        domain["primary_domain"],
        keywords,
        gaps["identified_gaps"],
    )

    return {
        "title": title,
        "metadata": metadata,
        "pdf_stats": {
            "word_count": pdf_result["word_count"],
            "page_count": metadata.get("page_count", 0),
        },
        "sections": {k: v for k, v in sections.items() if k != "full_text"},
        "section_stats": section_stats,
        "preprocessed": {
            "sentence_count": preproc["sentence_count"],
            "token_count": preproc["token_count"],
        },
        "keywords": keywords,
        "domain": domain,
        "summaries": summaries,
        "contributions": contributions,
        "gaps": gaps,
        "citations": citations,
        "topics": topics,
        "keyword_frequency": kw_freq,
        "quality": quality,
        "ideas": ideas,
    }


# ──────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "PaperIQ API is running. Visit /docs for the API reference."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_paper(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    token: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Upload a single PDF and receive a full analysis.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Uploaded file is too small or empty.")

    try:
        doc_title = title or file.filename
        result = analyze_paper_bytes(pdf_bytes, title=doc_title)
        
        # Save to history if logged in
        if token:
            username = get_current_user_optional(token.credentials)
            if username:
                history_doc = {
                    "username": username,
                    "title": doc_title,
                    "timestamp": datetime.datetime.utcnow(),
                    "analysis": result
                }
                await history_collection.insert_one(history_doc)

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/compare")
async def compare_papers(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDFs and get a comparison report.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least 2 papers.")
    if len(files) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 papers for comparison.")

    paper_analyses = []
    paper_texts = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF.")
        pdf_bytes = await f.read()
        analysis = analyze_paper_bytes(pdf_bytes, title=f.filename)
        paper_analyses.append(analysis)
        paper_texts.append({
            "title": analysis["title"],
            "text": analysis["sections"].get("full_text", "")[:4000],
        })

    # Similarity matrix
    sim_result = similarity_engine.compare_papers(paper_texts)

    # Comparison table
    comparison = comparator.compare(paper_analyses)

    return JSONResponse(content={
        "papers": paper_analyses,
        "similarity": sim_result,
        "comparison": comparison,
    })


@app.post("/api/similarity")
async def compute_similarity(files: List[UploadFile] = File(...)):
    """Compute pairwise semantic similarity between uploaded papers."""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 PDFs.")

    texts = []
    for f in files:
        pdf_bytes = await f.read()
        proc = pdf_processor.process(pdf_bytes)
        texts.append({"title": f.filename, "text": proc["cleaned_text"]})

    result = similarity_engine.compare_papers(texts)
    return JSONResponse(content=result)


@app.post("/api/search")
async def semantic_search(
    query: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Search across uploaded papers using a natural language query."""
    corpus_texts = []
    titles = []
    for f in files:
        pdf_bytes = await f.read()
        proc = pdf_processor.process(pdf_bytes)
        corpus_texts.append(proc["cleaned_text"])
        titles.append(f.filename)

    scores = similarity_engine.batch_similarity(query, corpus_texts)
    results = sorted(
        [{"title": t, "score": s} for t, s in zip(titles, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return JSONResponse(content={"query": query, "results": results})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
