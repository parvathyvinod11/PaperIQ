"""
PaperIQ — Phase 2 Backend (FastAPI)
Run: uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import re, io, math
from collections import Counter
from typing import Optional

# ── optional heavy deps (graceful fallback) ──────────────────────────────────
try:
    import fitz                          # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import pdfplumber
    HAS_PLUMBER = True
except ImportError:
    HAS_PLUMBER = False

try:
    from textblob import TextBlob
    HAS_BLOB = True
except ImportError:
    HAS_BLOB = False

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PaperIQ API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─────────────────────────────────────────────────────────────────────────────
# AUTH  (simple token store — swap with JWT/DB in production)
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_TOKENS: dict[str, dict] = {}

class LoginRequest(BaseModel):
    email: str
    role: str

@app.post("/auth/login")
def login(req: LoginRequest):
    if "@" not in req.email:
        raise HTTPException(400, "Invalid email")
    token = f"piq_{abs(hash(req.email + req.role)):016x}"
    ACTIVE_TOKENS[token] = {"email": req.email, "role": req.role}
    return {"token": token, "user": ACTIVE_TOKENS[token]}

@app.post("/auth/logout")
def logout(creds: HTTPAuthorizationCredentials = Depends(security)):
    ACTIVE_TOKENS.pop(creds.credentials, None)
    return {"status": "logged_out"}

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not creds or creds.credentials not in ACTIVE_TOKENS:
        raise HTTPException(401, "Invalid or missing token")
    return ACTIVE_TOKENS[creds.credentials]

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN CLASSIFIER  (keyword-probability approach, no ML deps)
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Computer Science / AI":        ["neural", "algorithm", "machine learning", "deep learning",
                                      "classification", "dataset", "accuracy", "model", "training",
                                      "inference", "transformer", "reinforcement", "convolutional",
                                      "optimization", "feature extraction", "nlp"],
    "Biomedical / Health Sciences": ["patient", "clinical", "diagnosis", "treatment", "disease",
                                      "genome", "protein", "cell", "biomarker", "therapeutic",
                                      "epidemiology", "pathogen", "vaccine", "mortality", "cohort"],
    "Physics / Engineering":        ["velocity", "quantum", "semiconductor", "circuit", "thermal",
                                      "entropy", "waveform", "antenna", "torque", "photon",
                                      "electromagnetic", "nanotube", "spectroscopy", "frequency"],
    "Economics / Finance":          ["gdp", "inflation", "market", "fiscal", "monetary", "equity",
                                      "portfolio", "regression", "elasticity", "demand", "supply",
                                      "trade", "currency", "hedge", "volatility", "yield"],
    "Environmental Science":        ["carbon", "climate", "ecosystem", "biodiversity", "emission",
                                      "deforestation", "pollution", "renewable", "glacier",
                                      "precipitation", "habitat", "species", "ozone", "watershed"],
    "Social Sciences / Psychology": ["behavior", "cognitive", "survey", "respondent", "attitude",
                                      "perception", "qualitative", "ethnography", "intervention",
                                      "socioeconomic", "identity", "motivation", "wellbeing"],
    "Mathematics / Statistics":     ["theorem", "proof", "convergence", "stochastic", "distribution",
                                      "bayesian", "eigenvalue", "topology", "manifold", "integral",
                                      "variance", "hypothesis", "estimator", "probability"],
    "Chemistry / Materials":        ["compound", "synthesis", "catalyst", "polymer", "electrode",
                                      "spectroscopy", "crystalline", "reagent", "titration",
                                      "oxidation", "nanoparticle", "lattice", "solvent"],
}

def classify_domain(text: str) -> dict:
    lower = text.lower()
    words = re.findall(r'\b[a-z]{4,}\b', lower)
    word_set = Counter(words)
    scores: dict[str, float] = {}
    for domain, kws in DOMAIN_KEYWORDS.items():
        hit = sum(word_set.get(k.replace(" ", ""), 0) +
                  (lower.count(k) if " " in k else 0) for k in kws)
        scores[domain] = hit
    total = sum(scores.values()) or 1
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "primary":    ranked[0][0],
        "secondary":  ranked[1][0] if ranked[1][1] > 0 else "N/A",
        "confidence": round(ranked[0][1] / total * 100, 1),
        "all_scores": {d: round(s / total * 100, 1) for d, s in ranked},
    }

# ─────────────────────────────────────────────────────────────────────────────
# NLP ENGINE
# ─────────────────────────────────────────────────────────────────────────────
COHERENCE_MARKERS  = ["however","therefore","thus","consequently","furthermore",
                       "moreover","alternatively","in contrast","despite","whereas"]
REASONING_MARKERS  = ["because","evidence","suggests","implies","demonstrates",
                       "indicates","reveals","confirms","supports","hypothesize"]
HEDGING_MARKERS    = ["may","might","could","possibly","likely","arguably",
                       "approximately","presumably","seemingly","appears to"]
CITATION_PATTERN   = re.compile(r'\[\d+\]|\(\w+[\s,]+\d{4}\)')
PASSIVE_PATTERN    = re.compile(r'\b(is|are|was|were|been|be)\s+\w+ed\b')

def _sentences(text: str) -> list[str]:
    """Lightweight sentence splitter."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]

def _words(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Z]+\b', text)

def lexical_diversity(words: list[str]) -> float:
    if not words: return 0.0
    return round(len(set(w.lower() for w in words)) / len(words) * 100, 1)

def avg_sentence_length(sents: list[str]) -> float:
    lengths = [len(_words(s)) for s in sents]
    return round(sum(lengths) / len(lengths), 1) if lengths else 0

def sentence_length_variance(sents: list[str]) -> float:
    lengths = [len(_words(s)) for s in sents]
    if len(lengths) < 2: return 0.0
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return round(math.sqrt(variance), 1)

def flesch_kincaid_grade(text: str, sents: list[str], words: list[str]) -> float:
    syllables = sum(_count_syllables(w) for w in words)
    n_words = len(words) or 1
    n_sents = len(sents) or 1
    asl = n_words / n_sents
    asw = syllables / n_words
    score = 0.39 * asl + 11.8 * asw - 15.59
    return round(max(0, min(20, score)), 1)

def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiou"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def citation_density(text: str, words: list[str]) -> float:
    n = len(CITATION_PATTERN.findall(text))
    return round(n / (len(words) / 100) if words else 0, 2)

def passive_voice_ratio(text: str, sents: list[str]) -> float:
    passive = len(PASSIVE_PATTERN.findall(text))
    return round(passive / (len(sents) or 1) * 100, 1)

def abstract_word_ratio(words: list[str]) -> float:
    abstract = ["theory","concept","framework","model","approach","methodology",
                 "perspective","paradigm","construct","analysis","evaluation"]
    hits = sum(1 for w in words if w.lower() in abstract)
    return round(hits / (len(words) or 1) * 100, 1)

def compute_scores(text: str) -> dict:
    sents = _sentences(text)
    words = _words(text)
    lower = text.lower()

    # Raw feature values
    asl    = avg_sentence_length(sents)
    slv    = sentence_length_variance(sents)
    ld     = lexical_diversity(words)
    fk     = flesch_kincaid_grade(text, sents, words)
    coh_c  = sum(lower.count(m) for m in COHERENCE_MARKERS)
    rea_c  = sum(lower.count(m) for m in REASONING_MARKERS)
    hed_c  = sum(lower.count(m) for m in HEDGING_MARKERS)
    cd     = citation_density(text, words)
    pvr    = passive_voice_ratio(text, sents)
    awr    = abstract_word_ratio(words)

    # Normalise to 0-100
    language    = min(100, ld * 0.6 + awr * 2 + min(slv, 15) * 2 + 20)
    coherence   = min(100, coh_c * 4 + 40)
    reasoning   = min(100, rea_c * 5 + hed_c * 2 + cd * 3 + 30)
    readability = max(0, min(100, 110 - asl))
    citation_q  = min(100, cd * 10 + 20)
    passive_pen = max(0, 100 - pvr * 1.5)   # lower passive = better

    composite = (
        language    * 0.20 +
        coherence   * 0.20 +
        reasoning   * 0.20 +
        readability * 0.15 +
        citation_q  * 0.15 +
        passive_pen * 0.10
    )

    sentiment = 0.0
    if HAS_BLOB:
        sentiment = round(TextBlob(text[:5000]).sentiment.polarity, 3)

    long_sents = [s for s in sents if len(_words(s)) > 35]

    return {
        "scores": {
            "Language Sophistication": round(language, 1),
            "Coherence":               round(coherence, 1),
            "Reasoning Strength":      round(reasoning, 1),
            "Readability":             round(readability, 1),
            "Citation Quality":        round(citation_q, 1),
            "Conciseness":             round(passive_pen, 1),
            "Composite":               round(composite, 1),
        },
        "metrics": {
            "Total Words":             len(words),
            "Total Sentences":         len(sents),
            "Avg Sentence Length":     asl,
            "Sentence Length Variance":slv,
            "Lexical Diversity (%)":   ld,
            "Flesch-Kincaid Grade":    fk,
            "Citation Density":        cd,
            "Passive Voice (%)":       pvr,
            "Abstract Word Ratio (%)": awr,
            "Coherence Markers":       coh_c,
            "Reasoning Markers":       rea_c,
            "Hedging Markers":         hed_c,
        },
        "sentiment": sentiment,
        "long_sentences": long_sents[:5],
    }

# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_fitz(file_bytes: bytes) -> str:
    if not HAS_FITZ:
        return ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(p.get_text() for p in doc)

def extract_sections(file_bytes: bytes) -> dict[str, str]:
    if not HAS_FITZ:
        return {"Full Text": extract_text_fitz(file_bytes)}
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    sections: dict[str, str] = {}
    current = "Front Matter"
    content: list[str] = []

    font_sizes = []
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        font_sizes.append(round(s["size"]))
    body_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 10

    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if "lines" not in b:
                continue
            for l in b["lines"]:
                line_text = " ".join(s["text"] for s in l["spans"]).strip()
                if not line_text or len(line_text) < 2:
                    continue
                is_bold = any("Bold" in s["font"] or (s["flags"] & 16) for s in l["spans"])
                size    = max(s["size"] for s in l["spans"])
                if (size > body_size + 1 and is_bold) or (line_text.isupper() and len(line_text) < 60):
                    if content:
                        sections[current] = " ".join(content)
                    current = line_text
                    content = []
                else:
                    content.append(line_text)
    if content:
        sections[current] = " ".join(content)
    return sections

def extract_metadata(text: str) -> dict:
    doi_m = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', text)
    doi   = doi_m.group(0) if doi_m else "Not Detected"
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
    bad   = ["IEEE","PROCEEDINGS","VOL.","ISSUE","COPYRIGHT","ISSN","JOURNAL"]
    title = "Not Detected"
    for line in lines[:15]:
        if not any(w in line.upper() for w in bad):
            title = line
            break

    # author pattern: capitalised words separated by commas/and
    author_m = re.search(
        r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:[,\s]+(?:and\s+)?[A-Z][a-z]+\s+[A-Z][a-z]+)*)',
        text[:2000]
    )
    authors = author_m.group(0) if author_m else "Not Detected"

    year_m  = re.search(r'\b(19|20)\d{2}\b', text[:500])
    year    = year_m.group(0) if year_m else "Not Detected"

    journal_m = re.search(
        r'(?:journal|conference|proceedings|transactions)\s+(?:of\s+)?[\w\s&]+',
        text[:1000], re.IGNORECASE
    )
    journal = journal_m.group(0).strip() if journal_m else "Not Detected"

    return {"title": title, "doi": doi, "authors": authors, "year": year, "journal": journal}

def extract_tables(file_bytes: bytes) -> list[list[list]]:
    if not HAS_PLUMBER:
        return []
    tables = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                if tbl and len(tbl) > 1:
                    tables.append(tbl)
    return tables

def top_keywords(text: str, n: int = 15) -> list[str]:
    stop = {"paper","study","using","results","analysis","based","method","methods",
            "proposed","approach","system","model","data","show","also","used",
            "between","their","within","other","which","these","those","there"}
    words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
    return [k for k, _ in Counter(words).most_common(n*2) if k not in stop][:n]

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze/pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    file_bytes = await file.read()
    full_text  = extract_text_fitz(file_bytes)
    if not full_text.strip():
        raise HTTPException(422, "Could not extract text from PDF.")

    sections   = extract_sections(file_bytes)
    meta       = extract_metadata(full_text)
    nlp        = compute_scores(full_text)
    domain     = classify_domain(full_text)
    keywords   = top_keywords(full_text)
    tables     = extract_tables(file_bytes)
    page_count = 0
    if HAS_FITZ:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)

    return {
        "metadata":   meta,
        "domain":     domain,
        "nlp":        nlp,
        "sections":   sections,
        "keywords":   keywords,
        "tables":     tables,
        "page_count": page_count,
    }


@app.post("/analyze/text")
async def analyze_text(
    payload: dict,
    user: dict = Depends(get_current_user),
):
    text = payload.get("text", "")
    if len(text) < 100:
        raise HTTPException(400, "Text too short (min 100 chars).")
    nlp     = compute_scores(text)
    domain  = classify_domain(text)
    keywords= top_keywords(text)
    return {"nlp": nlp, "domain": domain, "keywords": keywords}


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "fitz":       HAS_FITZ,
        "pdfplumber": HAS_PLUMBER,
        "textblob":   HAS_BLOB,
    }
