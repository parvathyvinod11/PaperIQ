"""
PDF Processing Module
Extracts and cleans text from PDF research papers.
"""

import fitz  # PyMuPDF
import pdfplumber
import re
import io
from typing import Optional


class PDFProcessor:
    def __init__(self):
        pass

    def extract_text_pymupdf(self, pdf_bytes: bytes) -> str:
        """Extract text using PyMuPDF (primary)."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text("text"))
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            return ""

    def extract_text_pdfplumber(self, pdf_bytes: bytes) -> str:
        """Extract text using pdfplumber (fallback)."""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        text_parts.append(txt)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            return ""

    def get_metadata(self, pdf_bytes: bytes) -> dict:
        """Extract PDF metadata."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            meta = doc.metadata
            page_count = doc.page_count
            doc.close()
            return {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "page_count": page_count,
            }
        except Exception:
            return {"title": "", "author": "", "subject": "", "creator": "", "page_count": 0}

    def clean_text(self, text: str) -> str:
        """Clean extracted text for NLP processing."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        # Remove page numbers (standalone numbers on a line)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove hyphenation
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Remove strange unicode characters
        text = re.sub(r'[^\x00-\x7F\u00C0-\u017F\u0370-\u03FF]', ' ', text)
        text = text.strip()
        return text

    def process(self, pdf_bytes: bytes) -> dict:
        """Full processing pipeline: extract + clean text."""
        metadata = self.get_metadata(pdf_bytes)

        # Try PyMuPDF first
        raw_text = self.extract_text_pymupdf(pdf_bytes)
        if len(raw_text.strip()) < 200:
            # Fallback to pdfplumber
            raw_text = self.extract_text_pdfplumber(pdf_bytes)

        cleaned_text = self.clean_text(raw_text)

        if not metadata.get("title") or "Untitled" in metadata.get("title", ""):
            # Heuristic: the first non-empty lengthy line of a paper is usually its title
            for line in raw_text.split('\n'):
                line_clean = line.strip()
                if line_clean and len(line_clean) > 8:
                    metadata["title"] = line_clean
                    break

        return {
            "metadata": metadata,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "word_count": len(cleaned_text.split()),
            "char_count": len(cleaned_text),
        }
