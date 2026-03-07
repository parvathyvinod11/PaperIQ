"""
Paper Summarization Module
Extractive summarization using sentence scoring (TF-IDF + position weighting).
Optionally uses HuggingFace BART/T5 for abstractive summarization.
"""

import re
import math
from typing import Dict, List, Optional
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

for pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


class Summarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self._abstractive_model = None
        self._abstractive_tokenizer = None

    def _score_sentences(self, sentences: List[str], top_n: int = 5) -> List[str]:
        """Score sentences using TF-IDF-like approach and position weighting."""
        if not sentences:
            return []

        # Build word frequency table
        word_freq: Dict[str, int] = Counter()
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word.isalpha() and word not in self.stop_words:
                    word_freq[word] += 1

        if not word_freq:
            return sentences[:top_n]

        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] /= max_freq

        # Score each sentence
        sentence_scores: Dict[int, float] = {}
        for idx, sent in enumerate(sentences):
            score = 0.0
            words = word_tokenize(sent.lower())
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
            # Normalize by sentence length to avoid bias
            if len(words) > 0:
                score /= len(words)
            # Give slight boost to early sentences (abstract effect)
            position_boost = 1.0 / (1 + idx * 0.05)
            sentence_scores[idx] = score * position_boost

        # Pick top N sentences and return in original order
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]
        top_indices_sorted = sorted(top_indices)
        return [sentences[i] for i in top_indices_sorted]

    def extractive_summarize(self, text: str, top_n: int = 6) -> str:
        """Generate extractive summary."""
        if not text or len(text.strip()) < 100:
            return text
        sentences = sent_tokenize(text)
        if len(sentences) <= top_n:
            return text
        top_sentences = self._score_sentences(sentences, top_n=top_n)
        return " ".join(top_sentences)

    def summarize_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Summarize each section of the paper."""
        summaries = {}
        section_configs = {
            "abstract": 3,
            "introduction": 4,
            "methodology": 5,
            "results": 5,
            "conclusion": 4,
            "related_work": 4,
        }
        for section, top_n in section_configs.items():
            content = sections.get(section, "")
            if content and len(content.split()) > 50:
                summaries[section] = self.extractive_summarize(content, top_n=top_n)
            else:
                summaries[section] = content

        # Overall summary using most informative sections
        overall_text = " ".join([
            sections.get("abstract", ""),
            sections.get("conclusion", ""),
            sections.get("results", ""),
        ])
        summaries["overall"] = self.extractive_summarize(overall_text, top_n=8)

        return summaries

    def get_key_contributions(self, text: str) -> List[str]:
        """Extract sentences describing contributions."""
        contribution_patterns = [
            r'(?:we (?:propose|present|introduce|develop|demonstrate|show)|'
            r'this paper (?:proposes|presents|introduces|describes)|'
            r'our (?:approach|method|system|framework|model)|'
            r'(?:novel|new|proposed) (?:approach|method|framework|model|algorithm))',
        ]
        sentences = sent_tokenize(text[:6000])
        contributions = []
        import re
        for sent in sentences:
            for pat in contribution_patterns:
                if re.search(pat, sent, re.IGNORECASE):
                    contributions.append(sent.strip())
                    break
        return contributions[:5]
