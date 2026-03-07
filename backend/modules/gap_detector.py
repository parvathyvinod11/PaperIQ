"""
Research Gap Detection Module
Detects potential research gaps by analyzing limitations and unsolved problems.
"""

import re
from typing import List, Dict

import nltk
from nltk.tokenize import sent_tokenize

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# Patterns that indicate research gaps / limitations
GAP_PATTERNS = [
    r'\b(?:limitation|limitation of|limited by|limited to)\b',
    r'\b(?:future work|future research|future direction|future study)\b',
    r'\b(?:however|nonetheless|nevertheless|although|despite)\b',
    r'\b(?:not (?:studied|explored|addressed|considered|investigated|tested))\b',
    r'\b(?:remain(?:s)? (?:an? )?(?:open|unsolved|unresolved|unexplored|unclear|challenging))\b',
    r'\b(?:lack(?:s|ing)? of|absence of|insufficient|inadequate)\b',
    r'\b(?:challenging|challenge|difficult|difficulty|obstacle|barrier)\b',
    r'\b(?:further (?:research|study|investigation|exploration|work) (?:is )?(?:needed|required|necessary))\b',
    r'\b(?:gap in|gap between|research gap)\b',
    r'\b(?:beyond the scope|out of scope)\b',
    r'\b(?:cannot|could not|unable to|fail(?:s|ed)? to)\b',
    r'\b(?:one (?:major |key )?(?:limitation|drawback|shortcoming|weakness))\b',
]

FUTURE_PATTERNS = [
    r'\b(?:future work|future research|in future|as future)\b',
    r'\b(?:we plan to|we intend to|we will|will be (?:explored|investigated|studied))\b',
    r'\b(?:promising direction|open problem|interesting avenue)\b',
    r'\b(?:extend(?:ing)?|improve(?:ment)?|enhance) (?:the|our) (?:work|model|approach|method|system)\b',
]


class GapDetector:
    def __init__(self):
        self.gap_patterns = [re.compile(p, re.IGNORECASE) for p in GAP_PATTERNS]
        self.future_patterns = [re.compile(p, re.IGNORECASE) for p in FUTURE_PATTERNS]

    def _matches(self, sentence: str, patterns: List) -> bool:
        return any(p.search(sentence) for p in patterns)

    def detect_gaps(self, text: str, sections: Dict[str, str]) -> Dict:
        """Detect research gaps from text."""
        # Focus on conclusion, future work, and introduction sections
        target_text = " ".join([
            sections.get("conclusion", ""),
            sections.get("future_work", ""),
            sections.get("results", ""),
            sections.get("introduction", "")[:1000],
        ])
        if not target_text.strip():
            target_text = text[:6000]

        sentences = sent_tokenize(target_text)

        gap_sentences = []
        future_sentences = []
        limitation_sentences = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30:
                continue
            if self._matches(sent, self.gap_patterns):
                gap_sentences.append(sent)
            if self._matches(sent, self.future_patterns):
                future_sentences.append(sent)
            if re.search(r'\blimitation', sent, re.IGNORECASE):
                limitation_sentences.append(sent)

        # Deduplicate
        gap_sentences = list(dict.fromkeys(gap_sentences))[:8]
        future_sentences = list(dict.fromkeys(future_sentences))[:5]
        limitation_sentences = list(dict.fromkeys(limitation_sentences))[:5]

        return {
            "identified_gaps": gap_sentences,
            "future_directions": future_sentences,
            "limitations": limitation_sentences,
            "gap_count": len(gap_sentences),
        }
