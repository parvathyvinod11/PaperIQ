"""
Paper Quality Scorer
Computes a composite research quality score using the 6 heuristic NLP components:
Language Sophistication, Coherence, Reasoning Strength, Readability, Citation Quality, Conciseness.
"""

import re
import math
from typing import Dict

class QualityScorer:
    def __init__(self):
        pass

    def _score_language_sophistication(self, text: str, words: list, sentences: list) -> float:
        """lexical diversity + sentence variance"""
        if not words or not sentences:
            return 0.0
        
        # Lexical Diversity (Unique words / total)
        unique_words = len(set([w.lower() for w in words]))
        lex_ratio = unique_words / len(words)
        # Typically long papers have lex ratio around 0.1 - 0.3 depending on length.
        # Short texts have higher. Let's scale based on a logistic curve or simple cap.
        lex_score = min(1.0, lex_ratio / 0.25)
        
        # Sentence Variance (stdev of sentence lengths)
        sent_lengths = [len(s.split()) for s in sentences]
        avg_len = sum(sent_lengths) / len(sent_lengths)
        variance = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        stdev = math.sqrt(variance)
        # Good variance is around 5-15. 
        var_score = min(1.0, stdev / 10.0)
        
        return round((lex_score * 0.5) + (var_score * 0.5), 3)

    def _score_coherence(self, text_lower: str, total_words: int) -> float:
        """however / therefore / consequently / furthermore marker frequency"""
        if total_words == 0:
            return 0.0
        markers = ['however', 'therefore', 'consequently', 'furthermore', 'moreover', 'thus', 'hence', 'additionally']
        count = sum(len(re.findall(r'\b' + m + r'\b', text_lower)) for m in markers)
        freq_per_1000 = (count / total_words) * 1000
        # 3-5 markers per 1000 words is very coherent
        score = min(1.0, freq_per_1000 / 4.0)
        return round(score, 3)

    def _score_reasoning_strength(self, text_lower: str, total_words: int, cite_density: float) -> float:
        """evidence / suggests / implies markers + citation density + hedging"""
        if total_words == 0:
            return 0.0
            
        reasoning_markers = ['evidence', 'suggests', 'implies', 'indicates', 'demonstrates', 'proves']
        hedging_markers = ['might', 'could', 'possibly', 'probably', 'may', 'perhaps', 'likely']
        
        reasoning_count = sum(len(re.findall(r'\b' + m + r'\b', text_lower)) for m in reasoning_markers)
        hedging_count = sum(len(re.findall(r'\b' + m + r'\b', text_lower)) for m in hedging_markers)
        
        # density per 1000 words
        marker_density = ((reasoning_count + hedging_count) / total_words) * 1000
        marker_score = min(1.0, marker_density / 5.0)
        
        # citation density score
        cite_score = min(1.0, cite_density / 1.0) # cite density usually 0-2 per 100 words
        
        return round((marker_score * 0.6) + (cite_score * 0.4), 3)

    def _score_readability(self, words: list, sentences: list) -> float:
        """inverse of average sentence length (simple proxy)"""
        if not sentences:
            return 0.0
        avg_sent_len = len(words) / len(sentences)
        if avg_sent_len <= 0:
            return 0.0
            
        # Target good length = ~15-20 words
        if avg_sent_len <= 15:
            score = 1.0
        else:
            # penalize longer average sentences
            score = 15.0 / avg_sent_len
            
        return round(min(1.0, score), 3)

    def _score_citation_quality(self, citation_data: Dict) -> float:
        """citation pattern density per 100 words"""
        inline = citation_data.get("inline_citations", {})
        cite_density = inline.get("citation_density", 0.0)
        
        # if density is around 0.8 to 1.5 per 100 words, it's very solid
        score = min(1.0, cite_density / 0.8)
        return round(score, 3)

    def _score_conciseness(self, text_lower: str, total_sentences: int) -> float:
        """passive voice ratio penalty"""
        if total_sentences == 0:
            return 0.0
        # very simple heuristic for passive voice
        passive_count = len(re.findall(r'\b(?:is|are|was|were|been|being)\s+\w+ed\b', text_lower))
        passive_ratio = passive_count / total_sentences
        
        # Ideal passive ratio < 0.2 per sentence
        penalty = passive_ratio * 0.5
        score = max(0.0, 1.0 - penalty)
        return round(score, 3)

    def compute_score(
        self,
        text: str,
        sections: Dict[str, str],
        citation_data: Dict,
        keyword_data: Dict,
    ) -> Dict:
        """Compute composite quality score."""
        if not text:
            return {"composite_score": 0, "grade": "F", "breakdown": {}, "weights": {}}
            
        words = text.split()
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        text_lower = text.lower()
        total_words = len(words)
        total_sentences = len(sentences)
        
        cite_density = citation_data.get("inline_citations", {}).get("citation_density", 0.0)

        scores = {
            "Language Sophistication": self._score_language_sophistication(text, words, sentences),
            "Coherence": self._score_coherence(text_lower, total_words),
            "Reasoning Strength": self._score_reasoning_strength(text_lower, total_words, cite_density),
            "Readability": self._score_readability(words, sentences),
            "Citation Quality": self._score_citation_quality(citation_data),
            "Conciseness": self._score_conciseness(text_lower, total_sentences),
        }

        # Equal weighting for the requested 6 heuristics
        composite = sum(scores.values()) / len(scores) if scores else 0.0
        composite = round(composite * 100, 1)  # Convert to 0-100

        # Grade
        if composite >= 80:
            grade = "A"
        elif composite >= 65:
            grade = "B"
        elif composite >= 50:
            grade = "C"
        else:
            grade = "D"

        return {
            "composite_score": composite,
            "grade": grade,
            "breakdown": {k.replace(" ", "_").lower(): round(v * 100, 1) for k, v in scores.items()},
        }
