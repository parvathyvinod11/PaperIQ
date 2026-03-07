"""
Citation Analyzer Module
Extracts and analyzes citations and references from research papers.
"""

import re
from typing import Dict, List, Tuple
from collections import Counter


class CitationAnalyzer:
    def __init__(self):
        pass

    def extract_references(self, text: str) -> List[str]:
        """Extract reference list from paper text."""
        # Find reference section
        ref_match = re.search(
            r'\b(?:REFERENCES|BIBLIOGRAPHY|WORKS\s+CITED)\b([\s\S]+)',
            text,
            re.IGNORECASE,
        )
        if not ref_match:
            return []

        ref_text = ref_match.group(1)
        # Split by numbered entries [1], [2] ... or 1. 2. etc
        refs = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s+', ref_text)
        refs = [r.strip() for r in refs if len(r.strip()) > 20]
        return refs[:100]

    def extract_inline_citations(self, text: str) -> Dict:
        """Extract inline citation counts."""
        # Bracket citations: [1], [2,3], [4-6]
        bracket_cites = re.findall(r'\[(\d+(?:[,\-]\d+)*)\]', text)
        # Author-year citations: (Smith, 2020), (Jones et al., 2019)
        author_year_cites = re.findall(
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)\)', text
        )

        all_bracket = []
        for cite in bracket_cites:
            # Expand ranges like 4-6
            if '-' in cite:
                parts = cite.split('-')
                try:
                    all_bracket.extend(range(int(parts[0]), int(parts[1]) + 1))
                except Exception:
                    pass
            else:
                for c in cite.split(','):
                    try:
                        all_bracket.append(int(c.strip()))
                    except Exception:
                        pass

        return {
            "bracket_citation_count": len(bracket_cites),
            "author_year_citation_count": len(author_year_cites),
            "unique_citations": len(set(all_bracket)),
            "citation_density": round(len(bracket_cites) / max(len(text.split()), 1) * 100, 3),
        }

    def analyze_year_distribution(self, references: List[str]) -> Dict[str, int]:
        """Extract publication years from references."""
        years: List[int] = []
        for ref in references:
            year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', ref)
            for y in year_matches:
                years.append(int(y))

        year_counts: Dict[str, int] = {}
        for y in years:
            year_counts[str(y)] = year_counts.get(str(y), 0) + 1

        return dict(sorted(year_counts.items()))

    def get_influential_authors(self, references: List[str], top_n: int = 10) -> List[Dict]:
        """Find most cited authors in reference list."""
        author_counts: Counter = Counter()
        for ref in references:
            # Simple first-author extraction: "Surname, F." or "Surname et al."
            match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)?)', ref.strip())
            if match:
                author_counts[match.group(1)] += 1

        return [
            {"author": author, "count": count}
            for author, count in author_counts.most_common(top_n)
        ]

    def analyze(self, text: str) -> Dict:
        """Full citation analysis."""
        references = self.extract_references(text)
        inline = self.extract_inline_citations(text)
        year_dist = self.analyze_year_distribution(references)
        authors = self.get_influential_authors(references)

        return {
            "reference_count": len(references),
            "references": references[:20],  # Return first 20
            "inline_citations": inline,
            "year_distribution": year_dist,
            "top_authors": authors,
        }
