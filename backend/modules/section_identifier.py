"""
Section Identification Module
Detects and extracts standard research paper sections.
"""

import re
from typing import Dict, List, Optional


# Common section header patterns
SECTION_PATTERNS = {
    "abstract": [
        r'\bABSTRACT\b',
        r'\bSummary\b',
    ],
    "introduction": [
        r'\b(?:1\.?\s+)?INTRODUCTION\b',
        r'\bI\.\s+INTRODUCTION\b',
    ],
    "related_work": [
        r'\bRELATED\s+WORK\b',
        r'\bLITERATURE\s+REVIEW\b',
        r'\bBACKGROUND\b',
        r'\bPRIOR\s+WORK\b',
    ],
    "methodology": [
        r'\bMETHODOLOGY\b',
        r'\bMETHOD\b',
        r'\bAPPROACH\b',
        r'\bPROPOSED\s+(?:METHOD|SYSTEM|APPROACH|FRAMEWORK)\b',
        r'\bSYSTEM\s+DESIGN\b',
    ],
    "experiments": [
        r'\bEXPERIMENT',
        r'\bEVALUATION\b',
        r'\bEXPERIMENTAL\s+SETUP\b',
        r'\bIMPLEMENTATION\b',
    ],
    "results": [
        r'\bRESULTS?\b',
        r'\bFINDINGS\b',
        r'\bRESULTS?\s+AND\s+DISCUSSION\b',
    ],
    "conclusion": [
        r'\bCONCLUSION',
        r'\bSUMMARY\s+AND\s+CONCLUSION',
        r'\bCONCLUDING\s+REMARKS\b',
    ],
    "future_work": [
        r'\bFUTURE\s+WORK\b',
        r'\bFUTURE\s+DIRECTIONS\b',
    ],
    "acknowledgements": [
        r'\bACKNOWLEDGEMENT',
        r'\bACKNOWLEDGMENT',
    ],
}


class SectionIdentifier:
    def __init__(self):
        self.compiled_patterns = {
            name: [re.compile(p, re.IGNORECASE) for p in pats]
            for name, pats in SECTION_PATTERNS.items()
        }

    def _find_section_boundaries(self, text: str) -> List[Dict]:
        """Find all section headers and their positions."""
        lines = text.split('\n')
        headings = []

        for line_num, line in enumerate(lines):
            stripped = line.strip()
            # A heading line is typically short (< 100 chars) and matches a pattern
            if len(stripped) == 0 or len(stripped) > 120:
                continue
            for section_name, patterns in self.compiled_patterns.items():
                for pat in patterns:
                    if pat.search(stripped):
                        headings.append({
                            "section": section_name,
                            "line_num": line_num,
                            "header_text": stripped,
                        })
                        break

        return headings

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from text."""
        lines = text.split('\n')
        boundaries = self._find_section_boundaries(text)

        sections: Dict[str, str] = {name: "" for name in SECTION_PATTERNS}
        sections["full_text"] = text

        if not boundaries:
            # Try to extract abstract heuristically
            abs_match = re.search(r'(?:abstract|summary)[:\s]+(.+?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
            if abs_match:
                sections["abstract"] = abs_match.group(1).strip()
            return sections

        # For each boundary, collect text until next boundary
        for i, boundary in enumerate(boundaries):
            start = boundary["line_num"] + 1
            end = boundaries[i + 1]["line_num"] if i + 1 < len(boundaries) else len(lines)
            section_text = "\n".join(lines[start:end]).strip()
            section_name = boundary["section"]
            # Append (in case a section appears multiple times under different headings)
            if sections[section_name]:
                sections[section_name] += "\n\n" + section_text
            else:
                sections[section_name] = section_text

        # Also try to detect abstract from beginning of document if not found
        if not sections["abstract"]:
            abs_match = re.search(
                r'(?:abstract|summary)[:\s\-]+(.+?)(?:\n\s*\n|\Z)',
                text[:3000],
                re.IGNORECASE | re.DOTALL,
            )
            if abs_match:
                sections["abstract"] = abs_match.group(1).strip()[:1500]

        return sections

    def get_section_stats(self, sections: Dict[str, str]) -> Dict:
        """Compute word counts per section."""
        return {
            name: len(content.split())
            for name, content in sections.items()
            if name != "full_text"
        }
