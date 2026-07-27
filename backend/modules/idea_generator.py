"""
Research Idea Generator
Converts paper insights into student project ideas and research extensions.
"""

import re
import os
import json
from typing import Dict, List


IDEA_TEMPLATES = [
    "Extend {domain} techniques to the {application} domain using {method}",
    "Build a dataset for {topic} comparable to {dataset}",
    "Implement a real-time {application} system using {method}",
    "Compare the performance of {method} against newer architectures on {dataset}",
    "Apply {method} to low-resource or multilingual settings",
    "Create an explainable version of {method} for {application}",
    "Combine {method} with {secondary_method} for improved {metric} performance",
    "Evaluate {method} for fairness and bias in {application}",
    "Federated learning adaptation of {method} for privacy-preserving {application}",
    "Lightweight mobile-friendly version of {method} for edge deployment",
]


class IdeaGenerator:
    def __init__(self):
        pass

    def _gemini_generate_ideas(self, domain: str, keyword_data: Dict, gaps: List[str], text_sample: str) -> Dict:
        """Use Gemini to generate creative context-aware research ideas."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {}
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            methods = self._extract_methods(keyword_data)
            method_str = ", ".join(methods)
            gaps_str = "\n".join(f"- {g}" for g in gaps[:3]) if gaps else "None provided."
            
            prompt = f"""You are an expert AI research advisor. Based on the following context, generate highly specific and creative research ideas for a student or researcher.
Domain: {domain}
Key Methods/Algorithms: {method_str}
Identified Gaps:
{gaps_str}

Paper Abstract/Intro Sample:
{text_sample[:3000]}

Return ONLY a valid JSON object with the following keys, containing lists of string ideas:
"research_extensions": 6 advanced ideas building on the paper.
"gap_based_ideas": 3 ideas explicitly addressing the gaps.
"implementation_projects": 4 practical software engineering or replication projects.
"dataset_ideas": 3 ideas related to collecting or modifying datasets.
"""
            response = model.generate_content(prompt)
            match = re.search(r"\{.*\}", response.text.strip(), re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {}
        except Exception:
            return {}

    def _extract_methods(self, keyword_data: Dict) -> List[str]:
        entities = keyword_data.get("entities", {})
        algorithms = entities.get("algorithms", [])
        top_kw = [k["keyword"] for k in keyword_data.get("top_keywords", [])[:5]]
        methods = algorithms[:3] + top_kw[:3]
        return methods if methods else ["the proposed method"]

    def _extract_datasets(self, keyword_data: Dict) -> List[str]:
        entities = keyword_data.get("entities", {})
        datasets = entities.get("datasets", [])
        return datasets if datasets else ["benchmark datasets"]

    def _extract_metrics(self, keyword_data: Dict) -> List[str]:
        entities = keyword_data.get("entities", {})
        metrics = entities.get("metrics", [])
        return metrics if metrics else ["accuracy"]

    def generate_ideas(
        self,
        sections: Dict[str, str],
        domain: str,
        keyword_data: Dict,
        gaps: List[str],
    ) -> Dict:
        """Generate research ideas and project suggestions."""

        methods = self._extract_methods(keyword_data)
        datasets = self._extract_datasets(keyword_data)
        metrics = self._extract_metrics(keyword_data)

        method = methods[0] if methods else "deep learning"
        secondary = methods[1] if len(methods) > 1 else "transfer learning"
        dataset = datasets[0] if datasets else "a public dataset"
        metric = metrics[0] if metrics else "accuracy"
        application = domain.lower()

        # Try Gemini first for extremely high quality ideas
        text_sample = sections.get("abstract", "") + "\n" + sections.get("introduction", "")
        gemini_ideas = self._gemini_generate_ideas(domain, keyword_data, gaps, text_sample)
        if gemini_ideas and isinstance(gemini_ideas, dict) and "research_extensions" in gemini_ideas:
            return gemini_ideas

        # Fallback to templated ideas
        ideas = []
        for tmpl in IDEA_TEMPLATES[:6]:
            idea = tmpl.format(
                domain=domain,
                application=application,
                method=method,
                secondary_method=secondary,
                dataset=dataset,
                topic=domain,
                metric=metric,
            )
            ideas.append(idea)

        # Gap-based ideas
        gap_ideas = []
        for gap in gaps[:3]:
            # Shorten gaps for readability
            short_gap = gap[:120] + ("..." if len(gap) > 120 else "")
            gap_ideas.append(f"Address the gap: {short_gap}")

        # Implementation project ideas
        impl_projects = [
            f"Build a web app that applies {method} for {application} analysis",
            f"Create a Python library wrapping {method} with a simple API",
            f"Fine-tune {method} on a domain-specific dataset for {application}",
            f"Reproduce the paper's experiments and publish results on {dataset}",
            "Develop a benchmarking toolkit comparing the methods in this paper",
        ]

        return {
            "research_extensions": ideas,
            "gap_based_ideas": gap_ideas,
            "implementation_projects": impl_projects[:4],
            "dataset_ideas": [
                f"Collect a multilingual dataset for {application}",
                f"Annotate a domain-specific {application} corpus",
                f"Create a synthetic dataset for evaluating {method}",
            ],
        }
