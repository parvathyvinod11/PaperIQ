"""
Multi-Paper Comparator
Compares multiple research papers across key dimensions.
"""

from typing import Dict, List


class PaperComparator:
    def __init__(self):
        pass

    def compare(self, papers: List[Dict]) -> Dict:
        """
        Build a structured comparison table for multiple papers.
        Each paper dict should have:
          - title, sections, keywords, domain, citations, quality_score
        """
        if not papers:
            return {"comparison_table": [], "summary": "No papers to compare."}

        # Build comparison table rows
        table_rows = []
        for p in papers:
            sections = p.get("sections", {})
            kw_data = p.get("keywords", {})
            citation_data = p.get("citations", {})
            quality = p.get("quality_score", {})
            domain_data = p.get("domain", {})

            entities = kw_data.get("entities", {})
            top_kws = [k["keyword"] for k in kw_data.get("top_keywords", [])[:5]]
            algorithms = entities.get("algorithms", [])
            datasets = entities.get("datasets", [])
            metrics_used = entities.get("metrics", [])

            row = {
                "title": p.get("title", "Unknown"),
                "domain": domain_data.get("primary_domain", "N/A"),
                "methodology_summary": (sections.get("methodology", "")[:200] + "...")
                if sections.get("methodology")
                else "N/A",
                "algorithms": ", ".join(algorithms[:3]) if algorithms else "N/A",
                "datasets": ", ".join(datasets[:3]) if datasets else "N/A",
                "metrics": ", ".join(metrics_used[:3]) if metrics_used else "N/A",
                "top_keywords": ", ".join(top_kws),
                "reference_count": citation_data.get("reference_count", 0),
                "quality_score": quality.get("composite_score", "N/A"),
                "abstract": sections.get("abstract", "")[:300] + "..."
                if sections.get("abstract")
                else "N/A",
            }
            table_rows.append(row)

        # Summary insights
        domains = [r["domain"] for r in table_rows]
        same_domain = len(set(domains)) == 1

        summary_lines = []
        if same_domain:
            summary_lines.append(
                f"All papers are in the **{domains[0]}** domain — directly comparable."
            )
        else:
            summary_lines.append(
                f"Papers span multiple domains: {', '.join(set(domains))}."
            )

        scores = [r["quality_score"] for r in table_rows if isinstance(r["quality_score"], (int, float))]
        if scores:
            best_idx = scores.index(max(scores))
            summary_lines.append(
                f"Highest quality score: **{table_rows[best_idx]['title']}** ({max(scores)}/100)."
            )

        return {
            "comparison_table": table_rows,
            "summary": " ".join(summary_lines),
            "paper_count": len(papers),
        }
