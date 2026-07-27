"""
LLM Chat Module – PaperIQ
Provides a chatbot interface for Q&A over an analyzed research paper.
Uses Google Gemini API (gemini-2.0-flash) as the primary LLM.
Falls back to a rule-based responder if no API key is configured.
"""

import os
import re
from typing import List, Dict, Optional


# ──────────────────────────────────────────────
# Gemini Client (lazy-loaded)
# ──────────────────────────────────────────────

def _get_gemini_client():
    """Return a configured Gemini GenerativeModel, or None if unavailable."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    except Exception:
        return None


# ──────────────────────────────────────────────
# System prompt builder from analysis data
# ──────────────────────────────────────────────

def build_system_prompt(paper_context: dict) -> str:
    """Build a rich system prompt from the paper's analysis result."""
    title = paper_context.get("title", "Unknown Paper")
    domain = paper_context.get("domain", {}).get("primary_domain", "N/A")
    summaries = paper_context.get("summaries", {})
    overall = summaries.get("overall", "")
    methodology = summaries.get("methodology", "")
    results = summaries.get("results", "")
    conclusion = summaries.get("conclusion", "")
    contributions = paper_context.get("contributions", [])
    gaps = paper_context.get("gaps", {}).get("identified_gaps", [])
    kws = paper_context.get("keywords", {}).get("top_keywords", [])
    top_keywords = ", ".join([k["keyword"] for k in kws[:10]]) if kws else "N/A"
    ideas = paper_context.get("ideas", {})
    quality = paper_context.get("quality", {})
    score = quality.get("composite_score", "N/A")
    grade = quality.get("grade", "N/A")
    sections = paper_context.get("sections", {})
    abstract = sections.get("abstract", "")[:800]

    contributions_text = "\n".join(f"- {c}" for c in contributions[:5]) if contributions else "N/A"
    gaps_text = "\n".join(f"- {g}" for g in gaps[:5]) if gaps else "N/A"
    research_ideas = ideas.get("research_extensions", [])
    ideas_text = "\n".join(f"- {i}" for i in research_ideas[:4]) if research_ideas else "N/A"

    return f"""You are PaperIQ Assistant, an expert AI research assistant specializing in academic paper analysis.

You have been provided with a fully analyzed research paper. Your job is to answer questions about this specific paper accurately, helpfully, and concisely. Always base your answers on the paper context provided below.

═══════════════════════════════════════════
PAPER ANALYSIS CONTEXT
═══════════════════════════════════════════

📄 Title: {title}
🌐 Domain: {domain}
⭐ Quality Score: {score}/100 (Grade: {grade})
🏷️ Top Keywords: {top_keywords}

📝 Abstract:
{abstract}

📋 Overall Summary:
{overall}

🔬 Methodology:
{methodology}

📈 Results:
{results}

🏁 Conclusion:
{conclusion}

💡 Key Contributions:
{contributions_text}

🚩 Research Gaps:
{gaps_text}

🔭 Suggested Research Ideas:
{ideas_text}

═══════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════

- Answer questions based on the paper context above.
- If asked something not covered by the paper, say so honestly.
- Be concise but thorough. Use bullet points where helpful.
- When suggesting research ideas, be specific and actionable.
- Always maintain a helpful, academic tone.
- Do NOT make up information not present in the paper context.
"""


# ──────────────────────────────────────────────
# Rule-based fallback (no API key)
# ──────────────────────────────────────────────

def _rule_based_response(question: str, paper_context: dict) -> str:
    """Simple keyword-based fallback when no LLM is available."""
    q = question.lower()
    summaries = paper_context.get("summaries", {})
    sections = paper_context.get("sections", {})

    if any(w in q for w in ["summary", "about", "overview", "describe", "what is"]):
        text = summaries.get("overall") or sections.get("abstract", "")
        return f"**Summary:**\n\n{text[:600]}" if text else "No summary available."

    if any(w in q for w in ["method", "approach", "technique", "how"]):
        text = summaries.get("methodology", "") or sections.get("methodology", "")
        return f"**Methodology:**\n\n{text[:600]}" if text else "No methodology details found."

    if any(w in q for w in ["result", "finding", "performance", "accuracy", "score"]):
        text = summaries.get("results", "") or sections.get("results", "")
        return f"**Results:**\n\n{text[:600]}" if text else "No results section found."

    if any(w in q for w in ["conclusion", "conclude", "final"]):
        text = summaries.get("conclusion", "") or sections.get("conclusion", "")
        return f"**Conclusion:**\n\n{text[:600]}" if text else "No conclusion found."

    if any(w in q for w in ["gap", "limitation", "weakness", "problem", "challenge"]):
        gaps = paper_context.get("gaps", {}).get("identified_gaps", [])
        if gaps:
            return "**Research Gaps:**\n\n" + "\n".join(f"• {g}" for g in gaps[:5])
        return "No explicit research gaps detected."

    if any(w in q for w in ["contribution", "novel", "propose", "introduce"]):
        contribs = paper_context.get("contributions", [])
        if contribs:
            return "**Key Contributions:**\n\n" + "\n".join(f"• {c}" for c in contribs[:5])
        return "No explicit contributions detected."

    if any(w in q for w in ["keyword", "topic", "theme"]):
        kws = paper_context.get("keywords", {}).get("top_keywords", [])
        if kws:
            return "**Top Keywords:** " + ", ".join(k["keyword"] for k in kws[:12])
        return "No keywords extracted."

    if any(w in q for w in ["idea", "future", "extend", "project", "suggestion"]):
        ideas = paper_context.get("ideas", {})
        all_ideas = (
            ideas.get("research_extensions", []) +
            ideas.get("implementation_projects", [])
        )
        if all_ideas:
            return "**Research Ideas:**\n\n" + "\n".join(f"• {i}" for i in all_ideas[:6])
        return "No ideas generated."

    if any(w in q for w in ["domain", "field", "area"]):
        domain = paper_context.get("domain", {}).get("primary_domain", "N/A")
        return f"This paper belongs to the **{domain}** research domain."

    if any(w in q for w in ["quality", "score", "grade", "rating"]):
        q_data = paper_context.get("quality", {})
        score = q_data.get("composite_score", "N/A")
        grade = q_data.get("grade", "N/A")
        return f"**Quality Score:** {score}/100 — Grade: **{grade}**"

    if any(w in q for w in ["reference", "citation", "cite"]):
        cit = paper_context.get("citations", {})
        return (
            f"**Citations:** {cit.get('reference_count', 0)} references, "
            f"{cit.get('inline_citations', {}).get('bracket_citation_count', 0)} inline citations."
        )

    return (
        "I can answer questions about:\n\n"
        "• **Summary / Overview**\n"
        "• **Methodology**\n"
        "• **Results & Findings**\n"
        "• **Research Gaps**\n"
        "• **Key Contributions**\n"
        "• **Research Ideas & Extensions**\n"
        "• **Keywords & Topics**\n"
        "• **Quality Score**\n"
        "• **Citations**\n\n"
        "Please rephrase your question or ask about one of these topics!"
    )


# ──────────────────────────────────────────────
# Main chat function
# ──────────────────────────────────────────────

def chat_with_paper(
    question: str,
    paper_context: dict,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict:
    """
    Send a question to the LLM (or fallback) about the paper.

    Args:
        question: The user's question.
        paper_context: The full analysis result from analyze_paper_bytes().
        chat_history: List of {"role": "user"/"assistant", "content": "..."} dicts.

    Returns:
        {"answer": str, "mode": "gemini" | "fallback"}
    """
    if chat_history is None:
        chat_history = []

    model = _get_gemini_client()

    if model is not None:
        try:
            system_prompt = build_system_prompt(paper_context)

            # Build conversation history for multi-turn context
            history_text = ""
            history_items = list(chat_history or [])[-6:]  # last 6 turns for context
            for msg in history_items:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"\n{role}: {msg['content']}"

            full_prompt = (
                system_prompt
                + "\n\n═══ CONVERSATION HISTORY ═══"
                + (history_text if history_text else "\n(No prior conversation)")
                + f"\n\nUser: {question}\nAssistant:"
            )

            response = model.generate_content(full_prompt)
            answer = response.text.strip()
            return {"answer": answer, "mode": "gemini"}

        except Exception as e:
            # Fall through to rule-based
            fallback_answer = _rule_based_response(question, paper_context)
            return {
                "answer": fallback_answer,
                "mode": "fallback",
                "note": f"Gemini error: {str(e)}"
            }
    else:
        answer = _rule_based_response(question, paper_context)
        return {"answer": answer, "mode": "fallback"}
