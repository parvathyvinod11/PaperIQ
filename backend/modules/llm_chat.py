"""
LLM Chat Module – PaperIQ  (RAG-powered)
=========================================
Architecture
------------
  User Question
       │
       ▼
  RAGEngine.retrieve()          ← embed question + cosine search over paper chunks
       │
       ▼
  Top-K relevant passages       ← the "retrieved" part of RAG
       │
       ▼
  build_rag_prompt()            ← inject passages + metadata into LLM context
       │
       ▼
  Gemini 2.0 Flash generate()   ← grounded, citation-aware answer
       │
       ▼
  {answer, mode, retrieved_chunks, total_chunks}

Falls back to rule-based responder when no API key is configured.
"""

import os
import re
from typing import List, Dict, Optional

from modules.rag_engine import RAGEngine

# Singleton retrieval engine (shared model weights with SimilarityEngine)
_rag = RAGEngine()


# ──────────────────────────────────────────────────────────
# Gemini client (lazy-loaded)
# ──────────────────────────────────────────────────────────

def _get_gemini_client():
    """Return a configured Gemini GenerativeModel, or None if unavailable."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai          # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# RAG-augmented prompt builder
# ──────────────────────────────────────────────────────────

def build_rag_prompt(
    question: str,
    paper_context: dict,
    retrieved_chunks: List[Dict],
    chat_history: List[Dict],
) -> str:
    """
    Construct the full prompt sent to Gemini.

    Structure
    ---------
    1. System role  – who PaperIQ is and what it must do
    2. Paper metadata – title, domain, quality score, keywords (structured summary)
    3. RETRIEVED CONTEXT – top-K passages returned by the RAG engine
       (this is the core RAG augmentation step)
    4. Conversation history – last 6 turns for multi-turn coherence
    5. Current question
    """
    title   = paper_context.get("title", "Unknown Paper")
    domain  = paper_context.get("domain", {}).get("primary_domain", "N/A")
    quality = paper_context.get("quality", {})
    score   = quality.get("composite_score", "N/A")
    grade   = quality.get("grade", "N/A")
    kws     = paper_context.get("keywords", {}).get("top_keywords", [])
    top_kw  = ", ".join(k["keyword"] for k in kws[:10]) if kws else "N/A"
    contribs = paper_context.get("contributions", [])
    gaps     = paper_context.get("gaps", {}).get("identified_gaps", [])
    sections = paper_context.get("sections", {})
    abstract = sections.get("abstract", "")[:600]

    contributions_text = "\n".join(f"  • {c}" for c in contribs[:5]) or "  N/A"
    gaps_text          = "\n".join(f"  • {g}" for g in gaps[:5])     or "  N/A"

    # ── Retrieved passages block ──
    if retrieved_chunks:
        rag_block_lines = []
        for i, item in enumerate(retrieved_chunks, 1):
            score_pct = int(item["score"] * 100)
            rag_block_lines.append(
                f"[Passage {i} | relevance {score_pct}%]\n{item['chunk']}"
            )
        rag_block = "\n\n".join(rag_block_lines)
    else:
        rag_block = "(No highly relevant passages found for this query.)"

    # ── Conversation history block ──
    history_text = ""
    for msg in list(chat_history or [])[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"\n{role}: {msg['content']}"

    return f"""You are PaperIQ Assistant, an expert AI research assistant.
You answer questions about a specific research paper using ONLY information from the paper.

═══════════════════════════════════════════
PAPER METADATA
═══════════════════════════════════════════
📄 Title  : {title}
🌐 Domain : {domain}
⭐ Quality : {score}/100  (Grade: {grade})
🏷️ Keywords: {top_kw}

📝 Abstract:
{abstract}

💡 Key Contributions:
{contributions_text}

🚩 Research Gaps:
{gaps_text}

═══════════════════════════════════════════
RETRIEVED CONTEXT  (RAG – semantic search results)
═══════════════════════════════════════════
The following passages were retrieved from the paper because they are most
semantically relevant to the user's question. Ground your answer in these
passages. If they do not contain enough information, say so.

{rag_block}

═══════════════════════════════════════════
CONVERSATION HISTORY
═══════════════════════════════════════════{history_text if history_text else chr(10) + "(No prior conversation)"}

═══════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════
- Base your answer primarily on the RETRIEVED CONTEXT passages above.
- Use the Paper Metadata for high-level facts (domain, score, keywords, gaps).
- Be concise but thorough; use bullet points where helpful.
- If the retrieved passages don't answer the question, say: "The paper doesn't
  appear to cover this directly." Do NOT hallucinate.
- Maintain an academic, helpful tone.

User: {question}
Assistant:"""


# ──────────────────────────────────────────────────────────
# Rule-based fallback (no API key)
# ──────────────────────────────────────────────────────────

def _rule_based_response(question: str, paper_context: dict) -> str:
    """Simple keyword-based fallback when no LLM is available."""
    q        = question.lower()
    summaries = paper_context.get("summaries", {})
    sections  = paper_context.get("sections", {})

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


# ──────────────────────────────────────────────────────────
# Main chat function  (RAG pipeline entry point)
# ──────────────────────────────────────────────────────────

def chat_with_paper(
    question: str,
    paper_context: dict,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict:
    """
    Full RAG pipeline:
      1. Retrieve  – semantic search over paper chunks for the question
      2. Augment   – inject retrieved passages into the prompt
      3. Generate  – Gemini 2.0 Flash produces a grounded answer

    Args:
        question:      The user's natural-language question.
        paper_context: Full analysis result from analyze_paper_bytes().
        chat_history:  List of {"role": "user"/"assistant", "content": "…"}.

    Returns:
        {
          "answer":           str,
          "mode":             "rag-gemini" | "fallback",
          "retrieved_chunks": int,   # how many passages were retrieved
          "total_chunks":     int,   # total indexed passages
        }
    """
    if chat_history is None:
        chat_history = []

    # ── Step 1: RETRIEVE relevant passages ────────────────
    retrieved_passages, total_chunks = _rag.retrieve_for_query(
        query=question,
        paper_context=paper_context,
        top_k=5,
    )

    # ── Step 2: GENERATE with Gemini (augmented prompt) ───
    model = _get_gemini_client()

    if model is not None:
        try:
            prompt = build_rag_prompt(
                question=question,
                paper_context=paper_context,
                retrieved_chunks=retrieved_passages,
                chat_history=chat_history,
            )

            response = model.generate_content(prompt)
            answer   = response.text.strip()

            return {
                "answer":           answer,
                "mode":             "rag-gemini",
                "retrieved_chunks": len(retrieved_passages),
                "total_chunks":     total_chunks,
            }

        except Exception as e:
            # Gemini failed → rule-based fallback
            return {
                "answer":           _rule_based_response(question, paper_context),
                "mode":             "fallback",
                "retrieved_chunks": len(retrieved_passages),
                "total_chunks":     total_chunks,
                "note":             f"Gemini error: {e}",
            }
    else:
        return {
            "answer":           _rule_based_response(question, paper_context),
            "mode":             "fallback",
            "retrieved_chunks": len(retrieved_passages),
            "total_chunks":     total_chunks,
        }
