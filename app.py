"""
PaperIQ — Phase 2 Frontend (Streamlit)
Run: streamlit run app.py
Requires: backend running at http://localhost:8000  (uvicorn backend:app --reload)
"""

import streamlit as st
import requests
import pandas as pd
import io, math
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title="PaperIQ Workspace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS  — refined dark-scientific aesthetic
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Serif+Display&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d0f14; color: #e2e8f0; }
section[data-testid="stSidebar"] {
    background: #080a0f;
    border-right: 1px solid #1e2433;
}

/* ── Animated gradient title ── */
.piq-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #e879f9);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradShift 4s ease infinite;
    letter-spacing: -1px;
}
@keyframes gradShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.piq-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #475569;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: -8px;
    margin-bottom: 24px;
}

/* ── Cards ── */
.card {
    background: #131720;
    border: 1px solid #1e2738;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.4);
}
.card-title {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Domain badge ── */
.domain-badge {
    display: inline-block;
    background: linear-gradient(135deg,#1e3a5f,#1e3356);
    border: 1px solid #2d5a8e;
    color: #7dd3fc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 6px 14px;
    border-radius: 6px;
    margin-right: 8px;
    margin-bottom: 8px;
}
.domain-primary {
    background: linear-gradient(135deg,#1a1050,#2d1a6e);
    border-color: #6d28d9;
    color: #c4b5fd;
    font-size: 0.9rem;
    font-weight: 600;
}

/* ── Keyword pills ── */
.kw-pill {
    display: inline-block;
    background: #1e2738;
    border: 1px solid #2a3650;
    color: #94a3b8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 4px;
    margin: 3px;
}

/* ── Score ring label ── */
.score-number {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    background: linear-gradient(135deg,#38bdf8,#818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.score-label { font-size: 0.7rem; color: #64748b; letter-spacing: 2px; text-transform: uppercase; }

/* ── Metric tile ── */
.metric-tile {
    background: #131720;
    border: 1px solid #1e2738;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; color: #e2e8f0; }
.metric-key { font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

/* ── Alert sentence ── */
.long-sent {
    background: #1a1200;
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #d97706;
    margin-bottom: 8px;
    line-height: 1.5;
}

/* ── Sidebar label ── */
.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 2px;
}

/* ── Plotly override ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg,#1d4ed8,#6d28d9);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    padding: 10px 20px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

div[data-testid="stTab"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ss(key, default=None):
    return st.session_state.get(key, default)

def api(method: str, endpoint: str, **kwargs):
    """Authenticated API call."""
    token = ss("token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = getattr(requests, method)(
            f"{BACKEND}{endpoint}", headers=headers, timeout=120, **kwargs
        )
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "API error")
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to backend. Make sure it's running: `uvicorn backend:app --reload`"
    except Exception as e:
        return None, str(e)

# ──────────────────────────────────────────────────────────────────────────────
# AUTH SCREEN
# ──────────────────────────────────────────────────────────────────────────────
if not ss("token"):
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<p class="piq-title">PaperIQ</p>', unsafe_allow_html=True)
        st.markdown('<p class="piq-sub">Research Intelligence Platform</p>', unsafe_allow_html=True)
       # st.markdown('<div class="card">', unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="researcher@university.edu")
        role  = st.selectbox("Role", ["Student", "Professor", "Researcher", "Professional"])
        if st.button("Enter Workspace →"):
            data, err = api("post", "/auth/login", json={"email": email, "role": role})
            if err:
                st.error(err)
            else:
                st.session_state["token"] = data["token"]
                st.session_state["user"]  = data["user"]
                st.rerun()
      #  st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="piq-title" style="font-size:1.8rem">PaperIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="piq-sub" style="font-size:0.6rem">Research Intelligence</p>', unsafe_allow_html=True)
    st.divider()

    user = ss("user", {})
    st.markdown(f'<p class="sidebar-label">Logged in as</p>', unsafe_allow_html=True)
    st.markdown(f"**{user.get('role','—')}** · {user.get('email','—')}")
    st.divider()

    st.markdown('<p class="sidebar-label">Input Mode</p>', unsafe_allow_html=True)
    mode = st.radio("", ["PDF Upload", "Raw Text"], label_visibility="collapsed")

    st.divider()
    if st.button("Logout"):
        api("post", "/auth/logout")
        for k in ["token", "user", "result", "last_fn"]:
            st.session_state.pop(k, None)
        st.rerun()

    # Backend status
    health, _ = api("get", "/health")
    if health:
       # st.markdown('<p class="sidebar-label" style="margin-top:16px">Backend Modules</p>', unsafe_allow_html=True)
        for mod, ok in [("PyMuPDF", health.get("fitz")),
                        ("pdfplumber", health.get("pdfplumber")),
                        ("TextBlob", health.get("textblob"))]:
            icon = "🟢" if ok else "🔴"
          #  st.markdown(f"{icon} `{mod}`")
    else:
        st.warning("Backend offline")

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="piq-title">PaperIQ Research Workspace</p>', unsafe_allow_html=True)
# st.markdown('<p class="piq-sub">Structural · Semantic · Domain Intelligence</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# UPLOAD / TEXT INPUT
# ──────────────────────────────────────────────────────────────────────────────
result = ss("result")

if mode == "PDF Upload":
    uploaded = st.file_uploader("Upload PDF Research Paper", type=["pdf"])
    if uploaded:
        if ss("last_fn") != uploaded.name:
            with st.spinner("🔬 Running deep structural analysis via backend…"):
                data, err = api("post", "/analyze/pdf",
                                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
            if err:
                st.error(err)
            else:
                st.session_state["result"]  = data
                st.session_state["last_fn"] = uploaded.name
                result = data
else:
    raw_text = st.text_area("Paste research text (min 100 chars)", height=200)
    if st.button("Analyse Text →") and raw_text:
        with st.spinner("Analysing…"):
            data, err = api("post", "/analyze/text", json={"text": raw_text})
        if err:
            st.error(err)
        else:
            # Build minimal result structure for text mode
            data["metadata"] = {"title": "Text Input", "doi": "N/A",
                                 "authors": "N/A", "year": "N/A", "journal": "N/A"}
            data["sections"] = {}
            data["tables"]   = []
            data["page_count"] = 0
            st.session_state["result"] = data
            result = data

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
if not result:
    st.info("Upload a PDF or paste text above to begin analysis.")
    st.stop()

meta     = result.get("metadata", {})
nlp      = result.get("nlp", {})
domain   = result.get("domain", {})
sections = result.get("sections", {})
keywords = result.get("keywords", [])
tables   = result.get("tables", [])
scores   = nlp.get("scores", {})
metrics  = nlp.get("metrics", {})
pg_count = result.get("page_count", 0)

# ── Metadata + Domain card ────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
col_meta, col_domain = st.columns([3, 2])

with col_meta:
    st.markdown(f"#### {meta.get('title','Unknown Title')}")
    st.markdown(f"**Authors:** {meta.get('authors','—')} &nbsp;|&nbsp; **Year:** {meta.get('year','—')}")
    st.markdown(f"**DOI:** `{meta.get('doi','—')}` &nbsp;|&nbsp; **Journal:** _{meta.get('journal','—')}_")
    kw_html = "".join(f'<span class="kw-pill">{k}</span>' for k in keywords)
    st.markdown(f"<div style='margin-top:10px'>{kw_html}</div>", unsafe_allow_html=True)

with col_domain:
    st.markdown('<p class="card-title">Domain Classification</p>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="domain-badge domain-primary">🎯 {domain.get("primary","—")}</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<span class="domain-badge">⬡ {domain.get("secondary","—")}</span>',
        unsafe_allow_html=True
    )
    conf = domain.get("confidence", 0)
    st.progress(int(conf), text=f"Confidence: {conf}%")

st.markdown('</div>', unsafe_allow_html=True)

# ── Top Metrics row ───────────────────────────────────────────────────────────
composite = scores.get("Composite", 0)
m1, m2, m3, m4, m5, m6 = st.columns(6)
tile_data = [
    (f"{composite}/100", "Composite Score"),
    (pg_count or "—",    "Pages"),
    (len(sections),      "Sections"),
    (len(tables),        "Tables"),
    (metrics.get("Total Words", "—"),     "Words"),
    (metrics.get("Flesch-Kincaid Grade", "—"), "FK Grade"),
]
for col, (val, label) in zip([m1,m2,m3,m4,m5,m6], tile_data):
    col.markdown(
        f'<div class="metric-tile"><div class="metric-val">{val}</div>'
        f'<div class="metric-key">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Download button ───────────────────────────────────────────────────────────
def make_pdf_report(scores, metrics, meta, domain):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "PaperIQ Quality Audit Report", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Title: {meta.get('title','N/A')[:80]}", ln=1)
    pdf.cell(0, 8, f"Authors: {meta.get('authors','N/A')[:80]}", ln=1)
    pdf.cell(0, 8, f"Domain: {domain.get('primary','N/A')} (confidence {domain.get('confidence',0)}%)", ln=1)
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Quality Scores", ln=1)
    pdf.set_font("Arial", size=10)
    for k, v in scores.items():
        pdf.cell(0, 7, f"  {k}: {v}/100", ln=1)
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Detailed Metrics", ln=1)
    pdf.set_font("Arial", size=10)
    for k, v in metrics.items():
        pdf.cell(0, 7, f"  {k}: {v}", ln=1)
    return bytes(pdf.output(dest='S'))

report_bytes = make_pdf_report(scores, metrics, meta, domain)
st.download_button(
    "📩 Download Quality Audit PDF",
    report_bytes,
    "PaperIQ_Audit.pdf",
    "application/pdf",
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Analytics",
    "🌐 Domain Map",
    "📑 Sections",
    "🧮 Tables",
    "🔍 Search",
])

# ── TAB 1 : Analytics ────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns(2)

    # Radar chart
    with col_l:
        st.markdown("#### Writing Profile Radar")
        cats = [k for k in scores if k != "Composite"]
        vals = [scores[k] for k in cats]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill='toself',
            line_color='#818cf8',
            fillcolor='rgba(129,140,248,0.15)',
        ))
        fig_radar.update_layout(
            paper_bgcolor='#131720', plot_bgcolor='#131720',
            font_color='#94a3b8',
            polar=dict(
                bgcolor='#0d0f14',
                radialaxis=dict(visible=True, range=[0,100],
                                gridcolor='#1e2738', tickcolor='#475569'),
                angularaxis=dict(gridcolor='#1e2738'),
            ),
            height=380, margin=dict(l=20,r=20,t=20,b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Horizontal bar scores
    with col_r:
        st.markdown("#### Score Breakdown")
        fig_bar = go.Figure(go.Bar(
            x=vals, y=cats, orientation='h',
            marker=dict(
                color=vals,
                colorscale=[[0,'#ef4444'],[0.5,'#f59e0b'],[1,'#22d3ee']],
                showscale=False,
            ),
            text=[f"{v}" for v in vals],
            textposition='outside',
            textfont_color='#94a3b8',
        ))
        fig_bar.update_layout(
            paper_bgcolor='#131720', plot_bgcolor='#131720',
            font_color='#94a3b8',
            xaxis=dict(range=[0,110], gridcolor='#1e2738'),
            yaxis=dict(gridcolor='#1e2738'),
            height=380, margin=dict(l=20,r=40,t=20,b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed metrics table
    st.markdown("#### Detailed NLP Metrics")
    met_col1, met_col2 = st.columns(2)
    items = list(metrics.items())
    half  = math.ceil(len(items) / 2)
    for col, chunk in zip([met_col1, met_col2], [items[:half], items[half:]]):
        with col:
            for k, v in chunk:
                c1, c2 = st.columns([2,1])
                c1.markdown(f'<span style="color:#64748b;font-size:0.8rem">{k}</span>',
                             unsafe_allow_html=True)
                c2.markdown(f'<span style="font-family:IBM Plex Mono,monospace;color:#e2e8f0">{v}</span>',
                             unsafe_allow_html=True)

    # Long sentence alerts
    long_sents = nlp.get("long_sentences", [])
    if long_sents:
        st.markdown("#### ⚠️ Sentence Complexity Alerts")
        for s in long_sents:
            st.markdown(f'<div class="long-sent">{s[:200]}…</div>', unsafe_allow_html=True)

    # Sentiment
    sentiment = nlp.get("sentiment", 0)
    sent_col1, sent_col2 = st.columns([1, 3])
    with sent_col1:
        st.markdown("#### Sentiment Polarity")
        colour = "#22d3ee" if sentiment > 0.05 else ("#ef4444" if sentiment < -0.05 else "#f59e0b")
        st.markdown(
            f'<div class="metric-tile"><div class="metric-val" style="color:{colour}">{sentiment:+.3f}</div>'
            f'<div class="metric-key">{"Positive" if sentiment>0.05 else "Negative" if sentiment<-0.05 else "Neutral"}</div></div>',
            unsafe_allow_html=True,
        )

# ── TAB 2 : Domain Map ───────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Domain Classification Breakdown")
    all_scores = domain.get("all_scores", {})
    if all_scores:
        dom_df = pd.DataFrame(list(all_scores.items()), columns=["Domain","Confidence (%)"])
        dom_df = dom_df.sort_values("Confidence (%)", ascending=False)

        fig_dom = px.bar(
            dom_df, x="Confidence (%)", y="Domain", orientation='h',
            color="Confidence (%)",
            color_continuous_scale=[[0,'#1e293b'],[0.4,'#1d4ed8'],[1,'#818cf8']],
        )
        fig_dom.update_layout(
            paper_bgcolor='#131720', plot_bgcolor='#131720',
            font_color='#94a3b8',
            xaxis=dict(gridcolor='#1e2738'),
            yaxis=dict(gridcolor='#1e2738', categoryorder='total ascending'),
            coloraxis_showscale=False,
            height=420, margin=dict(l=20,r=30,t=20,b=20),
        )
        st.plotly_chart(fig_dom, use_container_width=True)

        st.markdown("#### Interpretation")
        primary   = domain.get("primary", "—")
        secondary = domain.get("secondary", "—")
        conf      = domain.get("confidence", 0)
        st.markdown(
            f"This paper is classified as **{primary}** with **{conf}%** confidence. "
            f"The secondary domain signal is **{secondary}**, suggesting "
            f"interdisciplinary elements. Domain classification is based on "
            f"lexical frequency analysis of domain-specific terminology."
        )

# ── TAB 3 : Sections ─────────────────────────────────────────────────────────
with tab3:
    if sections:
        for title, body in sections.items():
            with st.expander(f"📍 {title}", expanded=title.upper() in ["ABSTRACT","1. INTRODUCTION"]):
                st.write(body)
    else:
        st.info("Section extraction not available for text-mode input.")

# ── TAB 4 : Tables ───────────────────────────────────────────────────────────
with tab4:
    if tables:
        for idx, tbl in enumerate(tables):
            st.markdown(f"**Table {idx + 1}**")
            if tbl and len(tbl) > 1:
                try:
                    df = pd.DataFrame(tbl[1:], columns=[str(c).replace('\n',' ') for c in tbl[0]])
                    st.dataframe(df.dropna(how='all'), use_container_width=True)
                except Exception:
                    st.write(tbl)
    else:
        st.info("No structured tables detected in this document.")

# ── TAB 5 : Search ───────────────────────────────────────────────────────────
with tab5:
    query = st.text_input("🔍 Search for keywords or concepts across sections…")
    if query and sections:
        matches = [(t, b) for t, b in sections.items() if query.lower() in b.lower()]
        if matches:
            st.success(f"Found **{len(matches)}** section(s) containing '{query}'")
            for t, b in matches:
                start = max(0, b.lower().find(query.lower()) - 100)
                snippet = b[start:start+300]
                highlighted = snippet.replace(
                    query, f"**{query}**"
                ).replace(
                    query.upper(), f"**{query.upper()}**"
                )
                with st.expander(f"📍 {t}"):
                    st.markdown(f"…{highlighted}…")
        else:
            st.warning(f"No matches found for '{query}' in extracted sections.")
    elif query:
        st.info("Section search not available in text-mode analysis.")
