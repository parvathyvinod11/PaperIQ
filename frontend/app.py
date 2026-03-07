"""
PaperIQ – Streamlit Frontend Dashboard
AI-powered Research Paper Analysis Platform
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="PaperIQ – AI Research Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0e1117; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }
    .hero-banner h1 {
        font-size: 2.6rem;
        font-weight: 700;
        color: #fff;
        margin: 0;
    }
    .hero-banner p {
        color: #a8b2d8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .accent { color: #e94560; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e3a, #16213e);
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #e94560; }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #e94560;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 0.3rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ccd6f6;
        border-left: 4px solid #e94560;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }

    /* Tag chips */
    .tag {
        display: inline-block;
        background: #1e3a5f;
        color: #64b5f6;
        border: 1px solid #1565c0;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        margin: 3px;
    }

    /* Quality score bar */
    .score-bar-wrapper {
        background: #1e1e3a;
        border-radius: 8px;
        border: 1px solid #2d2d5e;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }

    /* Idea cards */
    .idea-card {
        background: #12192d;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e94560;
    }
    .idea-card p { color: #ccd6f6; margin: 0; font-size: 0.9rem; }

    /* Gap cards */
    .gap-card {
        background: #1a0a0a;
        border: 1px solid #3d1515;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f44336;
    }
    .gap-card p { color: #ffcdd2; margin: 0; font-size: 0.88rem; }

    /* Stacked sidebar */
    .css-1d391kg { background: #0d1117; }

    /* Upload area enhancement */
    .stFileUploader > div > div {
        border: 2px dashed #e94560 !important;
        background: #0d1117 !important;
        border-radius: 12px !important;
    }

    /* Summary box */
    .summary-box {
        background: #12192d;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        color: #a8b2d8;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    stTabs [data-baseweb="tab"] {
        color: #8892b0;
    }
    stTabs [aria-selected="true"] {
        color: #e94560 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def check_api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def render_tags(items, max_items=12):
    tags_html = "".join(f'<span class="tag">{item}</span>' for item in items[:max_items])
    st.markdown(tags_html, unsafe_allow_html=True)


def quality_color(score):
    if score >= 80:
        return "#4caf50"
    elif score >= 65:
        return "#2196f3"
    elif score >= 50:
        return "#ff9800"
    else:
        return "#f44336"


def plot_quality_breakdown(breakdown: dict):
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    labels_clean = [l.replace("_", " ").title() for l in labels]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]] if values else [],
        theta=labels_clean + [labels_clean[0]] if labels_clean else [],
        fill='toself',
        fillcolor='rgba(233, 69, 96, 0.4)',
        line=dict(color='#e94560'),
        marker=dict(color='#e94560')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='#8892b0',
                gridcolor='#2d2d5e',
                linecolor='#2d2d5e'
            ),
            angularaxis=dict(
                color='#ccd6f6',
                gridcolor='#2d2d5e',
                linecolor='#2d2d5e'
            ),
            bgcolor='#12192d'
        ),
        paper_bgcolor="#12192d",
        font_color="#ccd6f6",
        margin=dict(l=40, r=40, t=30, b=30),
        height=350,
    )
    return fig


def plot_keyword_bar(kw_freq: list):
    if not kw_freq:
        return None
    df = pd.DataFrame(kw_freq[:15])
    fig = px.bar(
        df, x="frequency", y="keyword", orientation="h",
        color="frequency",
        color_continuous_scale=["#1565c0", "#e94560"],
        labels={"frequency": "Frequency", "keyword": ""},
    )
    fig.update_layout(
        plot_bgcolor="#12192d",
        paper_bgcolor="#12192d",
        font_color="#ccd6f6",
        coloraxis_showscale=False,
        margin=dict(l=10, r=20, t=10, b=10),
        height=350,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=False),
    )
    return fig


def plot_similarity_heatmap(matrix: list, labels: list):
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="RdBu",
        reversescale=True,
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Paper 1: %{y}<br>Paper 2: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="#12192d",
        paper_bgcolor="#12192d",
        font_color="#ccd6f6",
        margin=dict(l=10, r=20, t=10, b=10),
        height=400,
    )
    return fig


def plot_citation_years(year_dist: dict):
    if not year_dist:
        return None
    years = list(year_dist.keys())
    counts = list(year_dist.values())
    fig = px.bar(
        x=years, y=counts,
        labels={"x": "Year", "y": "References"},
        color=counts,
        color_continuous_scale=["#1565c0", "#e94560"],
    )
    fig.update_layout(
        plot_bgcolor="#12192d",
        paper_bgcolor="#12192d",
        font_color="#ccd6f6",
        coloraxis_showscale=False,
        margin=dict(l=10, r=20, t=10, b=10),
        height=280,
    )
    return fig


def plot_domain_confidence(domains: list):
    if not domains:
        return None
    df = pd.DataFrame(domains)
    fig = go.Figure(go.Bar(
        x=df["confidence"] * 100,
        y=df["domain"],
        orientation="h",
        marker_color="#e94560",
        text=[f"{v*100:.1f}%" for v in df["confidence"]],
        textposition="outside",
    ))
    fig.update_layout(
        plot_bgcolor="#12192d",
        paper_bgcolor="#12192d",
        font_color="#ccd6f6",
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(range=[0, 110], showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=260,
    )
    return fig


def plot_topics(topics: list):
    if not topics:
        return None
    rows = []
    for t in topics:
        for kw in t["keywords"][:5]:
            rows.append({"topic": t["label"], "keyword": kw, "weight": t["weight"]})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="keyword", color="topic", barmode="group",
        labels={"keyword": "Keyword", "topic": "Topic"},
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(
        plot_bgcolor="#12192d",
        paper_bgcolor="#12192d",
        font_color="#ccd6f6",
        legend=dict(font=dict(color="#ccd6f6")),
        margin=dict(l=10, r=20, t=10, b=10),
        height=320,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=False),
    )
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size:2.5rem">🔬</span>
        <h2 style="color:#e94560; margin:0.3rem 0;">PaperIQ</h2>
        <p style="color:#8892b0; font-size:0.85rem;">AI Research Insight Analyzer</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    mode = st.radio(
        "**Analysis Mode**",
        ["📄 Single Paper Analysis", "📚 Multi-Paper Comparison", "🔍 Semantic Search", "🕒 History"],
        label_visibility="visible",
    )
    st.divider()

    if st.session_state["token"]:
        st.success(f"👤 Logged in as {st.session_state['username']} ({st.session_state['role']})")
        if st.button("Logout"):
            st.session_state["token"] = None
            st.session_state["username"] = None
            st.session_state["role"] = None
            st.rerun()

    api_ok = check_api()
    if api_ok:
        st.success("✅ Backend API: Online", icon=None)
    else:
        st.error("❌ Backend API: Offline\n\nRun: `cd backend && python main.py`")

    st.markdown("""
    <div style="margin-top: 2rem; color: #4a5568; font-size: 0.78rem;">
        <b>PaperIQ v1.0</b><br>
        AI-powered research intelligence<br>
        FastAPI + Streamlit + NLP
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <h1>Paper<span class="accent">IQ</span> 🔬</h1>
    <p>Transform academic papers into structured knowledge, insights, and research opportunities using AI.</p>
</div>
""", unsafe_allow_html=True)

def render_auth():
    tab1, tab2 = st.tabs(["Login", "Signup"])
    
    with tab1:
        st.subheader("Login to your account")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                try:
                    resp = requests.post(f"{API_BASE}/api/auth/login", json={"email": email, "password": password})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["token"] = data["access_token"]
                        st.session_state["username"] = data["username"]
                        st.session_state["role"] = data["role"]
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Login failed"))
                except Exception:
                    st.error("Could not connect to backend API")
                    
    with tab2:
        st.subheader("Create a new account")
        with st.form("signup_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            role = st.selectbox("I am a:", ["Student", "Professional", "Professor", "Researcher", "Other"])
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/auth/signup", 
                        json={"username": new_username, "email": new_email, "password": new_password, "role": role}
                    )
                    if resp.status_code == 200:
                        st.success("Account created successfully! You can now login.")
                    else:
                        st.error(resp.json().get("detail", "Signup failed"))
                except Exception:
                    st.error("Could not connect to backend API")


# ─────────────────────────────────────────────
# Mode: Single Paper Analysis
# ─────────────────────────────────────────────

def render_single_paper():
    st.markdown('<div class="section-title">📤 Upload Research Paper</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a PDF research paper",
        type=["pdf"],
        label_visibility="collapsed",
    )
    custom_title = st.text_input("📝 Paper Title (optional)", placeholder="Leave blank to auto-detect")

    if uploaded and st.button("🚀 Analyze Paper", type="primary", use_container_width=True):
        if not api_ok:
            st.error("Backend API is not running. Please start it first.")
            return

        with st.spinner("🤖 Analyzing paper... This may take 30–60 seconds."):
            try:
                headers = {}
                if st.session_state["token"]:
                    headers["Authorization"] = f"Bearer {st.session_state['token']}"
                    
                resp = requests.post(
                    f"{API_BASE}/api/analyze",
                    files={"file": (uploaded.name, uploaded.read(), "application/pdf")},
                    data={"title": custom_title} if custom_title else {},
                    headers=headers,
                    timeout=120,
                )
                if resp.status_code != 200:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
                    return
                data = resp.json()
                st.session_state["analysis"] = data
                st.success("✅ Analysis complete!")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Make sure it's running.")
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

    if "analysis" in st.session_state:
        render_analysis(st.session_state["analysis"])


def render_analysis(data: dict):
    title = data.get("title", "Paper")
    metadata = data.get("metadata", {})
    pdf_stats = data.get("pdf_stats", {})
    domain = data.get("domain", {})
    quality = data.get("quality", {})
    keywords = data.get("keywords", {})
    summaries = data.get("summaries", {})
    sections = data.get("sections", {})
    gaps = data.get("gaps", {})
    citations = data.get("citations", {})
    topics = data.get("topics", [])
    kw_freq = data.get("keyword_frequency", [])
    ideas = data.get("ideas", {})
    contributions = data.get("contributions", [])
    preprocessed = data.get("preprocessed", {})

    # Title bar
    st.markdown(f"""
    <div style="background:#12192d; border:1px solid #1e3a5f; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1.5rem;">
        <h2 style="color:#e0e0e0; margin:0;">{title}</h2>
        <p style="color:#8892b0; margin:0.4rem 0 0;">
            {metadata.get("author","") or ""}  
            {"·" if metadata.get("author") else ""}
            {pdf_stats.get("page_count",0)} pages &nbsp;|&nbsp;
            {pdf_stats.get("word_count",0):,} words &nbsp;|&nbsp;
            {preprocessed.get("sentence_count",0)} sentences
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    qs = quality.get("composite_score", 0)
    grade = quality.get("grade", "—")
    ref_count = citations.get("reference_count", 0)
    kw_count = len(keywords.get("top_keywords", []))
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:{quality_color(qs)}">{qs}</div>
            <div class="label">Quality Score /100</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{grade}</div>
            <div class="label">Quality Grade</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{ref_count}</div>
            <div class="label">References</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{kw_count}</div>
            <div class="label">Keywords</div></div>""", unsafe_allow_html=True)
    with col5:
        dom_short = domain.get("primary_domain", "N/A")
        dom_short = dom_short[:16] + "…" if len(dom_short) > 16 else dom_short
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="font-size:1.1rem">{dom_short}</div>
            <div class="label">Research Domain</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tabs = st.tabs([
        "📋 Summary",
        "🏷️ Keywords",
        "📊 Analytics",
        "🔍 Gaps & Ideas",
        "📑 Sections",
        "📚 Citations",
    ])

    # ── Tab 1: Summary ──
    with tabs[0]:
        st.markdown('<div class="section-title">🎯 Key Contributions</div>', unsafe_allow_html=True)
        if contributions:
            for c in contributions:
                st.markdown(f'<div class="idea-card"><p>💡 {c}</p></div>', unsafe_allow_html=True)
        else:
            st.info("No explicit contribution statements detected.")

        st.markdown('<div class="section-title">📝 Overall Summary</div>', unsafe_allow_html=True)
        overall = summaries.get("overall", "Not available.")
        st.markdown(f'<div class="summary-box">{overall}</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">🔬 Methodology Summary</div>', unsafe_allow_html=True)
            met_sum = summaries.get("methodology", "Not available.")
            st.markdown(f'<div class="summary-box">{met_sum}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="section-title">📈 Results Summary</div>', unsafe_allow_html=True)
            res_sum = summaries.get("results", "Not available.")
            st.markdown(f'<div class="summary-box">{res_sum}</div>', unsafe_allow_html=True)

    # ── Tab 2: Keywords ──
    with tabs[1]:
        st.markdown('<div class="section-title">🏷️ Top Keywords</div>', unsafe_allow_html=True)
        top_kws = keywords.get("top_keywords", [])
        if top_kws:
            render_tags([k["keyword"] for k in top_kws])

        st.markdown('<div class="section-title">📊 Keyword Frequency</div>', unsafe_allow_html=True)
        fig_kw = plot_keyword_bar(kw_freq)
        if fig_kw:
            st.plotly_chart(fig_kw, use_container_width=True)

        col_e1, col_e2, col_e3 = st.columns(3)
        entities = keywords.get("entities", {})
        with col_e1:
            st.markdown("**🤖 Algorithms / Models**")
            algos = entities.get("algorithms", [])
            if algos:
                render_tags(algos)
            else:
                st.caption("None detected")
        with col_e2:
            st.markdown("**📦 Datasets**")
            dsets = entities.get("datasets", [])
            if dsets:
                render_tags(dsets)
            else:
                st.caption("None detected")
        with col_e3:
            st.markdown("**📏 Metrics**")
            mets = entities.get("metrics", [])
            if mets:
                render_tags(mets)
            else:
                st.caption("None detected")

    # ── Tab 3: Analytics ──
    with tabs[2]:
        col_d, col_q = st.columns(2)
        with col_d:
            st.markdown('<div class="section-title">🌐 Domain Classification</div>', unsafe_allow_html=True)
            dom_fig = plot_domain_confidence(domain.get("top_domains", []))
            if dom_fig:
                st.plotly_chart(dom_fig, use_container_width=True)

        with col_q:
            st.markdown('<div class="section-title">⭐ Quality Breakdown</div>', unsafe_allow_html=True)
            breakdown = quality.get("breakdown", {})
            if breakdown:
                fig_q = plot_quality_breakdown(breakdown)
                st.plotly_chart(fig_q, use_container_width=True)

        st.markdown('<div class="section-title">💬 Topic Modeling (LDA)</div>', unsafe_allow_html=True)
        fig_t = plot_topics(topics)
        if fig_t:
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Not enough content for topic modeling.")

        st.markdown('<div class="section-title">📅 Reference Year Distribution</div>', unsafe_allow_html=True)
        fig_cy = plot_citation_years(citations.get("year_distribution", {}))
        if fig_cy:
            st.plotly_chart(fig_cy, use_container_width=True)

    # ── Tab 4: Gaps & Ideas ──
    with tabs[3]:
        col_g, col_f = st.columns(2)
        with col_g:
            st.markdown('<div class="section-title">🚩 Research Gaps</div>', unsafe_allow_html=True)
            gap_list = gaps.get("identified_gaps", [])
            if gap_list:
                for g in gap_list:
                    st.markdown(f'<div class="gap-card"><p>⚠️ {g}</p></div>', unsafe_allow_html=True)
            else:
                st.info("No explicit research gaps detected.")

        with col_f:
            st.markdown('<div class="section-title">🔭 Future Directions</div>', unsafe_allow_html=True)
            fut_list = gaps.get("future_directions", [])
            if fut_list:
                for f in fut_list:
                    st.markdown(f'<div class="idea-card"><p>→ {f}</p></div>', unsafe_allow_html=True)
            else:
                st.info("No explicit future directions mentioned.")

        st.markdown('<div class="section-title">💡 Research Ideas & Project Suggestions</div>', unsafe_allow_html=True)
        idea_tabs = st.tabs(["Research Extensions", "Implementation Projects", "Dataset Ideas"])
        with idea_tabs[0]:
            for idea in ideas.get("research_extensions", []):
                st.markdown(f'<div class="idea-card"><p>🔬 {idea}</p></div>', unsafe_allow_html=True)
        with idea_tabs[1]:
            for idea in ideas.get("implementation_projects", []):
                st.markdown(f'<div class="idea-card"><p>🛠️ {idea}</p></div>', unsafe_allow_html=True)
        with idea_tabs[2]:
            for idea in ideas.get("dataset_ideas", []):
                st.markdown(f'<div class="idea-card"><p>📦 {idea}</p></div>', unsafe_allow_html=True)

        if ideas.get("gap_based_ideas"):
            st.markdown('<div class="section-title">🎯 Gap-Based Research Directions</div>', unsafe_allow_html=True)
            for idea in ideas["gap_based_ideas"]:
                st.markdown(f'<div class="gap-card"><p>{idea}</p></div>', unsafe_allow_html=True)

    # ── Tab 5: Sections ──
    with tabs[4]:
        section_keys = ["abstract", "introduction", "related_work", "methodology",
                        "experiments", "results", "conclusion"]
        for sk in section_keys:
            content = sections.get(sk, "")
            if content:
                with st.expander(f"📄 {sk.replace('_', ' ').title()} ({len(content.split())} words)"):
                    st.markdown(f'<div class="summary-box">{content[:2000]}{"…" if len(content) > 2000 else ""}</div>',
                                unsafe_allow_html=True)

    # ── Tab 6: Citations ──
    with tabs[5]:
        inline = citations.get("inline_citations", {})
        cy1, cy2, cy3 = st.columns(3)
        with cy1:
            st.metric("Total References", citations.get("reference_count", 0))
        with cy2:
            st.metric("Inline Citations", inline.get("bracket_citation_count", 0))
        with cy3:
            st.metric("Citation Density", f"{inline.get('citation_density', 0):.3f}")

        st.markdown('<div class="section-title">📋 Reference List (first 15)</div>', unsafe_allow_html=True)
        refs = citations.get("references", [])
        for i, ref in enumerate(refs[:15], 1):
            st.markdown(f"**[{i}]** {ref[:200]}")

        top_authors = citations.get("top_authors", [])
        if top_authors:
            st.markdown('<div class="section-title">👤 Top Cited Authors</div>', unsafe_allow_html=True)
            df_auth = pd.DataFrame(top_authors)
            fig_auth = px.bar(df_auth, x="count", y="author", orientation="h",
                              color="count", color_continuous_scale=["#1565c0", "#e94560"])
            fig_auth.update_layout(plot_bgcolor="#12192d", paper_bgcolor="#12192d",
                                   font_color="#ccd6f6", height=280,
                                   coloraxis_showscale=False, margin=dict(l=10, r=20, t=5, b=5))
            st.plotly_chart(fig_auth, use_container_width=True)


# ─────────────────────────────────────────────
# Mode: Multi-Paper Comparison
# ─────────────────────────────────────────────

def render_multi_paper():
    st.markdown('<div class="section-title">📤 Upload Multiple Papers (2–6 PDFs)</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and len(uploaded_files) >= 2:
        st.info(f"📁 {len(uploaded_files)} papers uploaded")
        if st.button("🔀 Compare Papers", type="primary", use_container_width=True):
            if not api_ok:
                st.error("Backend offline.")
                return
            with st.spinner("🤖 Comparing papers... Please wait."):
                try:
                    files_payload = [
                        ("files", (f.name, f.read(), "application/pdf"))
                        for f in uploaded_files
                    ]
                    resp = requests.post(
                        f"{API_BASE}/api/compare",
                        files=files_payload,
                        timeout=300,
                    )
                    if resp.status_code != 200:
                        st.error(f"API Error: {resp.text}")
                        return
                    data = resp.json()
                    st.session_state["comparison"] = data
                    st.success("✅ Comparison complete!")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif uploaded_files:
        st.warning("Please upload at least 2 PDF files.")

    if "comparison" in st.session_state:
        render_comparison(st.session_state["comparison"])


def render_comparison(data: dict):
    comparison = data.get("comparison", {})
    similarity = data.get("similarity", {})
    papers = data.get("papers", [])

    st.markdown(f"**{comparison.get('summary', '')}**")

    # Similarity heatmap
    if similarity.get("matrix") and len(similarity["matrix"]) > 1:
        st.markdown('<div class="section-title">🔥 Semantic Similarity Heatmap</div>', unsafe_allow_html=True)
        fig_heat = plot_similarity_heatmap(similarity["matrix"], similarity["labels"])
        st.plotly_chart(fig_heat, use_container_width=True)

    # Comparison table
    table = comparison.get("comparison_table", [])
    if table:
        st.markdown('<div class="section-title">📊 Comparison Table</div>', unsafe_allow_html=True)
        df = pd.DataFrame(table)
        display_cols = ["title", "domain", "algorithms", "datasets", "metrics",
                        "reference_count", "quality_score"]
        df_display = df[[c for c in display_cols if c in df.columns]]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Individual paper summaries
    st.markdown('<div class="section-title">📄 Individual Paper Summaries</div>', unsafe_allow_html=True)
    for p in papers:
        with st.expander(f"📄 {p.get('title', 'Paper')} — Score: {p.get('quality', {}).get('composite_score', 'N/A')}/100"):
            st.markdown(p.get("summaries", {}).get("overall", "No summary."))


# ─────────────────────────────────────────────
# Mode: Semantic Search
# ─────────────────────────────────────────────

def render_search():
    st.markdown('<div class="section-title">🔍 Semantic Research Search</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Natural Language Query",
        placeholder="e.g., deep learning for medical image segmentation",
    )
    uploaded_files = st.file_uploader(
        "Upload PDFs to search across",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if query and uploaded_files and st.button("🔎 Search", type="primary", use_container_width=True):
        if not api_ok:
            st.error("Backend offline.")
            return
        with st.spinner("Searching..."):
            try:
                files_payload = [
                    ("files", (f.name, f.read(), "application/pdf"))
                    for f in uploaded_files
                ]
                resp = requests.post(
                    f"{API_BASE}/api/search",
                    data={"query": query},
                    files=files_payload,
                    timeout=120,
                )
                if resp.status_code != 200:
                    st.error(f"API Error: {resp.text}")
                    return
                results = resp.json().get("results", [])

                st.markdown(f'<div class="section-title">Results for: "{query}"</div>', unsafe_allow_html=True)
                for i, r in enumerate(results, 1):
                    score_pct = int(r["score"] * 100)
                    color = quality_color(score_pct)
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:left; margin-bottom:0.8rem; display:flex; align-items:center; justify-content:space-between;">
                        <span style="color:#ccd6f6;">#{i} {r['title']}</span>
                        <span style="color:{color}; font-weight:700;">{r['score']:.3f} relevance</span>
                    </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# Route by mode
# ─────────────────────────────────────────────

def render_history():
    st.markdown('<div class="section-title">🕒 Your Analysis History</div>', unsafe_allow_html=True)
    if not st.session_state.get("token"):
        st.warning("Please login to view history.")
        return
        
    try:
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        resp = requests.get(f"{API_BASE}/api/history/", headers=headers, timeout=10)
        
        if resp.status_code == 200:
            history_data = resp.json().get("history", [])
            if not history_data:
                st.info("No past analysis found. Go to 'Single Paper Analysis' to analyze your first paper!")
                return
                
            for i, item in enumerate(history_data):
                dt = str(item.get("timestamp", ""))[:19].replace("T", " ")
                title = item.get("title", "Unknown Paper")
                analysis = item.get("analysis", {})
                score = analysis.get("quality", {}).get("composite_score", "N/A")
                domain = analysis.get("domain", {}).get("primary_domain", "Unknown")
                
                with st.expander(f"📄 {title}  |  {dt}  |  Score: {score}"):
                    st.markdown(f"**Domain:** {domain}")
                    st.markdown(f"**Summary:** {analysis.get('summaries', {}).get('overall', 'N/A')}")
                    if st.button("Load Full Analysis", key=f"load_{i}"):
                        st.session_state["analysis"] = analysis
                        st.success("Loaded! Switch to 'Single Paper Analysis' tab to view it.")
        else:
            st.error("Could not fetch history")
    except Exception as e:
        st.error(f"Error accessing history API: {str(e)}")


# ─────────────────────────────────────────────
# Route by mode
# ─────────────────────────────────────────────

if not st.session_state["token"]:
    render_auth()
else:
    if mode == "📄 Single Paper Analysis":
        render_single_paper()
    elif mode == "📚 Multi-Paper Comparison":
        render_multi_paper()
    elif mode == "🔍 Semantic Search":
        render_search()
    elif mode == "🕒 History":
        render_history()
