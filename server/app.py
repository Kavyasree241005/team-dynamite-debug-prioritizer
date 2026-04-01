"""
app.py — Team Dynamite Debug Prioritizer
Professional Streamlit dashboard for AI-Enabled Debug Prioritization.

Theme: Deep navy (#0F172A), PSG blue (#1E40AF), SanDisk red (#E30613).
No emojis. Clean, modern, judge-ready.
"""

import os
import sys
import glob
import tempfile
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from pipeline import (
    DebugPrioritizer, run_full_pipeline,
    suggest_fix, simulate_fix_impact, estimate_bug_cost
)
from utils import get_tag_summary, SEVERITY_ORDER


def render_custom_card(html_content: str):
    """Safely renders HTML UI elements independently of Streamlit's Markdown engine."""
    if hasattr(st, "html"):
        st.html(html_content)
    else:
        render_custom_card(html_content)

# ═══════════════════════════════════════════════════════════════════════════
# Theme Constants
# ═══════════════════════════════════════════════════════════════════════════

NAVY       = "#0F172A"
NAVY_LIGHT = "#1E293B"
CARD_BG    = "#1A2332"
PSG_BLUE   = "#1E40AF"
PSG_LIGHT  = "#3B82F6"
SANDISK_RED = "#E30613"
WHITE      = "#F8FAFC"
GRAY       = "#94A3B8"
GRAY_DIM   = "#64748B"
SUCCESS    = "#10B981"
WARNING_CLR = "#F59E0B"

# DNA bar colors
DNA_COLORS = {
    "FATAL":   "#E30613",
    "ERROR":   "#F59E0B",
    "SVA":     "#1E40AF",
    "WARNING": "#64748B",
}

# ═══════════════════════════════════════════════════════════════════════════
# Page Config & CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Team Dynamite | Debug Prioritizer",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect fill='%230F172A' width='100' height='100' rx='20'/><text y='70' x='15' font-size='60' fill='%23E30613'>D</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_custom_card(f"""
<style>
    /* Global */
    .stApp {{
        background-color: {NAVY};
        color: {WHITE};
    }}
    .stApp header {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {NAVY_LIGHT};
        border-right: 1px solid #334155;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {WHITE} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: {NAVY_LIGHT};
        border-radius: 10px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {GRAY};
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PSG_BLUE} !important;
        color: {WHITE} !important;
    }}

    /* Cards — Global */
    .metric-card {{
        background: linear-gradient(135deg, {CARD_BG}, {NAVY_LIGHT});
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
    }}
    .metric-card .value {{
        font-size: 2.2rem;
        font-weight: 700;
        margin: 4px 0;
    }}
    .metric-card .label {{
        font-size: 0.85rem;
        color: {GRAY};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* XAI card */
    .xai-card {{
        background: {CARD_BG};
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .xai-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
    }}
    .xai-card.root-cause {{
        border-left: 4px solid {SANDISK_RED};
    }}
    .xai-card.cascading {{
        border-left: 4px solid {PSG_LIGHT};
    }}

    /* Cluster / Timeline / Noise / Fix cards — uniform hover */
    div[style*="background:{CARD_BG}"] {{
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    div[style*="background:{CARD_BG}"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.35);
    }}

    /* Table styling */
    .stDataFrame div[data-testid="stDataFrameResizable"] {{
        background-color: {CARD_BG} !important;
        border-radius: 10px;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {PSG_BLUE}, #2563EB);
        color: {WHITE};
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 600;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, #2563EB, {PSG_BLUE});
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.4);
        transform: translateY(-1px);
    }}

    /* Progress bar */
    .stProgress > div > div {{
        background-color: {PSG_BLUE} !important;
    }}

    /* File uploader */
    section[data-testid="stFileUploader"] {{
        background-color: {CARD_BG};
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 16px;
    }}

    /* Hide default Streamlit components */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Plotly dark background fix */
    .js-plotly-plot .plotly .main-svg {{
        background: transparent !important;
    }}
    .cost-rollup {{
        display: inline-block;
        padding: 4px 12px;
        background: rgba(245, 158, 11, 0.08); /* WARNING_CLR */
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 6px;
        text-align: right;
        margin-top: 10px;
        animation: fadeInUp 0.5s ease-out;
    }}
    .cost-rollup:hover {{
        background: rgba(245, 158, 11, 0.15);
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
""")


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value, color: str = WHITE):
    """Render a styled KPI metric card."""
    render_custom_card(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color: {color};">{value}</div>
    </div>
    """)


def make_dna_bar(dna: list, width: int = 200, height: int = 24) -> go.Figure:
    """Create a compact stacked horizontal bar for DNA fingerprint."""
    labels = ["FATAL", "ERROR", "SVA", "WARNING"]
    colors = [DNA_COLORS[l] for l in labels]

    fig = go.Figure()
    x_start = 0
    for i, (lbl, val) in enumerate(zip(labels, dna)):
        if val > 0:
            fig.add_trace(go.Bar(
                x=[val], y=[""], orientation="h",
                marker_color=colors[i],
                name=lbl,
                text=f"{val:.0%}" if val >= 0.15 else "",
                textposition="inside",
                textfont=dict(size=9, color="white"),
                hovertemplate=f"{lbl}: {val:.1%}<extra></extra>",
                showlegend=False,
            ))

    fig.update_layout(
        barmode="stack",
        height=height + 8,
        width=width,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, max(sum(dna), 0.01)]),
        yaxis=dict(visible=False),
    )
    return fig


def build_dag_figure(dag: nx.DiGraph, root_causes: list) -> go.Figure:
    """Build an interactive Plotly figure for the causality DAG."""
    if dag.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No clusters to display", showarrow=False,
                           font=dict(size=18, color=GRAY))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False), height=400,
        )
        return fig

    # Layout using spring layout
    pos = nx.spring_layout(dag, seed=42, k=2.0)

    # Edges
    edge_x, edge_y = [], []
    for src, dst in dag.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color=GRAY_DIM),
        hoverinfo="none",
    )

    # Arrow annotations
    annotations = []
    for src, dst in dag.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        annotations.append(dict(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.5, arrowwidth=1.5,
            arrowcolor=GRAY_DIM,
        ))

    # Nodes
    node_x = [pos[n][0] for n in dag.nodes()]
    node_y = [pos[n][1] for n in dag.nodes()]
    node_ids = list(dag.nodes())

    node_colors = []
    node_sizes = []
    node_texts = []
    hover_texts = []

    root_set = set(root_causes)
    for n in node_ids:
        data = dag.nodes[n]
        is_root = n in root_set
        sev = data.get("dominant_severity", "INFO")
        count = data.get("count", 0)
        depth = data.get("depth", 0)
        impact = data.get("impact", 0)

        if is_root:
            node_colors.append(SANDISK_RED)
            node_sizes.append(max(22, min(40, count + 15)))
        else:
            node_colors.append(PSG_LIGHT)
            node_sizes.append(max(16, min(32, count + 10)))

        node_texts.append(f"C{n}")
        hover_texts.append(
            f"<b>Cluster {n}</b><br>"
            f"Severity: {sev}<br>"
            f"Lines: {count}<br>"
            f"Depth: {depth}<br>"
            f"Impact: {impact} downstream<br>"
            f"{'ROOT CAUSE' if is_root else 'Cascading symptom'}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_texts,
        textposition="top center",
        textfont=dict(size=11, color=WHITE, family="Inter"),
        hovertext=hover_texts,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="#1E293B"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=annotations,
        font=dict(color=WHITE),
    )
    return fig


def build_umap_scatter(df: pd.DataFrame) -> go.Figure:
    """Build UMAP scatter plot colored by cluster."""
    color_map = {
        "FATAL": SANDISK_RED,
        "ERROR": "#F97316",
        "WARNING": WARNING_CLR,
        "INFO": GRAY_DIM,
    }
    df_plot = df.copy()
    df_plot["cluster_label"] = df_plot["cluster"].apply(
        lambda c: f"C{c}" if c != -1 else "Noise"
    )

    fig = px.scatter(
        df_plot, x="umap_x", y="umap_y",
        color="severity",
        color_discrete_map=color_map,
        hover_data=["source_file", "severity", "cluster_label"],
        custom_data=["cleaned_text"],
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color="#1E293B")),
        hovertemplate=(
            "<b>%{customdata[0]:.60s}</b><br>"
            "Severity: %{marker.color}<br>"
            "Cluster: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15,
            font=dict(color=GRAY, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(color=WHITE),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    render_custom_card(f"""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size: 1.6rem; font-weight: 800; color: {WHITE};
                    letter-spacing: -0.5px;">
            TEAM DYNAMITE
        </div>
        <div style="font-size: 0.75rem; color: {GRAY}; letter-spacing: 2px;
                    text-transform: uppercase; margin-top: 4px;">
            AI-Enabled Debug Prioritization
        </div>
        <div style="margin-top: 12px; height: 3px;
                    background: linear-gradient(90deg, {SANDISK_RED}, {PSG_BLUE});
                    border-radius: 2px;">
        </div>
    </div>
    """)

    st.markdown("---")
    render_custom_card(f"""
    <div style="font-size: 0.8rem; color: {GRAY};">
        <p><b style="color:{WHITE};">Pipeline Layers</b></p>
        <p>L0  Cross-Project Memory</p>
        <p>L1  Ingestion & Denoising</p>
        <p>L2  Semantic Embedding (MiniLM)</p>
        <p>L3  UMAP + HDBSCAN Clustering</p>
        <p>L4  Topological RCA (DAG)</p>
        <p>L5  Prioritization + XAI</p>
    </div>
    """)


# ═══════════════════════════════════════════════════════════════════════════
# Top Banner Removed Based on User Request
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Main Tabs
# ═══════════════════════════════════════════════════════════════════════════

tab_upload, tab_ranked, tab_dag, tab_xai, tab_timeline, tab_unique, tab_fixes = st.tabs([
    "Upload & Run",
    "Ranked Debug Tasks",
    "Root Cause DAG",
    "XAI + DNA Fingerprint",
    "Regression Timeline",
    "Unique Bugs",
    "Fix History",
])


# ─── Tab 1: Upload & Run ─────────────────────────────────────────────────
with tab_upload:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_custom_card(f"<h3 style='color:{WHITE}; margin-bottom:8px;'>Upload Simulation Logs</h3>")
        st.markdown(f"<p style='color:{GRAY}; font-size:0.9rem;'>"
                    "Upload one or more UVM/SVA simulation log files (.txt). "
                    "The pipeline processes all 5 layers automatically.</p>",
                    unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drop log files here",
            type=["txt", "log"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    with col_right:
        render_custom_card(f"<h3 style='color:{WHITE}; margin-bottom:8px;'>Quick Start</h3>")
        st.markdown(f"<p style='color:{GRAY}; font-size:0.9rem;'>"
                    "Load sample UVM logs to see the pipeline in action.</p>",
                    unsafe_allow_html=True)

        demo_clicked = st.button("Run Demo Mode", use_container_width=True)

    # Determine which files to process
    log_files_to_run = []

    if demo_clicked:
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_logs")
        log_files_to_run = sorted(glob.glob(os.path.join(sample_dir, "*.txt")))
        if log_files_to_run:
            st.session_state["run_mode"] = "demo"
            st.session_state["log_paths"] = log_files_to_run

    if uploaded_files:
        # Save uploaded files to temp dir
        tmp_dir = tempfile.mkdtemp()
        paths = []
        for uf in uploaded_files:
            path = os.path.join(tmp_dir, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            paths.append(path)
        st.session_state["run_mode"] = "upload"
        st.session_state["log_paths"] = paths

    # Run pipeline
    should_run = (
        demo_clicked
        or (uploaded_files and "results" not in st.session_state)
        or (uploaded_files and st.session_state.get("run_mode") == "upload"
            and "results" not in st.session_state)
    )

    if should_run and st.session_state.get("log_paths"):
        log_paths = st.session_state["log_paths"]
        render_custom_card(f"<p style='color:{GRAY};'>Processing {len(log_paths)} log file(s)...</p>")

        progress = st.progress(0, text="Initializing pipeline...")

        try:
            dp = DebugPrioritizer()

            progress.progress(10, text="Layer 1: Ingesting & denoising logs...")
            dp.ingest(log_paths)

            progress.progress(30, text="Layer 2-3: NLP embedding + UMAP/HDBSCAN clustering...")
            dp.embed_and_cluster()

            progress.progress(55, text="Layer 0: Querying cross-project failure memory...")
            from pipeline import query_memory
            dp.memory_insights = {}
            for _cid, _centroid in dp.cluster_centroids.items():
                if _cid == -1: continue
                _match = query_memory(_centroid)
                if _match:
                    dp.memory_insights[_cid] = _match

            progress.progress(70, text="Layer 4: Building topological root cause DAG...")
            dp.build_dag()

            progress.progress(85, text="Layer 5: Calculating priority XAI & regression trends...")
            dp.score()

            progress.progress(95, text="Assembling final results...")
            from pipeline import get_tag_summary as p_tag_summary, append_run_to_history
            rca = dp.rca_result or {}
            results = {
                "df":                dp.df,
                "embeddings":        dp.embeddings,
                "hybrid_embeddings": dp.hybrid_embeddings,
                "reduced":           dp.reduced,
                "dag":               dp.dag,
                "priority_table":    dp.priority_table,
                "xai_results":       dp.xai_results,
                "cluster_summaries": dp.cluster_summaries,
                "root_causes":       rca.get("root_causes", []),
                "cascading":         rca.get("cascading", []),
                "causal_chains":     rca.get("causal_chains", []),
                "top_chains":        rca.get("top_chains", []),
                "adjacency":         rca.get("adjacency", []),
                "root_cause_details": rca.get("root_cause_details", []),
                "depth":             rca.get("depth", {}),
                "impact":            rca.get("impact", {}),
                "node_meta":         rca.get("node_meta", {}),
                "dna_fingerprints":  dp.dna_fingerprints,
                "noise_analysis":    dp.noise_analysis,
                "cluster_centroids": dp.cluster_centroids,
                "signatures":        dp.signatures,
                "tag_summary":       p_tag_summary(dp.df),
                "memory_insights":   dp.memory_insights,
                "regression_history": append_run_to_history(dp.xai_results),
            }

            progress.progress(98, text="Generating Agentic Batch Graph Insights...")
            from pipeline import generate_batch_insights
            results["gemini_insights"] = generate_batch_insights(results)

            progress.progress(100, text="Pipeline complete!")
            st.session_state["results"] = results
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")

    # Show summary if results exist
    if "results" in st.session_state:
        res = st.session_state["results"]
        df = res["df"]
        tag_sum = res["tag_summary"]
        n_clusters = len([c for c in df["cluster"].unique() if c != -1])
        n_root = len(res.get("root_causes", []))

        render_custom_card(f"<div style='margin-top:20px;'></div>")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            metric_card("Log Lines", f"{len(df):,}", PSG_LIGHT)
        with c2:
            metric_card("Source Files", df["source_file"].nunique(), WHITE)
        with c3:
            metric_card("Clusters", n_clusters, SUCCESS)
        with c4:
            metric_card("Root Causes", n_root, SANDISK_RED)
        with c5:
            metric_card("FATAL Tags", tag_sum["tag_fatal"], SANDISK_RED)
        with c6:
            metric_card("ERROR Tags", tag_sum["tag_error"], "#F97316")

        # UMAP scatter
        render_custom_card(f"<h4 style='color:{WHITE}; margin-top:28px;'>UMAP Cluster Map</h4>")
        fig_umap = build_umap_scatter(df)
        st.plotly_chart(fig_umap, use_container_width=True, config={"displayModeBar": False}, key="umap_scatter")

        # Severity breakdown
        render_custom_card(f"<h4 style='color:{WHITE}; margin-top:12px;'>Severity Distribution</h4>")
        sev_counts = df["severity"].value_counts().to_dict()
        sev_order = ["FATAL", "ERROR", "WARNING", "INFO"]
        sev_vals = [sev_counts.get(s, 0) for s in sev_order]
        sev_colors = [SANDISK_RED, "#F97316", WARNING_CLR, GRAY_DIM]

        fig_sev = go.Figure(go.Bar(
            x=sev_order, y=sev_vals,
            marker_color=sev_colors,
            text=sev_vals, textposition="outside",
            textfont=dict(color=WHITE, size=12),
        ))
        fig_sev.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=40, r=20, t=10, b=40),
            xaxis=dict(color=GRAY), yaxis=dict(color=GRAY, gridcolor="#1E293B"),
            font=dict(color=WHITE),
        )
        st.plotly_chart(fig_sev, use_container_width=True, config={"displayModeBar": False}, key="severity_dist")

        if "gemini_insights" in res and res["gemini_insights"].get("dna_insight"):
            render_custom_card(f"""
            <div style="background:{CARD_BG}; border:1px solid #8B5CF6; border-left:4px solid #8B5CF6;
                        border-radius:8px; padding:16px 20px; margin-top:8px; margin-bottom:20px; animation: fadeInUp 0.4s ease-out;">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <span style="font-size:1.2rem;">✨</span>
                    <span style="color:#A78BFA; font-weight:700; font-size:1rem; letter-spacing:0.5px;">Gemini Insight: Run Profile Analysis</span>
                </div>
                <div style="color:{WHITE}; font-size:0.95rem; line-height:1.5; font-style:italic;">
                    "{res['gemini_insights']['dna_insight']}"
                </div>
            </div>
            """)


# ─── Tab 2: Ranked Debug Tasks ───────────────────────────────────────────
with tab_ranked:
    if "results" not in st.session_state:
        st.markdown(f"<p style='color:{GRAY}; text-align:center; padding:60px;'>"
                    "Run the pipeline first from the Upload & Run tab.</p>",
                    unsafe_allow_html=True)
    else:
        res = st.session_state["results"]
        xai_results = res.get("xai_results", [])
        dna_fp = res.get("dna_fingerprints", {})

        render_custom_card(f"<h3 style='color:{WHITE};'>Priority-Ranked Failure Clusters</h3>")
        st.markdown(f"<p style='color:{GRAY}; font-size:0.9rem; margin-bottom:20px;'>"
                    "Clusters ranked by composite score: frequency x severity x recency "
                    "x root-cause bonus x impact bonus. The Failure DNA column shows "
                    "the severity profile as a stacked bar.</p>",
                    unsafe_allow_html=True)

        for rec in xai_results:
            rank = rec["rank"]
            cid = rec["cluster_id"]
            score = rec["priority_score"]
            sev = rec["severity"]
            freq = rec["frequency"]
            is_root = rec["root_cause"]
            dna = rec.get("dna_fingerprint", [0, 0, 0, 0])
            import html
            sig = html.escape(rec["signature"][:100])
            owner = rec.get("suggested_owner", "Unassigned")
            blame_file = rec.get("blame_file", "")
            blame_line = rec.get("blame_line", "")
            blame_ref = f"{blame_file}:{blame_line}" if blame_file else ""

            # Trend & Fix data
            from pipeline import load_regression_history, get_cluster_trend, check_fix_status
            history = load_regression_history()
            trend_data = get_cluster_trend(history, rec["signature"])
            trend = trend_data["trend"]
            if trend == "worsening":
                trend_badge = f'<span style="background:#991B1B; color:#FCA5A5; padding:1px 8px; border-radius:4px; font-size:0.68rem; font-weight:600; margin-left:8px;">&#8593; Worsening</span>'
            elif trend == "improving":
                trend_badge = f'<span style="background:#064E3B; color:#6EE7B7; padding:1px 8px; border-radius:4px; font-size:0.68rem; font-weight:600; margin-left:8px;">&#8595; Improving</span>'
            elif trend == "stable":
                trend_badge = f'<span style="background:#374151; color:{GRAY}; padding:1px 8px; border-radius:4px; font-size:0.68rem; font-weight:600; margin-left:8px;">&#8596; Stable</span>'
            else:
                trend_badge = f'<span style="background:#1E293B; color:{GRAY_DIM}; padding:1px 8px; border-radius:4px; font-size:0.68rem; font-weight:600; margin-left:8px;">NEW</span>'

            fix_status = check_fix_status(rec["signature"])
            if fix_status:
                fix_badge = f'<span style="background:#991B1B; color:white; padding:1px 8px; border-radius:4px; font-size:0.68rem; font-weight:600; margin-left:8px;">&#9888; Fix Incomplete</span>'
            else:
                fix_badge = ""

            # Determine accent
            if is_root:
                border_color = SANDISK_RED
                badge = f'<span style="background:{SANDISK_RED}; color:white; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:600;">ROOT CAUSE</span>'
            else:
                border_color = PSG_LIGHT
                badge = f'<span style="background:{PSG_BLUE}; color:white; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:600;">CASCADING</span>'

            # Owner badge
            owner_badge = f'<span style="background:{PSG_BLUE}; color:white; padding:2px 10px; border-radius:4px; font-size:0.72rem; font-weight:600; margin-left:8px;">{owner}</span>'

            # Severity color
            sev_color = {
                "FATAL": SANDISK_RED, "ERROR": "#F97316",
                "WARNING": WARNING_CLR, "INFO": GRAY_DIM,
            }.get(sev, GRAY)

            with st.container():
                render_custom_card(f"""
<div style="background:{CARD_BG}; border:1px solid #334155;
            border-left:4px solid {border_color};
            border-radius:12px; padding:20px 24px; margin-bottom:12px;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="display:flex; gap:16px; align-items:center;">
            <div style="background:{NAVY}; border:2px solid {border_color};
                        border-radius:50%; width:44px; height:44px;
                        display:flex; align-items:center; justify-content:center;
                        font-weight:800; font-size:1.1rem; color:{border_color};">
                #{rank}
            </div>
            <div>
                <div style="font-weight:700; color:{WHITE}; font-size:1rem;">
                    Cluster {cid}
                    <span style="color:{sev_color}; font-size:0.85rem; margin-left:8px;">{sev}</span>
                    {badge}
                    {owner_badge}
                    {trend_badge}
                    {fix_badge}
                </div>
                <div style="color:{GRAY}; font-size:0.82rem; margin-top:4px;">
                    {sig}
                </div>
                {'<div style="color:' + GRAY_DIM + '; font-size:0.72rem; margin-top:2px; font-family:monospace;">' + blame_ref + '</div>' if blame_ref else ''}
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:1.4rem; font-weight:800; color:{border_color};">
                {score:.4f}
            </div>
            <div style="color:{GRAY}; font-size:0.75rem;">
                {freq} lines | depth {rec.get('dag_depth', 0)} | impact {rec.get('dag_impact', 0)}
            </div>
            <div class="cost-rollup" title="Cost = (Freq x 4hrs x $120/hr) + Tapeout Risk Premium">
                <span style="font-size:1.05rem; color:{WARNING_CLR}; font-weight:800;">
                    ${estimate_bug_cost(freq, sev=="FATAL", is_root):,.0f}
                </span>
                <span style="font-size:0.65rem; color:{GRAY}; display:block; text-transform:uppercase; letter-spacing:0.5px;">Est. Cost Impact</span>
            </div>
        </div>
    </div>
</div>
""")

                # HTML DNA bar inline (Replaced Plotly for extreme UI performance)
                if any(v > 0 for v in dna):
                    render_custom_card(f"""
                    <div style="display:flex; width:100%; height:6px; border-radius:3px; overflow:hidden; margin-bottom:16px; margin-top:-8px;">
                        <div style="width:{dna[0]*100}%; background:{DNA_COLORS['FATAL']};" title="FATAL: {dna[0]:.1%}"></div>
                        <div style="width:{dna[1]*100}%; background:{DNA_COLORS['ERROR']};" title="ERROR: {dna[1]:.1%}"></div>
                        <div style="width:{dna[2]*100}%; background:{DNA_COLORS['SVA']};" title="SVA: {dna[2]:.1%}"></div>
                        <div style="width:{dna[3]*100}%; background:{DNA_COLORS['WARNING']};" title="WARN: {dna[3]:.1%}"></div>
                    </div>
                    """)
                                   
                # Buttons (Suggest Fix & Mark as Fixed)
                col_b1, col_b2, col_b3 = st.columns([6, 2, 2])
                with col_b2:
                    if st.button("Suggest Fix", key=f"btn_sug_ranked_{cid}", use_container_width=True):
                        st.session_state[f"show_fix_ranked_{cid}"] = not st.session_state.get(f"show_fix_ranked_{cid}", False)
                with col_b3:
                    if not fix_status:
                        if st.button("Mark as Fixed", key=f"btn_fix_{cid}", use_container_width=True):
                            from pipeline import mark_as_fixed
                            run_id = st.session_state.get("run_id", "ui_run")
                            mark_as_fixed(rec["signature"], run_id)
                            st.rerun()
                    else:
                        st.button("Fix Pending Verification", key=f"btn_pending_{cid}", disabled=True, use_container_width=True)

                if st.session_state.get(f"show_fix_ranked_{cid}", False):
                    fix_data = suggest_fix(sig, "")
                    st.success(f"**Agentic Suggestion ({fix_data['confidence']}% confidence):**\n\n`{fix_data['fix']}`")


# ─── Tab 3: Root Cause DAG ───────────────────────────────────────────────
with tab_dag:
    if "results" not in st.session_state:
        st.markdown(f"<p style='color:{GRAY}; text-align:center; padding:60px;'>"
                    "Run the pipeline first from the Upload & Run tab.</p>",
                    unsafe_allow_html=True)
    else:
        res = st.session_state["results"]
        dag = res["dag"]
        root_causes = res.get("root_causes", [])
        # New Causal Graph Output Keys
        root_cause_details = res.get("root_cause_details", [])
        top_chains = res.get("top_chains", [])
        adjacency = res.get("adjacency", [])

        render_custom_card(f"<h3 style='color:{WHITE};'>Cluster-Level Causal Graph Analysis</h3>")
        st.markdown(f"<p style='color:{GRAY}; font-size:0.9rem; margin-bottom:24px;'>"
                    "Analyzing temporal transitions between failure clusters to identify "
                    "true root causes and cascading chains.</p>",
                    unsafe_allow_html=True)



        # ── Section 1: Root Cause Summary ───────────────────────────
        render_custom_card(f"<h4 style='color:{WHITE}; margin-top:10px; margin-bottom:16px;'>Top Root Causes</h4>")
        if not root_cause_details:
            st.info("No definitive root causes found.")
        else:
            for rc in root_cause_details:
                import html
                sig = html.escape(rc["signature"])
                reason = html.escape(rc["human_reason"])
                stats = rc["explanation"]
                cid = rc["cluster_id"]
                render_custom_card(f"""
                <div style="background:{CARD_BG}; border:1px solid #334155; border-left:4px solid {SANDISK_RED};
                            border-radius:8px; padding:16px 20px; margin-bottom:12px;
                            transition: transform 0.2s, box-shadow 0.2s;"
                     onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 16px -4px rgba(0,0,0,0.35)';"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                        <div style="font-weight:700; color:{WHITE}; font-size:1.05rem;">
                            Cluster {cid}
                            <span style="background:{SANDISK_RED}; color:white; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:700; margin-left:12px;">ROOT CAUSE</span>
                        </div>
                        <div style="color:{SANDISK_RED}; font-weight:800; font-size:1.1rem;">
                            Score: {rc['score']:.3f}
                        </div>
                    </div>
                    <div style="color:{GRAY_DIM}; font-size:0.85rem; font-family:monospace; margin-bottom:10px;">
                        {sig}
                    </div>
                    <div style="background:{NAVY}; padding:8px 12px; border-radius:6px; margin-bottom:10px;">
                        <span style="color:{PSG_LIGHT}; font-size:0.8rem; font-weight:600;">{stats}</span>
                    </div>
                    <div style="color:{GRAY}; font-size:0.9rem; font-style:italic;">
                        {reason}
                    </div>
                </div>
                """)

                col_b1, col_b2, col_b3 = st.columns([5, 2.5, 2.5])
                with col_b2:
                    if st.button("Suggest Fix", key=f"btn_sug_rc_{cid}", use_container_width=True):
                        st.session_state[f"show_fix_rc_{cid}"] = not st.session_state.get(f"show_fix_rc_{cid}", False)
                with col_b3:
                    if st.button("Simulate Fix", key=f"btn_sim_rc_{cid}", use_container_width=True):
                        st.session_state[f"show_sim_rc_{cid}"] = not st.session_state.get(f"show_sim_rc_{cid}", False)
                        
                if st.session_state.get(f"show_fix_rc_{cid}", False):
                    fix_data = suggest_fix(sig, reason)
                    st.success(f"**Agentic Suggestion ({fix_data['confidence']}% confidence):**\n\n`{fix_data['fix']}`")
                    
                if st.session_state.get(f"show_sim_rc_{cid}", False):
                    with st.spinner("Simulating downstream graph removal..."):
                        import time
                        time.sleep(0.4)
                        sim_data = simulate_fix_impact(dag, cid)
                        
                        render_custom_card(f"""
                        <div style="background:{CARD_BG}; border:1px solid #10B981; border-radius:8px; padding:16px; margin-bottom:12px; margin-top:12px; animation: fadeInUp 0.4s ease-out;">
                            <div style="color:#10B981; font-weight:700; font-size:1.05rem; margin-bottom:8px;">
                                &#9889; What-If Impact Simulation Complete
                            </div>
                            <div style="color:{WHITE}; font-size:0.9rem; line-height:1.5;">
                                Fixing this root cause eliminates <b style="color:{WARNING_CLR}; font-size:1.05rem;">{sim_data['downstream_nodes']}</b> downstream cascading transitions. 
                                Projected structural reduction in global volume: <b style="color:#10B981; font-size:1.05rem;">{sim_data['reduction_pct']}%</b>!
                            </div>
                        </div>
                        """)
                        
                        fig_sim = go.Figure(data=[
                            go.Bar(name='Current Impact', x=['Impact Volume'], y=[sim_data['before']], marker_color=SANDISK_RED),
                            go.Bar(name='Expected After Fix', x=['Impact Volume'], y=[sim_data['after']], marker_color='#10B981')
                        ])
                        fig_sim.update_layout(
                            barmode='group',
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color=WHITE),
                            height=220,
                            margin=dict(l=20, r=20, t=30, b=20),
                            title=dict(text="Simulated Target Reduction", font=dict(size=14, color=GRAY)),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(gridcolor="#1E293B"),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False}, key=f"sim_chart_rc_{cid}")
                        render_custom_card("<hr style='border-color:#334155; margin: 20px 0;'>")

        # ── Auto Gemini AI Causal Graph Explainer ───────────────
        if "gemini_insights" in res and res["gemini_insights"].get("dag_insight"):
            render_custom_card(f"""
            <div style="background:{CARD_BG}; border:1px solid #8B5CF6; border-left:4px solid #8B5CF6;
                        border-radius:8px; padding:16px 20px; margin-top:20px; margin-bottom:20px; animation: fadeInUp 0.4s ease-out;">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="font-size:1.4rem;">✨</span>
                    <span style="color:#A78BFA; font-weight:700; font-size:1.1rem; letter-spacing:0.5px;">Gemini Insight: Structural Causal Analysis</span>
                </div>
                <div style="color:{WHITE}; font-size:0.95rem; line-height:1.6; font-style:italic;">
                    "{res['gemini_insights']['dag_insight']}"
                </div>
            </div>
            """)

        # ── Section 2: Top Failure Chains ───────────────────────────
        render_custom_card(f"<h4 style='color:{WHITE}; margin-top:30px; margin-bottom:16px;'>Top Failure Chains</h4>")
        if not top_chains:
            st.info("No prominent failure chains identified.")
        else:
            for i, chain_info in enumerate(top_chains):
                readable = chain_info["readable"]
                readable_html = readable.replace(" → ", f' <strong style="color:{SANDISK_RED}; font-size:1.1rem; padding:0 6px;">&rarr;</strong> ')
                render_custom_card(f"""
                <div style="background:{CARD_BG}; border:1px solid #334155; border-radius:8px;
                            padding:14px 18px; margin-bottom:10px; font-size:0.95rem; line-height:1.6;
                            transition: transform 0.2s, box-shadow 0.2s;"
                     onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 6px 12px -4px rgba(0,0,0,0.3)';"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <span style="color:{PSG_LIGHT}; font-weight:700; margin-right:12px;">Chain {i+1}:</span>
                    <span style="color:{WHITE};">{readable_html}</span>
                    <div style="color:{GRAY}; font-size:0.75rem; margin-top:6px; text-align:right;">
                        Total Weight: {chain_info['total_weight']:.3f}
                    </div>
                </div>
                """)

        # ── Section 3: Causal Transition Table ───────────────────────
        render_custom_card(f"<h4 style='color:{WHITE}; margin-top:30px; margin-bottom:12px;'>Causal Transition Table</h4>")

        show_all = st.checkbox("Show All Transitions (include < 5% probability)", value=False)
        
        if adjacency:
            df_adj = pd.DataFrame(adjacency)
            if not show_all:
                df_adj = df_adj[df_adj["probability"] >= 0.05]
            
            if not df_adj.empty:
                # Format to display nicely
                df_show = df_adj[["from_cluster", "to_cluster", "probability", "frequency"]].copy()
                df_show.columns = ["From Cluster", "To Cluster", "Probability", "Frequency"]
                df_show["Probability"] = df_show["Probability"].apply(lambda x: f"{x:.1%}")
                
                # Render clean dataframe
                st.dataframe(
                    df_show,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No transitions match the current filter.")
        else:
            st.info("No temporal transitions detected.")


# ─── Tab 4: XAI + DNA Fingerprint ────────────────────────────────────────
with tab_xai:
    if "results" not in st.session_state:
        st.markdown(f"<p style='color:{GRAY}; text-align:center; padding:60px;'>"
                    "Run the pipeline first from the Upload & Run tab.</p>",
                    unsafe_allow_html=True)
    else:
        res = st.session_state["results"]
        xai_results = res.get("xai_results", [])
        dna_fp = res.get("dna_fingerprints", {})

        render_custom_card(f"<h3 style='color:{WHITE};'>Explainable AI Analysis</h3>")
        st.markdown(f"<p style='color:{GRAY}; font-size:0.9rem; margin-bottom:20px;'>"
                    "Each cluster receives a full explainability report: mathematical "
                    "score derivation, root-cause classification, and Failure DNA "
                    "Fingerprint interpretation.</p>",
                    unsafe_allow_html=True)

        # DNA legend
        render_custom_card(f"""
        <div style="display:flex; gap:20px; margin-bottom:20px; flex-wrap:wrap;">
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; border-radius:3px;
                            background:{DNA_COLORS['FATAL']};"></div>
                <span style="color:{GRAY}; font-size:0.8rem;">FATAL</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; border-radius:3px;
                            background:{DNA_COLORS['ERROR']};"></div>
                <span style="color:{GRAY}; font-size:0.8rem;">ERROR</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; border-radius:3px;
                            background:{DNA_COLORS['SVA']};"></div>
                <span style="color:{GRAY}; font-size:0.8rem;">SVA</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; border-radius:3px;
                            background:{DNA_COLORS['WARNING']};"></div>
                <span style="color:{GRAY}; font-size:0.8rem;">WARNING</span>
            </div>
        </div>
        """)

        for rec in xai_results:
            cid = rec["cluster_id"]
            rank = rec["rank"]
            score = rec["priority_score"]
            is_root = rec["root_cause"]
            sev = rec["severity"]
            dna = rec.get("dna_fingerprint", [0, 0, 0, 0])
            freq = rec["frequency"]
            freq_r = rec["frequency_ratio"]
            sev_w = rec["severity_weight"]
            recency = rec["recency_factor"]
            root_b = rec["root_bonus"]
            impact_b = rec["impact_bonus"]
            dag_depth = rec.get("dag_depth", 0)
            dag_impact = rec.get("dag_impact", 0)
            import html
            sig = html.escape(rec["signature"][:120])
            src_files = ", ".join(rec.get("source_files", []))
            tag_counts = rec.get("tag_counts", {})

            # DNA interpretation
            dna_labels = ["FATAL", "ERROR", "SVA", "WARNING"]
            uvm_labels = ["UVM_FATAL", "UVM_ERROR", "SVA_FAIL", "UVM_WARNING"]
            dominant_idx = max(range(4), key=lambda i: dna[i]) if any(v > 0 for v in dna) else -1
            if dominant_idx >= 0 and dna[dominant_idx] > 0:
                dominant_pct = dna[dominant_idx] * 100
                dominant_uvm = uvm_labels[dominant_idx]
                dna_text = f"This cluster is {dominant_pct:.0f}% {dominant_uvm}"
                if dominant_idx == 0:
                    dna_text += " → indicates critical hardware/system failure requiring immediate escalation."
                elif dominant_idx == 1:
                    dna_text += " → indicates functional mismatches or protocol violations in the design."
                elif dominant_idx == 2:
                    dna_text += " → driven by formal assertion failures, high-confidence RTL bug indicators."
                else:
                    dna_text += " → advisory-level issues that may escalate under stress conditions."
            else:
                dna_text = "INFO-only cluster with no severity tags."

            card_class = "root-cause" if is_root else "cascading"
            role_label = "ROOT CAUSE" if is_root else "CASCADING SYMPTOM"
            role_color = SANDISK_RED if is_root else PSG_LIGHT

            render_custom_card(f"""
<div class="xai-card {card_class}">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
        <div>
            <div style="font-size:1.1rem; font-weight:700; color:{WHITE};">
                Rank #{rank} &mdash; Cluster {cid}
            </div>
            <div style="color:{GRAY}; font-size:0.82rem; margin-top:4px;">{sig}</div>
        </div>
        <div style="text-align:right;">
            <span style="background:{role_color}; color:white; padding:3px 12px;
                         border-radius:6px; font-size:0.75rem; font-weight:700;">
                {role_label}
            </span>
            <div style="font-size:1.6rem; font-weight:800; color:{role_color}; margin-top:4px;">
                {score:.6f}
            </div>
        </div>
    </div>

    <div style="background:{NAVY}; border-radius:8px; padding:14px 18px; margin-bottom:14px;
                font-family:monospace; font-size:0.82rem; color:{GRAY}; line-height:1.7;">
        <div style="color:{WHITE}; font-weight:600; margin-bottom:6px;">Score Derivation</div>
        P = freq_ratio x sev_weight x recency x (1 + root_bonus) x (1 + impact_bonus)<br>
        P = {freq_r:.4f} x {sev_w:.2f} x {recency:.4f} x (1 + {root_b:.1f}) x (1 + {impact_b:.4f})<br>
        P = <span style="color:{WHITE}; font-weight:700;">{score:.6f}</span><br><br>
        freq_ratio &nbsp;&nbsp;= {freq}/{rec.get('frequency', freq)} lines = {freq_r:.4f}<br>
        sev_weight &nbsp;&nbsp;= SEVERITY_WEIGHT[{sev}] = {sev_w:.2f}<br>
        recency &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {recency:.4f}<br>
        root_bonus &nbsp;&nbsp;= {root_b:.1f} {'(ROOT CAUSE)' if is_root else '(cascading)'}<br>
        impact_bonus = 0.1 x {dag_impact} downstream = {impact_b:.4f}
    </div>

    <div style="margin-bottom:14px;">
        <div style="color:{WHITE}; font-weight:600; font-size:0.85rem; margin-bottom:8px;">
            Failure DNA Fingerprint
        </div>
        <div style="display:flex; gap:12px; margin-bottom:8px;">
            <span style="color:{DNA_COLORS['FATAL']};">FATAL: {dna[0]:.1%}</span>
            <span style="color:{DNA_COLORS['ERROR']};">ERROR: {dna[1]:.1%}</span>
            <span style="color:{DNA_COLORS['SVA']};">SVA: {dna[2]:.1%}</span>
            <span style="color:{DNA_COLORS['WARNING']};">WARN: {dna[3]:.1%}</span>
        </div>
        <div style="color:{GRAY}; font-size:0.85rem; font-style:italic;">
            {dna_text}
        </div>
    </div>

    <div style="display:flex; gap:24px; color:{GRAY_DIM}; font-size:0.78rem;">
        <span>Depth: {dag_depth}</span>
        <span>Impact: {dag_impact} downstream</span>
        <span>Tags: F={tag_counts.get('FATAL',0)} E={tag_counts.get('ERROR',0)} S={tag_counts.get('SVA',0)} W={tag_counts.get('WARNING',0)}</span>
        <span>Files: {src_files}</span>
    </div>
</div>
""")

            # Memory Match HTML Component
            memory_insight_html = ""
            mem_data = res.get("memory_insights", {}).get(cid)
            if mem_data:
                m_proj = mem_data.get("project_name", "Unknown")
                m_note = mem_data.get("previous_fix_note", "No notes.")
                m_sim = mem_data.get("similarity", 0.0)
                memory_insight_html = f"""
                <div style="background:rgba(16, 185, 129, 0.08); border:1px solid #10B981; border-radius:8px; padding:12px 16px; margin-top:14px; margin-bottom:4px;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                        <span style="background:#10B981; color:white; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;">&#129504; MEMORY MATCH</span>
                        <span style="color:#10B981; font-weight:700; font-size:0.75rem;">({m_sim:.1%} Match)</span>
                    </div>
                    <div style="color:{WHITE}; font-size:0.85rem; line-height:1.5;">
                        Seen before in <b>{m_proj}</b> — previously fixed by: <i>{m_note}</i>
                    </div>
                </div>
                """

            if memory_insight_html:
                render_custom_card(memory_insight_html)

            # DNA bar chart
            if any(v > 0 for v in dna):
                fig_bar = make_dna_bar(dna, width=500, height=22)
                st.plotly_chart(fig_bar, use_container_width=False,
                               config={"displayModeBar": False},
                               key=f"dna_xai_{cid}")


# ─── Tab 5: Regression Timeline ──────────────────────────────────────────
with tab_timeline:
    if "results" not in st.session_state:
        st.markdown(f"<p style='color:{GRAY}; text-align:center; padding:60px;'>"
                    "Run the pipeline first from the Upload & Run tab.</p>",
                    unsafe_allow_html=True)
    else:
        res = st.session_state["results"]
        xai_results = res.get("xai_results", [])

        render_custom_card(f"<h3 style='color:{WHITE}; margin-bottom:8px;'>Regression Trend Timeline</h3>")
        st.markdown(f"<p style='color:{GRAY}; margin-bottom:24px;'>"
                    "Track how cluster priority scores evolve across pipeline runs. "
                    "Worsening trends are highlighted in red.</p>",
                    unsafe_allow_html=True)

        from pipeline import load_regression_history, get_cluster_trend
        history = load_regression_history()

        if not history:
            st.markdown(f"<p style='color:{GRAY_DIM}; text-align:center; padding:40px;'>"
                        "No previous runs recorded yet. Run the pipeline multiple times "
                        "to see trend data.</p>", unsafe_allow_html=True)
        else:
            # Show timeline for each cluster
            for rec in xai_results:
                cid = rec["cluster_id"]
                sig = rec["signature"][:80]
                trend_data = get_cluster_trend(history, rec["signature"])
                scores = trend_data["scores"]
                run_ids = trend_data["run_ids"]
                trend = trend_data["trend"]
                delta = trend_data["delta"]

                if not scores:
                    continue

                # Trend color
                if trend == "worsening":
                    trend_color = SANDISK_RED
                    trend_icon = "&#8593;"
                    trend_label = "Worsening"
                elif trend == "improving":
                    trend_color = "#10B981"
                    trend_icon = "&#8595;"
                    trend_label = "Improving"
                elif trend == "stable":
                    trend_color = GRAY
                    trend_icon = "&#8596;"
                    trend_label = "Stable"
                else:
                    trend_color = GRAY_DIM
                    trend_icon = ""
                    trend_label = "New"

                with st.container():
                    render_custom_card(f"""
                    <div style="background:{CARD_BG}; border:1px solid #334155;
                                border-radius:12px; padding:16px 20px; margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="font-weight:700; color:{WHITE}; font-size:0.95rem;">
                                    Cluster {cid}
                                </span>
                                <span style="color:{GRAY}; font-size:0.8rem; margin-left:12px;">
                                    {rec['severity']} | {len(scores)} run(s) recorded
                                </span>
                            </div>
                            <div>
                                <span style="color:{trend_color}; font-weight:700; font-size:0.85rem;">
                                    {trend_icon} {trend_label}
                                </span>
                                <span style="color:{GRAY_DIM}; font-size:0.72rem; margin-left:8px;">
                                    delta: {delta:+.6f}
                                </span>
                            </div>
                        </div>
                        <div style="color:{GRAY_DIM}; font-size:0.75rem; margin-top:4px;">
                            {sig}
                        </div>
                    </div>
                    """)

                    # Full sparkline chart
                    if len(scores) >= 2:
                        fig_tl = go.Figure()
                        fig_tl.add_trace(go.Scatter(
                            x=run_ids, y=scores,
                            mode="lines+markers",
                            line=dict(color=trend_color, width=2.5),
                            marker=dict(size=6, color=trend_color,
                                       line=dict(width=1, color=WHITE)),
                            fill="tozeroy",
                            fillcolor=f"rgba({','.join(str(int(trend_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
                            hovertemplate="%{x}<br>Score: %{y:.6f}<extra></extra>",
                        ))
                        fig_tl.update_layout(
                            height=140,
                            margin=dict(l=40, r=20, t=10, b=30),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(color=GRAY, gridcolor="#1E293B",
                                      tickfont=dict(size=9, color=GRAY_DIM)),
                            yaxis=dict(color=GRAY, gridcolor="#1E293B",
                                      tickfont=dict(size=9, color=GRAY_DIM),
                                      tickformat=".6f"),
                            font=dict(color=WHITE),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_tl, use_container_width=True,
                                       config={"displayModeBar": False},
                                       key=f"timeline_{cid}")


# ─── Tab 6: Unique Bugs (Noise Confidence Scoring) ───────────────────────
with tab_unique:
    if "results" not in st.session_state:
        st.markdown(f"<p style='color:{GRAY}; text-align:center; padding:60px;'>"
                    "Run the pipeline first from the Upload & Run tab.</p>",
                    unsafe_allow_html=True)
    else:
        res = st.session_state["results"]
        noise_data = res.get("noise_analysis", [])

        render_custom_card(f"<h3 style='color:{WHITE}; margin-bottom:8px;'>Unique Bugs (Noise Confidence Scoring)</h3>")
        st.markdown(f"<p style='color:{GRAY}; margin-bottom:24px;'>"
                    "Log lines that HDBSCAN classified as noise (-1 cluster). "
                    "High novelty = potentially new, undiscovered bug class. "
                    "Sorted by novelty score descending.</p>",
                    unsafe_allow_html=True)

        if not noise_data:
            st.markdown(f"<p style='color:{GRAY_DIM}; text-align:center; padding:40px;'>"
                        "No noise points detected. All log lines were assigned to clusters.</p>",
                        unsafe_allow_html=True)
        else:
            # KPI Summary
            n_hard = sum(1 for n in noise_data if n["noise_class"] == "Hard Noise")
            n_med  = sum(1 for n in noise_data if n["noise_class"] == "Medium Noise")
            n_soft = sum(1 for n in noise_data if n["noise_class"] == "Soft Noise")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"<div style='background:{CARD_BG}; border:1px solid #334155; border-radius:8px; padding:16px; text-align:center;'>"
                            f"<div style='font-size:1.6rem; font-weight:800; color:{WHITE};'>{len(noise_data)}</div>"
                            f"<div style='color:{GRAY}; font-size:0.75rem;'>Total Noise Points</div></div>",
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='background:{CARD_BG}; border:1px solid #334155; border-radius:8px; padding:16px; text-align:center;'>"
                            f"<div style='font-size:1.6rem; font-weight:800; color:{SANDISK_RED};'>{n_hard}</div>"
                            f"<div style='color:{GRAY}; font-size:0.75rem;'>Hard Noise</div></div>",
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div style='background:{CARD_BG}; border:1px solid #334155; border-radius:8px; padding:16px; text-align:center;'>"
                            f"<div style='font-size:1.6rem; font-weight:800; color:#F59E0B;'>{n_med}</div>"
                            f"<div style='color:{GRAY}; font-size:0.75rem;'>Medium Noise</div></div>",
                            unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div style='background:{CARD_BG}; border:1px solid #334155; border-radius:8px; padding:16px; text-align:center;'>"
                            f"<div style='font-size:1.6rem; font-weight:800; color:#10B981;'>{n_soft}</div>"
                            f"<div style='color:{GRAY}; font-size:0.75rem;'>Soft Noise</div></div>",
                            unsafe_allow_html=True)

            render_custom_card("<div style='height:16px;'></div>")

            # Noise cards (Limited to top 100 to prevent thousands of DOM nodes lagging the browser)
            if len(noise_data) > 100:
                st.warning(f"Showing top 100 out of {len(noise_data)} noise points to prevent dashboard crashing and retain peak performance.")
                
            for i, nd in enumerate(noise_data[:100]):
                import html
                sig = html.escape(nd["signature"])
                sev = nd["severity"]
                novelty = nd["novelty_score"]
                nc = nd["noise_class"]
                nearest = nd["nearest_cluster"]
                cdist = nd["centroid_dist"]
                oscore = nd["outlier_score"]

                # Badge color
                if nc == "Hard Noise":
                    badge_bg = SANDISK_RED
                    badge_fg = "white"
                    bar_color = SANDISK_RED
                elif nc == "Medium Noise":
                    badge_bg = "#F59E0B"
                    badge_fg = "#1E293B"
                    bar_color = "#F59E0B"
                else:
                    badge_bg = "#10B981"
                    badge_fg = "white"
                    bar_color = "#10B981"

                sev_color = {
                    "FATAL": SANDISK_RED, "ERROR": "#F97316",
                    "WARNING": WARNING_CLR, "INFO": GRAY_DIM,
                }.get(sev, GRAY)

                with st.container():
                    render_custom_card(f"""
                    <div style="background:{CARD_BG}; border:1px solid #334155;
                                border-left:4px solid {bar_color};
                                border-radius:12px; padding:16px 20px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="flex:1;">
                                <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                                    <span style="background:{badge_bg}; color:{badge_fg}; padding:2px 10px;
                                                 border-radius:4px; font-size:0.72rem; font-weight:700;">{nc}</span>
                                    <span style="color:{sev_color}; font-size:0.8rem; font-weight:600;">{sev}</span>
                                    <span style="color:{GRAY_DIM}; font-size:0.72rem;">nearest: Cluster {nearest}</span>
                                </div>
                                <div style="color:{GRAY}; font-size:0.8rem; font-family:monospace; word-break:break-all;">
                                    {sig}
                                </div>
                            </div>
                            <div style="text-align:right; min-width:180px;">
                                <div style="font-size:1.3rem; font-weight:800; color:{bar_color};">
                                    {novelty:.1f}
                                </div>
                                <div style="color:{GRAY_DIM}; font-size:0.68rem;">Novelty Score</div>
                                <div style="background:#1E293B; border-radius:4px; height:6px; width:120px;
                                            margin-top:4px; display:inline-block; overflow:hidden;">
                                    <div style="background:{bar_color}; height:100%; width:{novelty}%;
                                                border-radius:4px;"></div>
                                </div>
                                <div style="color:{GRAY_DIM}; font-size:0.62rem; margin-top:4px;">
                                    dist: {cdist:.4f} | outlier: {oscore:.4f}
                                </div>
                            </div>
                        </div>
                    </div>
                    """)


# ─── Tab 7: Fix History ──────────────────────────────────────────────────
with tab_fixes:
    render_custom_card(f"<h3 style='color:{WHITE}; margin-bottom:8px;'>Fix Verification History</h3>")
    st.markdown(f"<p style='color:{GRAY}; margin-bottom:24px;'>"
                "Track bugs marked as fixed. If a fixed bug reappears in the current log run, "
                "it is marked as Fix Incomplete. If absent, it is considered Resolved.</p>",
                unsafe_allow_html=True)

    if "results" in st.session_state and st.session_state["results"].get("gemini_insights", {}).get("memory_insight"):
        render_custom_card(f"""
        <div style="background:{CARD_BG}; border:1px solid #8B5CF6; border-left:4px solid #8B5CF6;
                    border-radius:8px; padding:16px 20px; margin-bottom:20px; animation: fadeInUp 0.4s ease-out;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:1.2rem;">✨</span>
                <span style="color:#A78BFA; font-weight:700; font-size:1rem; letter-spacing:0.5px;">Gemini Insight: Cross-Project Memory</span>
            </div>
            <div style="color:{WHITE}; font-size:0.95rem; line-height:1.5; font-style:italic;">
                "{st.session_state['results']['gemini_insights']['memory_insight']}"
            </div>
        </div>
        """)

    from pipeline import load_fix_state
    fix_state = load_fix_state()

    if not fix_state:
        st.markdown(f"<p style='color:{GRAY_DIM}; text-align:center; padding:40px;'>"
                    "No bugs have been marked as fixed yet.</p>",
                    unsafe_allow_html=True)
    else:
        # Determine current signatures
        current_sigs = set()
        if "results" in st.session_state:
            res = st.session_state["results"]
            current_sigs = set([rec["signature"][:80] for rec in res.get("xai_results", [])])

        for raw_sig, data in dict(sorted(fix_state.items(), key=lambda item: item[1].get('fixed_on', ''), reverse=True)).items():
            fixed_on = data.get("fixed_on", "Unknown date")[:10]  # Just YYYY-MM-DD
            import html
            sig = html.escape(raw_sig)
            
            if raw_sig in current_sigs:
                status_color = "#991B1B" # Red
                status_text = "Fix Incomplete / Regression"
                status_icon = "&#9888;"
            else:
                status_color = "#064E3B" # Green
                status_text = "Resolved"
                status_icon = "&#10003;"

            with st.container():
                render_custom_card(f"""
                <div style="background:{CARD_BG}; border:1px solid #334155;
                            border-left:4px solid {status_color};
                            border-radius:12px; padding:16px 20px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="flex:1;">
                            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                                <span style="background:{status_color}; color:white; padding:2px 10px;
                                             border-radius:4px; font-size:0.75rem; font-weight:700;">
                                    {status_icon} {status_text}
                                </span>
                                <span style="color:{GRAY}; font-size:0.8rem; font-weight:600;">
                                    Marked fixed on: {fixed_on}
                                </span>
                            </div>
                            <div style="color:{GRAY_DIM}; font-size:0.8rem; font-family:monospace; word-break:break-all;">
                                {sig}...
                            </div>
                        </div>
                    </div>
                </div>
                """)
