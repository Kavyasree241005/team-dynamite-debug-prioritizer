# Team Dynamite — AI-Enabled Debug Prioritizer

**Automated root-cause analysis and failure prioritization for SoC verification logs.**

An AI-accelerated pipeline that ingests raw UVM/SVA simulation logs, clusters failures
semantically, builds a causal DAG, and delivers priority-ranked debug tasks with full
explainability — including per-cluster **Failure DNA Fingerprints**.

---

## Quick Start

### 1. Clone & Install

```bash
cd team-dynamite-debug-prioritizer

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Run Self-Test (Validates Entire Pipeline)

```bash
python pipeline.py --test
```

This runs all 5 layers on the sample logs and validates:
- Layer 1 parsing (tag extraction, cleaning)
- Layer 2 embeddings (384-dim text + 4-dim tag fusion)
- Layer 3 clustering (UMAP + HDBSCAN)
- Layer 4 causal DAG (acyclicity, root-cause identification)
- Layer 5 XAI (priority scoring, explanation generation)
- DNA fingerprint integrity (per-cluster severity vectors)

### 3. Launch Dashboard

```bash
streamlit run app.py
```

Then click **"Run Demo Mode"** to load the sample logs, or upload your own.

### 4. Run CLI Pipeline

```bash
python pipeline.py                     # defaults to sample_logs/
python pipeline.py path/to/your/logs/  # custom log directory
```

### 5. Final Enhancements Testing

```bash
python test_pipeline.py                # Runs the full pipeline and tests all layers + noise
```

---

## Pipeline Architecture

| Layer | Name | Method | Output |
|-------|------|--------|--------|
| L1 | Ingestion & Denoising | Regex parsing, timestamp extraction, file/line mapping | Structured DataFrame, Tag flags, Auto-Assignees |
| L2 | Representation | `all-MiniLM-L6-v2` embeddings + tag-ratio fusion (0.85/0.15) | 388-dim hybrid embedding vectors |
| L3 | Clustering | UMAP (2-D reduction) + HDBSCAN (density clustering) | Cluster labels, UMAP coordinates, Outlier prob |
| L4 | Cluster Causal Graph | Transition frequencies + probability pruning + degree scoring | Root Cause Summaries, Top Failure Chains, Transition Table |
| L5 | Prioritization + XAI | Composite scoring with mathematical proof | Ranked clusters with explainability reports |
| | Noise Confidence | Centroid distance + HDBSCAN outlier scoring | Novelty score (Soft/Medium/Hard Noise) |
| | Regression Timeline | JSON persistent storage of priority scores | Multi-run tracking and trend deltas |

### Priority Score Formula

```
P = freq_ratio x sev_weight x recency x (1 + root_bonus) x (1 + impact_bonus)
```

- `freq_ratio` — proportion of failure lines in the cluster
- `sev_weight` — weight of the dominant severity level (FATAL=1.0, ERROR=0.7, WARNING=0.3, INFO=0.1)
- `recency` — temporal ordering factor
- `root_bonus` — +0.5 if the cluster is a DAG root cause (no upstream cause)
- `impact_bonus` — 0.1 x number of downstream clusters affected

---

## New High-Impact Features (6 Upgrades)

### Failure DNA Fingerprint
A 4-dimensional severity ratio vector (`[fatal, error, sva, warning]`) extracted from UVM tags and concatenated with text embeddings to force structurally identical but functionally different bugs to separate cleanly during UMAP dimensionality reduction.

### Git Blame Auto-Assignee
Using intelligent regex mapping during log ingestion, the pipeline extracts the exact RTL/Testbench file and line number (`axi_master.sv:142`) responsible for the failure and assigns a realistic Suggested Owner to the cluster, converting a blank log artifact into an actionable Jira-ready bug ticket.

### Regression Trend Timeline
Persists the final Prioritization Score (`P_final`) for each unique failure cluster signature across multiple pipeline runs. Visualized in the dashboard using inline Sparklines and badging (↑ Worsening / ↓ Improving), engineers can track if a cascading failure is getting worse over CI/CD cycles.

### Noise Confidence & Unique Bugs
Instead of ignoring HDBSCAN "noise" points (`cluster=-1`), those logs are scored for Novelty using a fused metric of distance-to-nearest-cluster-centroid and HDBSCAN GLOSH outlier scores. Unique true-negative bugs that didn't cluster are surfaced as Hard/Medium/Soft Noise in the Unique Bugs panel, guaranteeing novel zero-day issues aren't suppressed.

### Fix Verification Loop
A closed-loop system where engineers click a UI button to persistently mark a cluster signature as 'Fixed'. The system automatically compares future ingestion runs against the fixed memory state, rendering instant `Fix Incomplete` or `Regression` warnings if the exact bug returns.

### Cross-Project Failure Memory
A lightweight 'Layer 0' index utilizing standard NumPy cosine similarity to emulate FAISS capabilities cleanly. It permanently catalogs resolved cluster vectors and surfaces identical root-cause bugs appearing in *new* projects, auto-recommending the identical historical Git fix context instantly.

---

## Dashboard Features

- **Dark navy theme** (#0F172A background, #1E40AF accents, #E30613 highlights)
- **Upload & Run tab**: File upload + Demo Mode + KPI metrics + UMAP scatter + severity bars
- **Ranked Debug Tasks tab**: Priority-ranked cluster cards with DNA fingerprint stacked bars, Git Blame assignment, and Sparkline Regression Trends
- **Cluster Causal Graph tab**: Professional summary of Top Root Causes, Top Failure Chains, and Causal Transition Table
- **XAI + DNA Fingerprint tab**: Full explainability cards with math derivation + DNA interpretation
- **Regression Timeline tab**: Full per-cluster regression charts over time
- **Unique Bugs tab**: HDBSCAN noise list ranked by Novelty Score

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Language Model | `sentence-transformers` (all-MiniLM-L6-v2) | 2.2.2 |
| Deep Learning | `torch` (CPU) | 2.1.0+cpu |
| Dimensionality Reduction | `umap-learn` | 0.5.5 |
| Density Clustering | `hdbscan` | 0.8.33 |
| Graph Analysis | `networkx` | 3.2.1 |
| ML Utilities | `scikit-learn` | 1.3.2 |
| Data | `pandas`, `numpy` | 2.1.4, 1.26.2 |
| Dashboard | `streamlit` | 1.29.0 |
| Visualization | `plotly` | 5.18.0 |

---

## Project Structure

```
team-dynamite-debug-prioritizer/
├── app.py               # Streamlit dashboard (7 Tabs)
├── pipeline.py          # Full ML pipeline (Layers 0-5 + DNA + Trend + Noise + Memory)
├── utils.py             # Core utilities (parsing, embedding, clustering, DAG)
├── test_pipeline.py     # Automated CLI test of all engine layers
├── regression_history.json  # Persistent trend storage
├── fix_state.json           # Fix Verification Loop state
├── project_memory.json      # Cross-Project Memory index
├── DEMO_SCRIPT.md       # 4-minute hackathon pitch script
├── sample_logs/
│   ├── log_run_1.txt    # DDR5 memory controller verification log
│   ├── log_run_2.txt    # AXI4 interconnect verification log
│   └── log_run_3.txt    # GICv3 interrupt controller + DMA log
├── requirements.txt     # Pinned dependencies (CPU-friendly)
└── README.md            # This file
```

---

## Team Dynamite | SanDisk | Western Digital
