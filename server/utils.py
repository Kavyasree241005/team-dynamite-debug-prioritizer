"""
utils.py — Team Dynamite Debug Prioritizer
Core utilities: Layer 1 (Ingestion & Denoising), Failure DNA Fingerprint,
embedding, clustering, DAG construction, and priority scoring.

Layer 1 produces a pandas DataFrame with columns:
  timestamp, raw_text, cleaned_text, component, severity,
  tag_fatal, tag_error, tag_sva, tag_warning  (binary 0/1 per log line)

Tag extraction is 100% deterministic via compiled regex patterns.
"""

import re
import os
import hashlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cosine as cosine_distance
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import HDBSCAN


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SEVERITY_ORDER = {"FATAL": 4, "ERROR": 3, "WARNING": 2, "INFO": 1}
SEVERITY_WEIGHT = {"FATAL": 1.0, "ERROR": 0.8, "WARNING": 0.4, "INFO": 0.1}

# ─── Compiled Regex — Deterministic Tag Detection ─────────────────────────

# UVM severity macros (case-insensitive, word-boundary anchored)
RE_UVM_FATAL   = re.compile(r"\bUVM_FATAL\b",   re.IGNORECASE)
RE_UVM_ERROR   = re.compile(r"\bUVM_ERROR\b",   re.IGNORECASE)
RE_UVM_WARNING = re.compile(r"\bUVM_WARNING\b", re.IGNORECASE)
RE_UVM_INFO    = re.compile(r"\bUVM_INFO\b",    re.IGNORECASE)

# SVA assertion failures
RE_SVA_FAIL = re.compile(r"\bSVA_FAIL\b", re.IGNORECASE)

# Combined severity extractor — tries UVM macros first, then generic keywords
RE_SEVERITY = re.compile(
    r"\b(UVM_FATAL|UVM_ERROR|UVM_WARNING|UVM_INFO|SVA_FAIL|FATAL|ERROR|WARNING|INFO)\b",
    re.IGNORECASE,
)

# Timestamp patterns for UVM-style simulation logs
# Matches: #   123ns, # 1.5ns, #     0ns
RE_SIM_TIMESTAMP = re.compile(r"#\s*([\d.]+)\s*ns")

# Matches: 2024-11-15T08:12:01.003Z or similar ISO timestamps
RE_ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)

# UVM component path extractor: "tb_foo.env.agent.driver"
RE_COMPONENT = re.compile(
    r"(?:^#\s*[\d.]+\s*ns\s+\w+\s+)"   # skip timestamp + severity
    r"([\w.]+)"                          # capture hierarchical path
)

# Dynamic tokens to mask for embedding denoising
RE_HEX_ADDR    = re.compile(r"0x[0-9a-fA-F_]+")
RE_NUMERIC_ID  = re.compile(r"(?<!\w)\d{4,}(?!\w)")
RE_SIM_TIME    = re.compile(r"#\s*[\d.]+\s*ns")
RE_ISO_TIME    = re.compile(
    r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)

# Lines to skip entirely (comments, blank, pure delimiter)
RE_SKIP_LINE = re.compile(r"^\s*(?://.*|$|={3,}\s*$)")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Ingestion & Denoising
# ═══════════════════════════════════════════════════════════════════════════

def _extract_timestamp(line: str) -> Optional[str]:
    """
    Extract the first timestamp from a log line.
    Checks UVM simulation time (#  Nns) first, then ISO format.
    Returns the matched string or None.
    """
    m = RE_SIM_TIMESTAMP.search(line)
    if m:
        return m.group(0).strip()
    m = RE_ISO_TIMESTAMP.search(line)
    if m:
        return m.group(0).strip()
    return None


def _extract_severity(line: str) -> str:
    """
    Extract the highest-priority severity from a log line.

    Priority: FATAL > ERROR > SVA_FAIL > WARNING > INFO.
    SVA_FAIL is mapped to severity "ERROR" for ranking but tagged separately.
    Returns one of: "FATAL", "ERROR", "WARNING", "INFO".
    """
    upper = line.upper()

    # Check in strict priority order — deterministic, no ambiguity
    if RE_UVM_FATAL.search(line):
        return "FATAL"
    if RE_SVA_FAIL.search(line):
        return "ERROR"          # SVA failures are error-class
    if RE_UVM_ERROR.search(line):
        return "ERROR"
    if RE_UVM_WARNING.search(line):
        return "WARNING"
    if RE_UVM_INFO.search(line):
        return "INFO"

    # Fallback: generic keywords (for non-UVM log formats)
    if "FATAL" in upper:
        return "FATAL"
    if "ERROR" in upper or "FAIL" in upper:
        return "ERROR"
    if "WARNING" in upper or "WARN" in upper:
        return "WARNING"
    return "INFO"


def _extract_component(line: str) -> str:
    """
    Extract the UVM component hierarchical path from a log line.
    Example: "tb_ddr5_mem_ctrl.env.scoreboard" from a UVM_ERROR line.
    Falls back to "unknown" if no path found.
    """
    m = RE_COMPONENT.search(line)
    if m:
        path = m.group(1)
        # Filter out things that are clearly not component paths
        if "." in path and len(path) > 3:
            return path
    # Try alternative: look for text in square brackets after severity
    m2 = re.search(r"\[([\w.]+)\]", line)
    if m2:
        return m2.group(1)
    return "unknown"


def _compute_tags(line: str) -> Dict[str, int]:
    """
    Compute binary tag indicators for a single log line.
    Each tag is 1 if the corresponding pattern is found, 0 otherwise.

    Tags:
        tag_fatal   : 1 if UVM_FATAL present
        tag_error   : 1 if UVM_ERROR present
        tag_sva     : 1 if SVA_FAIL present
        tag_warning : 1 if UVM_WARNING present

    This function is 100% deterministic — same input always produces same output.
    """
    return {
        "tag_fatal":   1 if RE_UVM_FATAL.search(line)   else 0,
        "tag_error":   1 if RE_UVM_ERROR.search(line)    else 0,
        "tag_sva":     1 if RE_SVA_FAIL.search(line)     else 0,
        "tag_warning": 1 if RE_UVM_WARNING.search(line)  else 0,
    }


def mask_dynamic_tokens(line: str) -> str:
    """
    Replace volatile tokens (addresses, IDs, timestamps) with placeholders.
    This normalises log lines for embedding — semantically identical failures
    with different addresses/timestamps will map to nearby vectors.
    """
    masked = RE_SIM_TIME.sub("<SIM_TIME>", line)
    masked = RE_ISO_TIME.sub("<TIMESTAMP>", masked)
    masked = RE_HEX_ADDR.sub("<ADDR>", masked)
    masked = RE_NUMERIC_ID.sub("<ID>", masked)
    return masked.strip()


def parse_log_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single UVM/SVA simulation log file into a structured DataFrame.

    Returns DataFrame with columns:
        timestamp    : str or None — extracted simulation/wall time
        raw_text     : str — original line, untouched
        cleaned_text : str — line with dynamic tokens masked
        component    : str — UVM hierarchical component path
        severity     : str — one of FATAL, ERROR, WARNING, INFO
        tag_fatal    : int — 1 if UVM_FATAL present, else 0
        tag_error    : int — 1 if UVM_ERROR present, else 0
        tag_sva      : int — 1 if SVA_FAIL present, else 0
        tag_warning  : int — 1 if UVM_WARNING present, else 0
        line_num     : int — 1-indexed line number in source file
        source_file  : str — basename of the source log file
    """
    records: List[Dict] = []
    basename = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for idx, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.rstrip("\n\r")

            # Skip blank lines, comments, delimiters
            if RE_SKIP_LINE.match(raw_line):
                continue

            # Core extraction
            timestamp = _extract_timestamp(raw_line)
            severity = _extract_severity(raw_line)
            component = _extract_component(raw_line)
            cleaned = mask_dynamic_tokens(raw_line)
            tags = _compute_tags(raw_line)

            records.append({
                "timestamp":    timestamp,
                "raw_text":     raw_line,
                "cleaned_text": cleaned,
                "component":    component,
                "severity":     severity,
                "tag_fatal":    tags["tag_fatal"],
                "tag_error":    tags["tag_error"],
                "tag_sva":      tags["tag_sva"],
                "tag_warning":  tags["tag_warning"],
                "line_num":     idx,
                "source_file":  basename,
            })

    df = pd.DataFrame(records)

    # Ensure correct dtypes
    if not df.empty:
        for col in ("tag_fatal", "tag_error", "tag_sva", "tag_warning"):
            df[col] = df[col].astype(int)

    return df


def parse_multiple_logs(filepaths: List[str]) -> pd.DataFrame:
    """Parse and concatenate multiple log files into a single DataFrame."""
    frames = [parse_log_file(fp) for fp in filepaths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_tag_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Return aggregate tag counts across the entire DataFrame.

    Returns:
        {"tag_fatal": N, "tag_error": N, "tag_sva": N, "tag_warning": N}
    """
    if df.empty:
        return {"tag_fatal": 0, "tag_error": 0, "tag_sva": 0, "tag_warning": 0}
    return {
        "tag_fatal":   int(df["tag_fatal"].sum()),
        "tag_error":   int(df["tag_error"].sum()),
        "tag_sva":     int(df["tag_sva"].sum()),
        "tag_warning": int(df["tag_warning"].sum()),
    }


# ── Git Blame: File & Line Extraction ──────────────────────────────────

# Patterns to extract RTL/Testbench file paths + line numbers from log lines
# Matches: file.sv:142, module.v:87, pkg.svh:33
RE_FILE_LINE_COLON = re.compile(
    r"([\w/\\.-]+\.(?:sv|v|svh|vhd|py|cpp|c|h))\s*:\s*(\d+)",
    re.IGNORECASE,
)
# Matches: (file.sv line 142), (module.v, line 87)
RE_FILE_LINE_PAREN = re.compile(
    r"\(\s*([\w/\\.-]+\.(?:sv|v|svh|vhd))\s*,?\s*line\s+(\d+)\s*\)",
    re.IGNORECASE,
)
# Matches: @ file.sv(142), file.v(87)
RE_FILE_LINE_AT = re.compile(
    r"@?\s*([\w/\\.-]+\.(?:sv|v|svh|vhd))\s*\(\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def extract_file_and_line(log_line: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse an RTL/Testbench file path and line number from a log line.

    Tries multiple common UVM log format patterns:
      - "axi_master.sv:142"
      - "(ddr5_ctrl.v line 87)"
      - "@ mem_scoreboard.sv(203)"

    Returns:
        (file_path, line_number) or (None, None) if not found.
    """
    for pattern in (RE_FILE_LINE_COLON, RE_FILE_LINE_PAREN, RE_FILE_LINE_AT):
        m = pattern.search(log_line)
        if m:
            return m.group(1), int(m.group(2))
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Embedding
# ═══════════════════════════════════════════════════════════════════════════

_model_cache: Dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load (and cache) the SentenceTransformer model."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """Compute sentence embeddings for a list of cleaned log lines."""
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    return embeddings


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Dimensionality Reduction
# ═══════════════════════════════════════════════════════════════════════════

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Apply UMAP to reduce embedding dimensions."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, max(2, embeddings.shape[0] - 1)),
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: Clustering
# ═══════════════════════════════════════════════════════════════════════════

def cluster_logs(
    reduced: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 2,
) -> np.ndarray:
    """Run HDBSCAN on UMAP-reduced embeddings. Returns cluster labels (-1 = noise)."""
    clusterer = HDBSCAN(
        min_cluster_size=min(min_cluster_size, max(2, reduced.shape[0])),
        min_samples=min(min_samples, max(1, reduced.shape[0])),
        metric="euclidean",
    )
    clusterer.fit(reduced)
    return clusterer.labels_


def cluster_logs_full(
    reduced: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 2,
) -> Dict:
    """
    Run HDBSCAN and return full clustering metadata.

    Returns dict with:
        labels:         np.ndarray — cluster labels (-1 = noise)
        probabilities:  np.ndarray — membership probability per point (0–1)
        outlier_scores: np.ndarray — GLOSH outlier score per point (higher = more outlier)
    """
    clusterer = HDBSCAN(
        min_cluster_size=min(min_cluster_size, max(2, reduced.shape[0])),
        min_samples=min(min_samples, max(1, reduced.shape[0])),
        metric="euclidean",
    )
    clusterer.fit(reduced)
    labels = clusterer.labels_
    probs = getattr(clusterer, "probabilities_", np.zeros(len(labels)))
    outlier = getattr(clusterer, "outlier_scores_", np.zeros(len(labels)))
    return {
        "labels": labels,
        "probabilities": probs if probs is not None else np.zeros(len(labels)),
        "outlier_scores": outlier if outlier is not None else np.zeros(len(labels)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Layer 5: Failure DNA Fingerprint — Causality DAG
# ═══════════════════════════════════════════════════════════════════════════

def _line_fingerprint(masked_text: str) -> str:
    """Create a short hash fingerprint for a masked log line."""
    return hashlib.md5(masked_text.encode()).hexdigest()[:10]


def build_causality_dag(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a Directed Acyclic Graph representing causal failure chains.

    Heuristic: within each source file, a higher-severity event at line N
    is considered a potential cause of a lower-or-equal severity event at
    line M > N.  Edges connect cluster representatives.

    Failure DNA Step 1: each node stores its tag fingerprint
    (tag_fatal, tag_error, tag_sva, tag_warning counts within the cluster).
    """
    dag = nx.DiGraph()

    if df.empty or "cluster" not in df.columns:
        return dag

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue
        subset = df[df["cluster"] == cluster_id].sort_values("line_num")
        rep_text = subset.iloc[0]["cleaned_text"]
        severity = subset.iloc[0]["severity"]

        # Failure DNA fingerprint: aggregate tag counts for this cluster
        dna = {
            "tag_fatal":   int(subset["tag_fatal"].sum()),
            "tag_error":   int(subset["tag_error"].sum()),
            "tag_sva":     int(subset["tag_sva"].sum()),
            "tag_warning": int(subset["tag_warning"].sum()),
        }

        dag.add_node(
            cluster_id,
            label=rep_text[:80],
            severity=severity,
            severity_rank=SEVERITY_ORDER.get(severity, 0),
            count=len(subset),
            fingerprint=_line_fingerprint(rep_text),
            dna=dna,
        )

    cluster_ids = [n for n in dag.nodes]
    for i, src in enumerate(cluster_ids):
        for dst in cluster_ids[i + 1:]:
            src_rank = dag.nodes[src]["severity_rank"]
            dst_rank = dag.nodes[dst]["severity_rank"]
            if src_rank >= dst_rank:
                dag.add_edge(src, dst, relation="causes")

    # Ensure DAG property
    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        for cycle in cycles:
            if len(cycle) >= 2:
                dag.remove_edge(cycle[-1], cycle[0])

    return dag


def find_root_causes(dag: nx.DiGraph) -> List[int]:
    """Return nodes with in-degree 0 (origin failures)."""
    return [n for n in dag.nodes if dag.in_degree(n) == 0]


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6: Priority Scoring
# ═══════════════════════════════════════════════════════════════════════════

def compute_priority_scores(
    df: pd.DataFrame,
    dag: nx.DiGraph,
    severity_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute a priority score for each cluster.

    Priority = (frequency_ratio) * severity_weight * recency_factor * (1 + root_bonus)
    """
    if severity_weights is None:
        severity_weights = SEVERITY_WEIGHT.copy()

    if df.empty or "cluster" not in df.columns:
        return pd.DataFrame()

    total_failures = len(df[df["cluster"] != -1])
    if total_failures == 0:
        total_failures = 1

    root_causes = set(find_root_causes(dag))
    records = []

    cluster_ids = sorted([c for c in df["cluster"].unique() if c != -1])
    n_clusters = len(cluster_ids)

    for rank, cluster_id in enumerate(cluster_ids):
        subset = df[df["cluster"] == cluster_id]
        count = len(subset)

        dominant_severity = (
            subset["severity"].mode().iloc[0] if len(subset) > 0 else "INFO"
        )
        sev_weight = severity_weights.get(dominant_severity, 0.1)

        frequency_ratio = count / total_failures
        recency = 1.0 - (rank / max(n_clusters, 1)) * 0.5
        root_bonus = 0.5 if cluster_id in root_causes else 0.0
        priority = frequency_ratio * sev_weight * recency * (1.0 + root_bonus)

        representative = subset.iloc[0]["cleaned_text"] if len(subset) > 0 else ""
        source_files = ", ".join(subset["source_file"].unique())

        # Aggregate tags for this cluster
        t_fatal   = int(subset["tag_fatal"].sum())
        t_error   = int(subset["tag_error"].sum())
        t_sva     = int(subset["tag_sva"].sum())
        t_warning = int(subset["tag_warning"].sum())

        records.append({
            "cluster_id":         cluster_id,
            "count":              count,
            "dominant_severity":  dominant_severity,
            "severity_weight":    sev_weight,
            "frequency_ratio":    round(frequency_ratio, 4),
            "recency_factor":     round(recency, 4),
            "root_cause":         cluster_id in root_causes,
            "priority_score":     round(priority, 6),
            "representative_log": representative[:120],
            "source_files":       source_files,
            "tag_fatal":          t_fatal,
            "tag_error":          t_error,
            "tag_sva":            t_sva,
            "tag_warning":        t_warning,
        })

    result = pd.DataFrame(records)
    result = result.sort_values("priority_score", ascending=False).reset_index(drop=True)
    result.index = result.index + 1
    result.index.name = "rank"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Layer 7: Cluster Summary Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_cluster_summary(df: pd.DataFrame, cluster_id: int) -> Dict:
    """Generate a structured summary dict for a single cluster."""
    subset = df[df["cluster"] == cluster_id]
    if subset.empty:
        return {}

    sev_counts = subset["severity"].value_counts().to_dict()
    dominant = max(sev_counts, key=sev_counts.get)
    example_lines = subset["raw_text"].head(5).tolist()
    files = subset["source_file"].unique().tolist()
    components = subset["component"].unique().tolist()

    # Aggregate tags
    tags = {
        "tag_fatal":   int(subset["tag_fatal"].sum()),
        "tag_error":   int(subset["tag_error"].sum()),
        "tag_sva":     int(subset["tag_sva"].sum()),
        "tag_warning": int(subset["tag_warning"].sum()),
    }

    summary_text = (
        f"Cluster {cluster_id} contains {len(subset)} log entries, "
        f"predominantly {dominant}-level events across {len(files)} source file(s). "
        f"The most common pattern is: \"{subset.iloc[0]['cleaned_text'][:100]}\". "
        f"This failure signature appears in: {', '.join(files)}. "
        f"Tags: {tags['tag_fatal']} FATAL, {tags['tag_error']} ERROR, "
        f"{tags['tag_sva']} SVA, {tags['tag_warning']} WARNING."
    )

    return {
        "cluster_id":          cluster_id,
        "size":                len(subset),
        "dominant_severity":   dominant,
        "severity_breakdown":  sev_counts,
        "source_files":        files,
        "components":          components,
        "tags":                tags,
        "summary":             summary_text,
        "example_lines":       example_lines,
    }
