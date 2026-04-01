"""
python_bridge.py — Node.js ↔ Python bridge for Team Dynamite Debug Prioritizer.

Runs the full ML pipeline from the command line (no Streamlit dependency).
Called by server.js via child_process.spawn().

Usage:
    python python_bridge.py run-demo
    python python_bridge.py upload <file1> <file2> ...
    python python_bridge.py suggest-fix "<signature>"
    python python_bridge.py mark-fixed "<signature>" "<run_id>"
    python python_bridge.py fix-state
    python python_bridge.py regression-history
"""

import os
import sys
import json
import glob
import warnings
warnings.filterwarnings("ignore")

# Patch streamlit imports so pipeline.py can load without streamlit running
import types
st_mock = types.ModuleType("streamlit")
st_mock.cache_resource = lambda *a, **kw: (lambda f: f) if not a else a[0]
st_mock.cache_data = lambda *a, **kw: (lambda f: f) if not a else a[0]
st_mock.set_page_config = lambda **kw: None
st_mock.html = lambda *a, **kw: None
st_mock.sidebar = types.SimpleNamespace()
st_mock.markdown = lambda *a, **kw: None
st_mock.progress = lambda *a, **kw: types.SimpleNamespace(progress=lambda *a2, **kw2: None)
st_mock.columns = lambda *a, **kw: [types.SimpleNamespace() for _ in range(10)]
st_mock.tabs = lambda *a, **kw: [types.SimpleNamespace() for _ in range(10)]
st_mock.rerun = lambda: None
st_mock.error = lambda *a, **kw: None
st_mock.session_state = {}
sys.modules["streamlit"] = st_mock

import numpy as np
import pandas as pd

# Now import pipeline (it will use our mock st)
from pipeline import (
    DebugPrioritizer,
    suggest_fix,
    simulate_fix_impact,
    load_fix_state,
    save_fix_state,
    mark_as_fixed as pipeline_mark_fixed,
    load_regression_history,
    append_run_to_history,
    get_tag_summary,
    query_memory,
    generate_batch_insights,
)
import networkx as nx


def serialize_results(results):
    """Convert pipeline results to JSON-serializable format."""
    out = {}

    # DataFrame → list of dicts
    if "df" in results and results["df"] is not None:
        df = results["df"]
        records = []
        for _, row in df.iterrows():
            rec = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                elif isinstance(val, np.ndarray):
                    val = val.tolist()
                elif pd.isna(val):
                    val = None
                rec[col] = val
            records.append(rec)
        out["df"] = records
    else:
        out["df"] = []

    # XAI results (already list of dicts, but may contain numpy)
    xai = results.get("xai_results", [])
    out["xai_results"] = json.loads(json.dumps(xai, default=_np_default))

    # Root causes (list of ints)
    out["root_causes"] = [int(x) for x in results.get("root_causes", [])]

    # Root cause details
    rcd = results.get("root_cause_details", [])
    if not rcd and results.get("rca_result"):
        rcd = results["rca_result"].get("root_cause_details", [])
    out["root_cause_details"] = json.loads(json.dumps(rcd, default=_np_default))

    # Top chains
    tc = results.get("top_chains", [])
    if not tc and results.get("rca_result"):
        tc = results["rca_result"].get("top_chains", [])
    out["top_chains"] = json.loads(json.dumps(tc, default=_np_default))

    # Adjacency
    adj = results.get("adjacency", [])
    if not adj and results.get("rca_result"):
        adj = results["rca_result"].get("adjacency", [])
    out["adjacency"] = json.loads(json.dumps(adj, default=_np_default))

    # DAG nodes/edges for visualization
    dag = results.get("dag")
    if dag is not None and hasattr(dag, "nodes"):
        nodes = []
        for n in dag.nodes():
            nd = dict(dag.nodes[n])
            nd["id"] = int(n)
            # Clean numpy values
            for k, v in nd.items():
                if isinstance(v, (np.integer,)):
                    nd[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    nd[k] = float(v)
                elif isinstance(v, np.ndarray):
                    nd[k] = v.tolist()
            nodes.append(nd)
        edges = []
        for u, v, d in dag.edges(data=True):
            ed = {"source": int(u), "target": int(v)}
            for k, val in d.items():
                if isinstance(val, (np.integer,)):
                    ed[k] = int(val)
                elif isinstance(val, (np.floating,)):
                    ed[k] = float(val)
                else:
                    ed[k] = val
            edges.append(ed)
        out["dag"] = {"nodes": nodes, "edges": edges}
    else:
        out["dag"] = {"nodes": [], "edges": []}

    # Tag summary
    out["tag_summary"] = results.get("tag_summary", {})
    if isinstance(out["tag_summary"], dict):
        out["tag_summary"] = {k: int(v) if isinstance(v, (np.integer,)) else v
                              for k, v in out["tag_summary"].items()}

    # Noise analysis
    out["noise_analysis"] = json.loads(
        json.dumps(results.get("noise_analysis", []), default=_np_default)
    )

    # DNA fingerprints
    dna = results.get("dna_fingerprints", {})
    out["dna_fingerprints"] = {str(k): v for k, v in dna.items()}

    # Memory insights
    mem = results.get("memory_insights", {})
    out["memory_insights"] = json.loads(json.dumps(
        {str(k): v for k, v in mem.items()}, default=_np_default
    ))

    # Regression history
    rh = results.get("regression_history", {})
    out["regression_history"] = json.loads(json.dumps(rh, default=_np_default))

    # Gemini insights
    out["gemini_insights"] = results.get("gemini_insights", {})

    return out


def _np_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def cmd_run_demo():
    """Run the pipeline with sample logs."""
    # sample_logs lives at project root (parent of server/)
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_logs")
    if not os.path.exists(sample_dir):
        # Fallback: check current directory
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_logs")

    log_files = sorted(glob.glob(os.path.join(sample_dir, "*.txt")))
    if not log_files:
        print(json.dumps({"error": f"No sample log files found in {sample_dir}"}))
        sys.exit(1)

    dp = DebugPrioritizer()
    results = dp.run(log_files)

    # Generate Gemini batch insights
    try:
        rca = dp.rca_result or {}
        full_results = dict(results)
        full_results["root_cause_details"] = rca.get("root_cause_details", [])
        full_results["top_chains"] = rca.get("top_chains", [])
        full_results["adjacency"] = rca.get("adjacency", [])
        full_results["rca_result"] = rca
        full_results["gemini_insights"] = generate_batch_insights(full_results)
    except Exception as e:
        full_results = dict(results)
        full_results["root_cause_details"] = (dp.rca_result or {}).get("root_cause_details", [])
        full_results["top_chains"] = (dp.rca_result or {}).get("top_chains", [])
        full_results["adjacency"] = (dp.rca_result or {}).get("adjacency", [])
        full_results["rca_result"] = dp.rca_result or {}
        full_results["gemini_insights"] = {
            "dna_insight": str(e),
            "dag_insight": str(e),
            "memory_insight": str(e),
        }

    serialized = serialize_results(full_results)
    print(json.dumps(serialized))


def cmd_upload(file_paths):
    """Run the pipeline on uploaded files."""
    dp = DebugPrioritizer()
    results = dp.run(file_paths)

    try:
        rca = dp.rca_result or {}
        full_results = dict(results)
        full_results["root_cause_details"] = rca.get("root_cause_details", [])
        full_results["top_chains"] = rca.get("top_chains", [])
        full_results["adjacency"] = rca.get("adjacency", [])
        full_results["rca_result"] = rca
        full_results["gemini_insights"] = generate_batch_insights(full_results)
    except Exception:
        full_results = dict(results)
        full_results["root_cause_details"] = (dp.rca_result or {}).get("root_cause_details", [])
        full_results["top_chains"] = (dp.rca_result or {}).get("top_chains", [])
        full_results["adjacency"] = (dp.rca_result or {}).get("adjacency", [])
        full_results["rca_result"] = dp.rca_result or {}
        full_results["gemini_insights"] = {}

    serialized = serialize_results(full_results)
    print(json.dumps(serialized))


def cmd_suggest_fix(signature, context=""):
    """Get an AI-powered fix suggestion."""
    result = suggest_fix(signature, context)
    print(json.dumps(result))


def cmd_mark_fixed(signature, run_id="ui_run"):
    """Mark a bug signature as fixed."""
    pipeline_mark_fixed(signature, run_id)
    print(json.dumps({"status": "ok"}))


def cmd_fix_state():
    """Get the current fix state."""
    state = load_fix_state()
    print(json.dumps(state))


def cmd_regression_history():
    """Get the regression history."""
    history = load_regression_history()
    print(json.dumps(history, default=_np_default))


def cmd_simulate_fix(cluster_id, dag_file_path):
    """Simulate fixing a root cause and compute downstream impact."""
    # Read DAG JSON from the temp file written by server.js
    with open(dag_file_path, "r", encoding="utf-8") as f:
        dag_data = json.load(f)
    # Reconstruct a NetworkX DiGraph from the serialized nodes/edges
    dag = nx.DiGraph()
    for node in dag_data.get("nodes", []):
        dag.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in dag_data.get("edges", []):
        dag.add_edge(edge["source"], edge["target"],
                     **{k: v for k, v in edge.items() if k not in ("source", "target")})
    result = simulate_fix_impact(dag, int(cluster_id))
    print(json.dumps(result, default=_np_default))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python python_bridge.py <command> [args...]"}))
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "run-demo":
            cmd_run_demo()
        elif command == "upload":
            cmd_upload(sys.argv[2:])
        elif command == "suggest-fix":
            sig = sys.argv[2] if len(sys.argv) > 2 else ""
            ctx = sys.argv[3] if len(sys.argv) > 3 else ""
            cmd_suggest_fix(sig, ctx)
        elif command == "mark-fixed":
            sig = sys.argv[2] if len(sys.argv) > 2 else ""
            rid = sys.argv[3] if len(sys.argv) > 3 else "ui_run"
            cmd_mark_fixed(sig, rid)
        elif command == "fix-state":
            cmd_fix_state()
        elif command == "regression-history":
            cmd_regression_history()
        elif command == "simulate-fix":
            cid = sys.argv[2] if len(sys.argv) > 2 else "0"
            dag_json = sys.argv[3] if len(sys.argv) > 3 else "{}"
            cmd_simulate_fix(cid, dag_json)
        else:
            print(json.dumps({"error": f"Unknown command: {command}"}))
            sys.exit(1)
    except Exception as e:
        import traceback
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)
