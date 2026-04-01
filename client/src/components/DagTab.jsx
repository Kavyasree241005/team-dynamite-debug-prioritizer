import React, { useState } from "react";
import { Lightbulb, Zap } from "lucide-react";
import Plot from "../PlotlyChart";
import { theme } from "../theme";
import { suggestFix, simulateFix } from "../api";

export default function DagTab({ results }) {
  const [showFix, setShowFix] = useState({});
  const [fixes, setFixes] = useState({});
  const [showSim, setShowSim] = useState({});
  const [simData, setSimData] = useState({});
  const [simLoading, setSimLoading] = useState({});
  const [showAll, setShowAll] = useState(false);

  if (!results) {
    return <div className="tab-empty">Run the pipeline first from the Upload &amp; Run tab.</div>;
  }

  const dag = results.dag || { nodes: [], edges: [] };
  const rootCauses = results.root_causes || [];
  const rootCauseDetails = results.root_cause_details || [];
  const topChains = results.top_chains || [];
  const adjacency = results.adjacency || [];
  const rootSet = new Set(rootCauses);

  const buildDagFigure = () => {
    if (dag.nodes.length === 0) return null;

    const nodes = dag.nodes;
    const edges = dag.edges;
    const positions = {};
    const n = nodes.length;
    nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / n;
      const r = 2;
      positions[node.id] = { x: r * Math.cos(angle), y: r * Math.sin(angle) };
    });

    const edgeX = [], edgeY = [];
    edges.forEach(({ source, target }) => {
      const s = positions[source], t = positions[target];
      if (s && t) {
        edgeX.push(s.x, t.x, null);
        edgeY.push(s.y, t.y, null);
      }
    });

    const edgeTrace = {
      x: edgeX, y: edgeY,
      mode: "lines",
      line: { width: 1.5, color: theme.GRAY_DIM },
      hoverinfo: "none",
      type: "scatter",
    };

    const nodeX = nodes.map((n) => positions[n.id]?.x);
    const nodeY = nodes.map((n) => positions[n.id]?.y);
    const nodeColors = nodes.map((n) => rootSet.has(n.id) ? theme.SANDISK_RED : theme.PSG_LIGHT);
    const nodeSizes = nodes.map((n) => Math.max(16, Math.min(32, (n.count || 5) + 10)));
    const nodeTexts = nodes.map((n) => `C${n.id}`);
    const hoverTexts = nodes.map((n) =>
      `<b>Cluster ${n.id}</b><br>` +
      `Severity: ${n.dominant_severity || "INFO"}<br>` +
      `Lines: ${n.count || 0}<br>` +
      `Depth: ${n.depth || 0}<br>` +
      `Impact: ${n.impact || 0} downstream<br>` +
      `${rootSet.has(n.id) ? "ROOT CAUSE" : "Cascading symptom"}`
    );

    const nodeTrace = {
      x: nodeX, y: nodeY,
      mode: "markers+text",
      text: nodeTexts,
      textposition: "top center",
      textfont: { size: 11, color: theme.WHITE, family: "Inter" },
      hovertext: hoverTexts,
      hoverinfo: "text",
      marker: {
        size: nodeSizes,
        color: nodeColors,
        line: { width: 2, color: "#0F172A" },
      },
      type: "scatter",
    };

    const annotations = edges.map(({ source, target }) => {
      const s = positions[source], t = positions[target];
      if (!s || !t) return null;
      return {
        ax: s.x, ay: s.y, x: t.x, y: t.y,
        xref: "x", yref: "y", axref: "x", ayref: "y",
        showarrow: true, arrowhead: 3, arrowsize: 1.5, arrowwidth: 1.5,
        arrowcolor: theme.GRAY_DIM,
      };
    }).filter(Boolean);

    return { data: [edgeTrace, nodeTrace], annotations };
  };

  const dagFig = buildDagFigure();

  const [fixLoading, setFixLoading] = useState({});

  const handleSuggestFix = async (cid, sig, context = "") => {
    const isShowing = showFix[cid];
    setShowFix((prev) => ({ ...prev, [cid]: !isShowing }));
    if (!isShowing && !fixes[cid]) {
      setFixLoading((prev) => ({ ...prev, [cid]: true }));
      try {
        const data = await suggestFix(sig, context);
        setFixes((prev) => ({ ...prev, [cid]: data }));
      } catch {
        setFixes((prev) => ({ ...prev, [cid]: { fix: "Failed to get suggestion. Check API connectivity.", confidence: 0 } }));
      } finally {
        setFixLoading((prev) => ({ ...prev, [cid]: false }));
      }
    }
  };

  const handleSimulateFix = async (cid) => {
    const isShowing = showSim[cid];
    setShowSim((prev) => ({ ...prev, [cid]: !isShowing }));
    if (!isShowing && !simData[cid]) {
      setSimLoading((prev) => ({ ...prev, [cid]: true }));
      try {
        const data = await simulateFix(cid, dag);
        setSimData((prev) => ({ ...prev, [cid]: data }));
      } catch {
        setSimData((prev) => ({ ...prev, [cid]: { before: 0, after: 0, reduction_pct: 0, downstream_nodes: 0 } }));
      } finally {
        setSimLoading((prev) => ({ ...prev, [cid]: false }));
      }
    }
  };

  return (
    <div className="tab-content">
      <h3 className="section-heading">Cluster-Level Causal Graph Analysis</h3>
      <p className="section-desc">
        Analyzing temporal transitions between failure clusters to identify true root causes and cascading chains.
      </p>

      {/* Root Cause Summary */}
      <h4 className="section-heading" style={{ marginTop: 10 }}>Top Root Causes</h4>
      {rootCauseDetails.length === 0 ? (
        <div className="info-bar">No definitive root causes found.</div>
      ) : (
        rootCauseDetails.map((rc) => (
          <div key={rc.cluster_id}>
            <div className="rc-card">
              <div className="rc-header">
                <div>
                  <span className="rc-title">Cluster {rc.cluster_id}</span>
                  <span className="role-badge" style={{ background: theme.SANDISK_RED, marginLeft: 12 }}>
                    ROOT CAUSE
                  </span>
                </div>
                <div className="rc-score">Score: {rc.score?.toFixed(3)}</div>
              </div>
              <div className="rc-sig">{rc.signature}</div>
              <div className="rc-stats">{rc.explanation}</div>
              <div className="rc-reason">{rc.human_reason}</div>
              <div className="ranked-actions" style={{ marginTop: 12 }}>
                <button
                  className="btn-secondary"
                  onClick={() => handleSuggestFix(rc.cluster_id, rc.signature, rc.human_reason || "")}
                  disabled={fixLoading[rc.cluster_id]}
                  style={{
                    borderColor: showFix[rc.cluster_id] ? "rgba(16,185,129,0.5)" : undefined,
                    color: showFix[rc.cluster_id] ? theme.SUCCESS : undefined,
                  }}
                >
                  <Lightbulb size={14} style={{ verticalAlign: "middle", marginRight: 4 }} />
                  {fixLoading[rc.cluster_id] ? "Loading..." : "Suggest Fix"}
                </button>
                <button
                  className="btn-secondary"
                  onClick={() => handleSimulateFix(rc.cluster_id)}
                  disabled={simLoading[rc.cluster_id]}
                  style={{
                    borderColor: showSim[rc.cluster_id] ? "rgba(16,185,129,0.5)" : undefined,
                    color: showSim[rc.cluster_id] ? theme.SUCCESS : undefined,
                  }}
                >
                  <Zap size={14} style={{ verticalAlign: "middle", marginRight: 4 }} />
                  {simLoading[rc.cluster_id] ? "Simulating..." : "Simulate Fix"}
                </button>
              </div>
            </div>

            {/* Fix Suggestion */}
            {showFix[rc.cluster_id] && fixes[rc.cluster_id] && (
              <div className="fix-suggestion-card">
                <div className="fix-suggestion-header">
                  Agentic Suggestion ({fixes[rc.cluster_id].confidence}% confidence):
                </div>
                <code className="fix-suggestion-code">{fixes[rc.cluster_id].fix}</code>
              </div>
            )}

            {/* Simulate Fix Results */}
            {showSim[rc.cluster_id] && simData[rc.cluster_id] && (
              <div style={{ animation: "fadeInUp 0.4s ease-out" }}>
                <div className="sim-impact-card">
                  <div className="sim-impact-header">
                    <Zap size={18} style={{ color: theme.SUCCESS }} />
                    <span>What-If Impact Simulation Complete</span>
                  </div>
                  <div className="sim-impact-body">
                    Fixing this root cause eliminates{" "}
                    <strong style={{ color: theme.WARNING, fontSize: "1.05rem" }}>
                      {simData[rc.cluster_id].downstream_nodes}
                    </strong>{" "}
                    downstream cascading transitions. Projected structural reduction in global volume:{" "}
                    <strong style={{ color: theme.SUCCESS, fontSize: "1.05rem" }}>
                      {simData[rc.cluster_id].reduction_pct}%
                    </strong>!
                  </div>
                </div>

                <Plot
                  data={[
                    {
                      name: "Current Impact",
                      x: ["Impact Volume"],
                      y: [simData[rc.cluster_id].before],
                      type: "bar",
                      marker: { color: theme.SANDISK_RED, line: { width: 0 } },
                    },
                    {
                      name: "Expected After Fix",
                      x: ["Impact Volume"],
                      y: [simData[rc.cluster_id].after],
                      type: "bar",
                      marker: { color: theme.SUCCESS, line: { width: 0 } },
                    },
                  ]}
                  layout={{
                    barmode: "group",
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    font: { color: theme.WHITE },
                    height: 220,
                    margin: { l: 40, r: 20, t: 30, b: 20 },
                    title: {
                      text: "Simulated Target Reduction",
                      font: { size: 14, color: theme.GRAY },
                    },
                    xaxis: { showgrid: false, color: theme.GRAY },
                    yaxis: { gridcolor: "#1E293B", color: theme.GRAY_DIM },
                    showlegend: true,
                    legend: {
                      orientation: "h",
                      yanchor: "bottom",
                      y: 1.02,
                      xanchor: "right",
                      x: 1,
                      font: { color: theme.GRAY, size: 11 },
                    },
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: "100%" }}
                />
              </div>
            )}
          </div>
        ))
      )}

      {/* Gemini Insight */}
      {results.gemini_insights?.dag_insight && (
        <div className="gemini-card" style={{ marginTop: 20 }}>
          <div className="gemini-header">
            <span className="gemini-icon">✨</span>
            <span className="gemini-title">Gemini Insight: Structural Causal Analysis</span>
          </div>
          <div className="gemini-text">"{results.gemini_insights.dag_insight}"</div>
        </div>
      )}

      {/* DAG Visualization */}
      {dagFig && (
        <Plot
          data={dagFig.data}
          layout={{
            showlegend: false,
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            xaxis: { visible: false },
            yaxis: { visible: false },
            height: 500,
            margin: { l: 20, r: 20, t: 20, b: 20 },
            annotations: dagFig.annotations,
            font: { color: theme.WHITE },
          }}
          config={{ displayModeBar: false }}
          style={{ width: "100%" }}
        />
      )}

      {/* Top Failure Chains */}
      <h4 className="section-heading" style={{ marginTop: 30 }}>Top Failure Chains</h4>
      {topChains.length === 0 ? (
        <div className="info-bar">No prominent failure chains identified.</div>
      ) : (
        topChains.map((chain, i) => (
          <div key={i} className="chain-card">
            <span className="chain-label">Chain {i + 1}:</span>
            <span className="chain-text" dangerouslySetInnerHTML={{
              __html: chain.readable?.replace(/ → /g,
                ` <strong style="color:${theme.SANDISK_RED};font-size:1.1rem;padding:0 6px">→</strong> `)
            }} />
            <div className="chain-weight">Total Weight: {chain.total_weight?.toFixed(3)}</div>
          </div>
        ))
      )}

      {/* Causal Transition Table */}
      <h4 className="section-heading" style={{ marginTop: 30 }}>Causal Transition Table</h4>
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={showAll}
          onChange={(e) => setShowAll(e.target.checked)}
        />
        Show All Transitions (include &lt; 5% probability)
      </label>
      {adjacency.length > 0 ? (
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>From Cluster</th>
                <th>To Cluster</th>
                <th>Probability</th>
                <th>Frequency</th>
              </tr>
            </thead>
            <tbody>
              {adjacency
                .filter((a) => showAll || a.probability >= 0.05)
                .map((a, i) => (
                  <tr key={i}>
                    <td>{a.from_cluster}</td>
                    <td>{a.to_cluster}</td>
                    <td>{(a.probability * 100).toFixed(1)}%</td>
                    <td>{a.frequency}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="info-bar">No temporal transitions detected.</div>
      )}
    </div>
  );
}
