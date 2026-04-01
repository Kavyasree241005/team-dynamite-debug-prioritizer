import React from "react";
import { ArrowUpRight, ArrowDownRight, ArrowRight } from "lucide-react";
import Plot from "../PlotlyChart";
import { theme } from "../theme";

export default function TimelineTab({ results }) {
  if (!results) {
    return <div className="tab-empty">Run the pipeline first from the Upload &amp; Run tab.</div>;
  }

  const xaiResults = results.xai_results || [];
  const history = results.regression_history || {};

  const getTrend = (signature) => {
    const key = signature?.slice(0, 80);
    const entries = history[key] || [];
    if (entries.length < 2) return { trend: "new", delta: 0, scores: entries.map((e) => e.score), runIds: entries.map((e) => e.run_id) };
    const scores = entries.slice(-8).map((e) => e.score);
    const runIds = entries.slice(-8).map((e) => e.run_id);
    const delta = scores[scores.length - 1] - scores[scores.length - 2];
    let trend = "stable";
    if (delta > 0.001) trend = "worsening";
    else if (delta < -0.001) trend = "improving";
    return { trend, delta, scores, runIds };
  };

  const anyHistory = Object.keys(history).length > 0;

  if (!anyHistory) {
    return (
      <div className="tab-content">
        <h3 className="section-heading">Regression Trend Timeline</h3>
        <div className="tab-empty">No previous runs recorded yet. Run the pipeline multiple times to see trend data.</div>
      </div>
    );
  }

  const trendColors = {
    worsening: theme.SANDISK_RED,
    improving: theme.SUCCESS,
    stable: theme.GRAY,
    new: theme.GRAY_DIM,
  };

  const TrendIcon = ({ trend }) => {
    if (trend === "worsening") return <ArrowUpRight size={14} />;
    if (trend === "improving") return <ArrowDownRight size={14} />;
    return <ArrowRight size={14} />;
  };

  const trendLabels = {
    worsening: "Worsening",
    improving: "Improving",
    stable: "Stable",
    new: "New",
  };

  return (
    <div className="tab-content">
      <h3 className="section-heading">Regression Trend Timeline</h3>
      <p className="section-desc">
        Track how cluster priority scores evolve across pipeline runs. Worsening trends are highlighted in red.
      </p>

      {xaiResults.map((rec) => {
        const cid = rec.cluster_id;
        const sig = rec.signature?.slice(0, 80);
        const trendData = getTrend(rec.signature);
        const { trend, delta, scores, runIds } = trendData;
        const tColor = trendColors[trend];

        if (scores.length === 0) return null;

        const hexToRgb = (hex) => {
          const r = parseInt(hex.slice(1, 3), 16);
          const g = parseInt(hex.slice(3, 5), 16);
          const b = parseInt(hex.slice(5, 7), 16);
          return `${r},${g},${b}`;
        };

        return (
          <div key={cid} className="timeline-card">
            <div className="timeline-header">
              <div>
                <span className="timeline-cluster">Cluster {cid}</span>
                <span className="timeline-meta">
                  {rec.severity} | {scores.length} run(s) recorded
                </span>
              </div>
              <div>
                <span className="timeline-trend" style={{ color: tColor, display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <TrendIcon trend={trend} /> {trendLabels[trend]}
                </span>
                <span className="timeline-delta">
                  delta: {delta > 0 ? "+" : ""}{delta.toFixed(6)}
                </span>
              </div>
            </div>
            <div className="timeline-sig">{sig}</div>

            {scores.length >= 2 && (
              <Plot
                data={[{
                  x: runIds,
                  y: scores,
                  mode: "lines+markers",
                  line: { color: tColor, width: 2.5 },
                  marker: { size: 6, color: tColor, line: { width: 1, color: theme.WHITE } },
                  fill: "tozeroy",
                  fillcolor: `rgba(${hexToRgb(tColor)},0.08)`,
                  hovertemplate: "%{x}<br>Score: %{y:.6f}<extra></extra>",
                  type: "scatter",
                }]}
                layout={{
                  height: 140,
                  margin: { l: 40, r: 20, t: 10, b: 30 },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  xaxis: { color: theme.GRAY, gridcolor: "#1E293B", tickfont: { size: 9, color: theme.GRAY_DIM } },
                  yaxis: { color: theme.GRAY, gridcolor: "#1E293B", tickfont: { size: 9, color: theme.GRAY_DIM }, tickformat: ".6f" },
                  font: { color: theme.WHITE },
                  showlegend: false,
                }}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
