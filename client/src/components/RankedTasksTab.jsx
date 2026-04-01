import React, { useState } from "react";
import { Lightbulb, CheckCircle, ArrowUpRight, ArrowDownRight, ArrowRight, Sparkles } from "lucide-react";
import DnaBar from "./DnaBar";
import { theme, SEV_COLORS } from "../theme";
import { suggestFix, markAsFixed } from "../api";

function estimateBugCost(frequency, isFatal, isRoot) {
  let base = frequency * 4 * 120;
  let risk = 0;
  if (isFatal && isRoot) risk = 2500000 * 0.05;
  else if (isFatal) risk = 2500000 * 0.01;
  return base + risk;
}

export default function RankedTasksTab({ results }) {
  const [fixes, setFixes] = useState({});
  const [showFix, setShowFix] = useState({});
  const [fixLoading, setFixLoading] = useState({});
  const [fixedItems, setFixedItems] = useState({});

  if (!results) {
    return <div className="tab-empty">Run the pipeline first from the Upload &amp; Run tab.</div>;
  }

  const xaiResults = results.xai_results || [];
  const history = results.regression_history || {};

  const getTrend = (signature) => {
    const key = signature?.slice(0, 80);
    const entries = history[key] || [];
    if (entries.length < 2) return { trend: "new", delta: 0 };
    const scores = entries.map((e) => e.score);
    const delta = scores[scores.length - 1] - scores[scores.length - 2];
    if (delta > 0.001) return { trend: "worsening", delta };
    if (delta < -0.001) return { trend: "improving", delta };
    return { trend: "stable", delta };
  };

  const handleSuggestFix = async (cid, sig) => {
    const isShowing = showFix[cid];
    setShowFix((prev) => ({ ...prev, [cid]: !isShowing }));
    if (!isShowing && !fixes[cid]) {
      setFixLoading((prev) => ({ ...prev, [cid]: true }));
      try {
        const data = await suggestFix(sig);
        setFixes((prev) => ({ ...prev, [cid]: data }));
      } catch {
        setFixes((prev) => ({ ...prev, [cid]: { fix: "Failed to get suggestion. Check API connectivity.", confidence: 0 } }));
      } finally {
        setFixLoading((prev) => ({ ...prev, [cid]: false }));
      }
    }
  };

  const handleMarkFixed = async (cid, sig) => {
    await markAsFixed(sig);
    setFixedItems((prev) => ({ ...prev, [cid]: true }));
  };

  const TrendIcon = ({ trend }) => {
    if (trend === "worsening") return <ArrowUpRight size={12} />;
    if (trend === "improving") return <ArrowDownRight size={12} />;
    if (trend === "stable") return <ArrowRight size={12} />;
    return <Sparkles size={12} />;
  };

  const trendLabels = { worsening: "Worsening", improving: "Improving", stable: "Stable", new: "NEW" };

  return (
    <div className="tab-content">
      <h3 className="section-heading">Priority-Ranked Failure Clusters</h3>
      <p className="section-desc">
        Clusters ranked by composite score: frequency × severity × recency × root-cause bonus × impact bonus.
      </p>

      {xaiResults.map((rec) => {
        const {
          rank, cluster_id: cid, priority_score: score, severity: sev,
          frequency: freq, root_cause: isRoot, dna_fingerprint: dna,
          signature: sig, suggested_owner: owner, blame_file, blame_line,
          dag_depth, dag_impact,
        } = rec;

        const trendData = getTrend(sig);
        const borderColor = isRoot ? theme.SANDISK_RED : theme.PSG_LIGHT;
        const sevColor = SEV_COLORS[sev] || theme.GRAY;
        const cost = estimateBugCost(freq, sev === "FATAL", isRoot);

        return (
          <div key={cid} className="ranked-card" style={{ borderLeftColor: borderColor }}>
            <div className="ranked-card-header">
              <div className="ranked-left">
                <div className="rank-circle" style={{ borderColor, color: borderColor }}>
                  #{rank}
                </div>
                <div>
                  <div className="ranked-title">
                    Cluster {cid}
                    <span className="sev-badge" style={{ color: sevColor }}>{sev}</span>
                    <span
                      className="role-badge"
                      style={{ background: isRoot ? theme.SANDISK_RED : theme.PSG_BLUE }}
                    >
                      {isRoot ? "ROOT CAUSE" : "CASCADING"}
                    </span>
                    {owner && owner !== "Unassigned" && (
                      <span className="owner-badge">{owner}</span>
                    )}
                    <span className={`trend-badge trend-${trendData.trend}`}>
                      <TrendIcon trend={trendData.trend} /> {trendLabels[trendData.trend]}
                    </span>
                  </div>
                  <div className="ranked-sig">{sig?.slice(0, 100)}</div>
                  {blame_file && (
                    <div className="blame-ref">{blame_file}:{blame_line}</div>
                  )}
                </div>
              </div>
              <div className="ranked-right">
                <div className="ranked-score" style={{ color: borderColor }}>
                  {score?.toFixed(4)}
                </div>
                <div className="ranked-meta">
                  {freq} lines | depth {dag_depth || 0} | impact {dag_impact || 0}
                </div>
                <div className="cost-rollup">
                  <span className="cost-value">${cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                  <span className="cost-label">Est. Cost Impact</span>
                </div>
              </div>
            </div>

            <DnaBar dna={dna} />

            <div className="ranked-actions">
              <button
                className="btn-secondary"
                onClick={() => handleSuggestFix(cid, sig)}
                disabled={fixLoading[cid]}
                style={{
                  borderColor: showFix[cid] ? "rgba(16,185,129,0.5)" : undefined,
                  color: showFix[cid] ? theme.SUCCESS : undefined,
                }}
              >
                <Lightbulb size={14} style={{ verticalAlign: "middle", marginRight: 4 }} />
                {fixLoading[cid] ? "Loading..." : "Suggest Fix"}
              </button>
              {!fixedItems[cid] ? (
                <button className="btn-secondary" onClick={() => handleMarkFixed(cid, sig)}>
                  <CheckCircle size={14} style={{ verticalAlign: "middle", marginRight: 4 }} />
                  Mark as Fixed
                </button>
              ) : (
                <button className="btn-secondary" disabled>Fix Pending Verification</button>
              )}
            </div>

            {showFix[cid] && fixes[cid] && (
              <div className="fix-suggestion-card">
                <div className="fix-suggestion-header">
                  Agentic Suggestion ({fixes[cid].confidence}% confidence):
                </div>
                <code className="fix-suggestion-code">{fixes[cid].fix}</code>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
