import React, { useState, useEffect } from "react";
import { AlertTriangle, CheckCircle } from "lucide-react";
import { theme } from "../theme";
import { getFixState } from "../api";

export default function FixHistoryTab({ results }) {
  const [fixState, setFixState] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getFixState()
      .then(setFixState)
      .catch(() => setFixState({}))
      .finally(() => setLoading(false));
  }, []);

  const currentSigs = new Set(
    (results?.xai_results || []).map((rec) => rec.signature?.slice(0, 80))
  );

  const entries = Object.entries(fixState).sort(
    (a, b) => (b[1].fixed_on || "").localeCompare(a[1].fixed_on || "")
  );

  return (
    <div className="tab-content">
      <h3 className="section-heading">Fix Verification History</h3>
      <p className="section-desc">
        Track bugs marked as fixed. If a fixed bug reappears in the current log run, it is marked as Fix Incomplete.
      </p>

      {results?.gemini_insights?.memory_insight && (
        <div className="gemini-card">
          <div className="gemini-header">
            <span className="gemini-icon">✨</span>
            <span className="gemini-title">Gemini Insight: Cross-Project Memory</span>
          </div>
          <div className="gemini-text">"{results.gemini_insights.memory_insight}"</div>
        </div>
      )}

      {loading ? (
        <div className="tab-empty">Loading fix history...</div>
      ) : entries.length === 0 ? (
        <div className="tab-empty">No bugs have been marked as fixed yet.</div>
      ) : (
        entries.map(([sigKey, data]) => {
          const fixedOn = (data.fixed_on || "Unknown date").slice(0, 10);
          const isRegression = currentSigs.has(sigKey);
          const statusColor = isRegression ? "rgba(244,63,94,0.3)" : "rgba(16,185,129,0.3)";
          const statusBorder = isRegression ? theme.SANDISK_RED : theme.SUCCESS;
          const statusText = isRegression ? "Fix Incomplete / Regression" : "Resolved";
          const StatusIcon = isRegression ? AlertTriangle : CheckCircle;

          return (
            <div key={sigKey} className="fix-card" style={{ borderLeftColor: statusBorder }}>
              <div className="fix-card-inner">
                <div className="fix-left">
                  <div className="fix-badges">
                    <span className="fix-status-badge" style={{ background: statusColor, color: isRegression ? "#FDA4AF" : "#6EE7B7" }}>
                      <StatusIcon size={12} style={{ verticalAlign: "middle", marginRight: 4 }} />
                      {statusText}
                    </span>
                    <span className="fix-date">Marked fixed on: {fixedOn}</span>
                  </div>
                  <div className="fix-sig">{sigKey}...</div>
                </div>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
