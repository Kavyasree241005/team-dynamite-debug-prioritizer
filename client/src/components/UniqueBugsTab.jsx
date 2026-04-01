import React from "react";
import { theme, SEV_COLORS } from "../theme";

export default function UniqueBugsTab({ results }) {
  if (!results) {
    return <div className="tab-empty">Run the pipeline first from the Upload &amp; Run tab.</div>;
  }

  const noiseData = results.noise_analysis || [];

  const nHard = noiseData.filter((n) => n.noise_class === "Hard Noise").length;
  const nMed = noiseData.filter((n) => n.noise_class === "Medium Noise").length;
  const nSoft = noiseData.filter((n) => n.noise_class === "Soft Noise").length;

  const classColors = {
    "Hard Noise": { bg: theme.SANDISK_RED, fg: "white", bar: theme.SANDISK_RED },
    "Medium Noise": { bg: "#FBBF24", fg: "#0F172A", bar: "#FBBF24" },
    "Soft Noise": { bg: theme.SUCCESS, fg: "white", bar: theme.SUCCESS },
  };

  return (
    <div className="tab-content">
      <h3 className="section-heading">Unique Bugs (Noise Confidence Scoring)</h3>
      <p className="section-desc">
        Log lines that HDBSCAN classified as noise (-1 cluster). High novelty = potentially new, undiscovered bug class.
      </p>

      {noiseData.length === 0 ? (
        <div className="tab-empty">No noise points detected. All log lines were assigned to clusters.</div>
      ) : (
        <>
          <div className="metrics-grid metrics-grid-4">
            <div className="metric-card">
              <div className="metric-label">Total Noise Points</div>
              <div className="metric-value" style={{ color: theme.WHITE }}>{noiseData.length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Hard Noise</div>
              <div className="metric-value" style={{ color: theme.SANDISK_RED }}>{nHard}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Medium Noise</div>
              <div className="metric-value" style={{ color: "#FBBF24" }}>{nMed}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Soft Noise</div>
              <div className="metric-value" style={{ color: theme.SUCCESS }}>{nSoft}</div>
            </div>
          </div>

          {noiseData.length > 100 && (
            <div className="info-bar" style={{ marginTop: 16, marginBottom: 8 }}>
              Showing top 100 out of {noiseData.length} noise points for performance.
            </div>
          )}

          {noiseData.slice(0, 100).map((nd, i) => {
            const cls = classColors[nd.noise_class] || classColors["Soft Noise"];
            const sevColor = SEV_COLORS[nd.severity] || theme.GRAY;

            return (
              <div key={i} className="noise-card" style={{ borderLeftColor: cls.bar }}>
                <div className="noise-card-inner">
                  <div className="noise-left">
                    <div className="noise-badges">
                      <span className="noise-class-badge" style={{ background: cls.bg, color: cls.fg }}>{nd.noise_class}</span>
                      <span className="noise-sev" style={{ color: sevColor }}>{nd.severity}</span>
                      <span className="noise-nearest">nearest: Cluster {nd.nearest_cluster}</span>
                    </div>
                    <div className="noise-sig">{nd.signature}</div>
                  </div>
                  <div className="noise-right">
                    <div className="novelty-score" style={{ color: cls.bar }}>{nd.novelty_score?.toFixed(1)}</div>
                    <div className="novelty-label">Novelty Score</div>
                    <div className="novelty-bar-bg">
                      <div className="novelty-bar-fill" style={{ width: `${nd.novelty_score}%`, background: cls.bar }} />
                    </div>
                    <div className="novelty-meta">
                      dist: {nd.centroid_dist?.toFixed(4)} | outlier: {nd.outlier_score?.toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
