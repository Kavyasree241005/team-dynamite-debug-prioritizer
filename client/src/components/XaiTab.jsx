import React from "react";
import { Brain } from "lucide-react";
import Plot from "../PlotlyChart";
import DnaBar from "./DnaBar";
import { theme, DNA_COLORS } from "../theme";

export default function XaiTab({ results }) {
  if (!results) {
    return <div className="tab-empty">Run the pipeline first from the Upload &amp; Run tab.</div>;
  }

  const xaiResults = results.xai_results || [];
  const memoryInsights = results.memory_insights || {};

  const uvm = ["UVM_FATAL", "UVM_ERROR", "SVA_FAIL", "UVM_WARNING"];
  const dnaLabels = ["FATAL", "ERROR", "SVA", "WARNING"];

  return (
    <div className="tab-content">
      <h3 className="section-heading">Explainable AI Analysis</h3>
      <p className="section-desc">
        Each cluster receives a full explainability report: mathematical score derivation, root-cause classification, and Failure DNA Fingerprint interpretation.
      </p>

      {/* DNA Legend */}
      <div className="dna-legend">
        {dnaLabels.map((lbl) => (
          <div key={lbl} className="dna-legend-item">
            <div className="dna-legend-dot" style={{ background: DNA_COLORS[lbl] }} />
            <span>{lbl}</span>
          </div>
        ))}
      </div>

      {xaiResults.map((rec) => {
        const {
          rank, cluster_id: cid, priority_score: score, root_cause: isRoot,
          severity: sev, dna_fingerprint: dna = [0, 0, 0, 0], frequency: freq,
          frequency_ratio: freqR, severity_weight: sevW, recency_factor: recency,
          root_bonus: rootB, impact_bonus: impactB, dag_depth: dagDepth,
          dag_impact: dagImpact, signature: sig, source_files: srcFiles = [],
          tag_counts = {},
        } = rec;

        const roleLabel = isRoot ? "ROOT CAUSE" : "CASCADING SYMPTOM";
        const roleColor = isRoot ? theme.SANDISK_RED : theme.PSG_LIGHT;

        const dominantIdx = dna.reduce((max, v, i, arr) => v > arr[max] ? i : max, 0);
        let dnaText = "INFO-only cluster with no severity tags.";
        if (dna.some((v) => v > 0)) {
          const pct = (dna[dominantIdx] * 100).toFixed(0);
          const uvmLabel = uvm[dominantIdx];
          const contexts = [
            "→ indicates critical hardware/system failure requiring immediate escalation.",
            "→ indicates functional mismatches or protocol violations in the design.",
            "→ driven by formal assertion failures, high-confidence RTL bug indicators.",
            "→ advisory-level issues that may escalate under stress conditions.",
          ];
          dnaText = `This cluster is ${pct}% ${uvmLabel} ${contexts[dominantIdx]}`;
        }

        const mem = memoryInsights[String(cid)];

        return (
          <div key={cid} className={`xai-card ${isRoot ? "root-cause" : "cascading"}`}>
            {/* Header */}
            <div className="xai-header">
              <div>
                <div className="xai-title">Rank #{rank} — Cluster {cid}</div>
                <div className="xai-sig">{sig?.slice(0, 120)}</div>
              </div>
              <div className="xai-right">
                <span className="role-badge" style={{ background: roleColor }}>{roleLabel}</span>
                <div className="xai-score" style={{ color: roleColor }}>{score?.toFixed(6)}</div>
              </div>
            </div>

            {/* Score Derivation */}
            <div className="score-derivation">
              <div className="derivation-title">Score Derivation</div>
              <div>P = freq_ratio × sev_weight × recency × (1 + root_bonus) × (1 + impact_bonus)</div>
              <div>P = {freqR?.toFixed(4)} × {sevW?.toFixed(2)} × {recency?.toFixed(4)} × (1 + {rootB?.toFixed(1)}) × (1 + {impactB?.toFixed(4)})</div>
              <div>P = <strong style={{ color: theme.WHITE }}>{score?.toFixed(6)}</strong></div>
              <br />
              <div>freq_ratio &nbsp;&nbsp;= {freq}/total = {freqR?.toFixed(4)}</div>
              <div>sev_weight &nbsp;&nbsp;= SEVERITY_WEIGHT[{sev}] = {sevW?.toFixed(2)}</div>
              <div>recency &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {recency?.toFixed(4)}</div>
              <div>root_bonus &nbsp;&nbsp;= {rootB?.toFixed(1)} {isRoot ? "(ROOT CAUSE)" : "(cascading)"}</div>
              <div>impact_bonus = 0.1 × {dagImpact} downstream = {impactB?.toFixed(4)}</div>
            </div>

            {/* DNA Fingerprint */}
            <div className="xai-dna-section">
              <div className="derivation-title" style={{ marginBottom: 8 }}>Failure DNA Fingerprint</div>
              <div className="dna-values">
                {dnaLabels.map((lbl, i) => (
                  <span key={lbl} style={{ color: DNA_COLORS[lbl] }}>
                    {lbl}: {(dna[i] * 100).toFixed(1)}%
                  </span>
                ))}
              </div>
              <div className="dna-interp">{dnaText}</div>
            </div>

            {/* Metadata footer */}
            <div className="xai-footer">
              <span>Depth: {dagDepth}</span>
              <span>Impact: {dagImpact} downstream</span>
              <span>Tags: F={tag_counts.FATAL || 0} E={tag_counts.ERROR || 0} S={tag_counts.SVA || 0} W={tag_counts.WARNING || 0}</span>
              <span>Files: {srcFiles.join(", ")}</span>
            </div>

            {/* Memory Match */}
            {mem && (
              <div className="memory-match">
                <div className="memory-header">
                  <span className="memory-badge">
                    <Brain size={12} style={{ verticalAlign: "middle", marginRight: 4 }} />
                    MEMORY MATCH
                  </span>
                  <span className="memory-sim">({(mem.similarity * 100).toFixed(1)}% Match)</span>
                </div>
                <div className="memory-text">
                  Seen before in <b>{mem.project_name}</b> — previously fixed by: <i>{mem.previous_fix_note}</i>
                </div>
              </div>
            )}

            <DnaBar dna={dna} height={8} />
          </div>
        );
      })}
    </div>
  );
}
