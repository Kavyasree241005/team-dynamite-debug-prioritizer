import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Zap, CheckCircle2 } from "lucide-react";
import Plot from "../PlotlyChart";
import MetricCard from "./MetricCard";
import { theme, SEV_COLORS } from "../theme";

export default function UploadTab({ results, onRunDemo, onUpload, loading, progress, error }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) onUpload(acceptedFiles);
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/plain": [".txt", ".log"] },
    maxSize: 200 * 1024 * 1024,
  });

  // KPIs from results
  const df = results?.df || [];
  const tagSum = results?.tag_summary || {};
  const nClusters = df.length > 0
    ? new Set(df.map((r) => r.cluster).filter((c) => c !== -1)).size
    : 0;
  const nRootCauses = results?.root_cause_details?.length || results?.root_causes?.length || 0;
  const noiseCount = results?.noise_analysis?.length || 0;
  const sourceFiles = df.length > 0
    ? new Set(df.map((r) => r.source_file)).size
    : 0;

  // Severity distribution
  const sevCounts = {};
  df.forEach((r) => { sevCounts[r.severity] = (sevCounts[r.severity] || 0) + 1; });
  const sevOrder = ["FATAL", "ERROR", "WARNING", "INFO"];
  const sevColors = [theme.SANDISK_RED, "#FB923C", theme.WARNING, theme.GRAY_DIM];

  // UMAP scatter data
  const validDf = df.filter((r) => r.umap_x != null && r.umap_y != null && isFinite(r.umap_x) && isFinite(r.umap_y));
  const umapData = validDf.length > 0 ? {
    x: validDf.map((r) => r.umap_x),
    y: validDf.map((r) => r.umap_y),
    severity: validDf.map((r) => r.severity || "INFO"),
    text: validDf.map((r) => (r.cleaned_text || "").slice(0, 60)),
  } : null;

  return (
    <div className="tab-content">
      <div className="upload-grid">
        <div className="upload-left">
          <h3 className="section-heading">Upload Simulation Logs</h3>
          <p className="section-desc">
            Upload one or more UVM/SVA simulation log files (.txt). The pipeline processes all 5 layers automatically.
          </p>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? "dropzone-active" : ""}`}
          >
            <input {...getInputProps()} />
            <div className="dropzone-icon">
              <Upload size={36} color={theme.GRAY_DIM} strokeWidth={1.5} />
            </div>
            <div className="dropzone-text">
              {isDragActive ? "Drop files here..." : "Drag and drop files here"}
            </div>
            <div className="dropzone-sub">Limit 200MB per file • TXT, LOG</div>
          </div>
        </div>
        <div className="upload-right">
          <h3 className="section-heading">Quick Start</h3>
          <p className="section-desc">
            Load sample UVM logs to see the full 6-layer pipeline in action.
          </p>
          <button
            className="btn-primary btn-demo"
            onClick={onRunDemo}
            disabled={loading}
          >
            {loading ? (
              <span className="btn-loading">
                <span className="spinner" />
                Running Pipeline...
              </span>
            ) : (
              <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                <Zap size={18} /> Run Demo Mode
              </span>
            )}
          </button>
          {results && (
            <div className="demo-success">
              <CheckCircle2 size={14} style={{ verticalAlign: "middle", marginRight: 6 }} />
              Pipeline complete — {df.length} lines processed, {nClusters} clusters found
            </div>
          )}
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="error-banner">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Progress bar */}
      {loading && (
        <div className="progress-container">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="progress-text">
            Processing pipeline layers... {Math.round(progress)}%
          </div>
        </div>
      )}

      {/* KPI metrics */}
      {results && (
        <>
          <div className="metrics-grid">
            <MetricCard label="Log Lines" value={df.length.toLocaleString()} color={theme.PSG_LIGHT} />
            <MetricCard label="Source Files" value={sourceFiles} color={theme.WHITE} />
            <MetricCard label="Clusters" value={nClusters} color={theme.SUCCESS} />
            <MetricCard label="Root Causes" value={nRootCauses} color={theme.SANDISK_RED} />
            <MetricCard label="FATAL Tags" value={tagSum.tag_fatal || 0} color={theme.SANDISK_RED} />
            <MetricCard label="Noise Points" value={noiseCount} color={theme.WARNING} />
          </div>

          {/* UMAP Scatter */}
          {umapData && (
            <div className="chart-section">
              <h4 className="section-heading">UMAP Cluster Map</h4>
              <p className="section-desc">Each point is a log line, positioned by semantic similarity. Colors = severity level.</p>
              <Plot
                data={sevOrder.map((sev) => {
                  const idxs = [];
                  umapData.severity.forEach((s, i) => { if (s === sev) idxs.push(i); });
                  return {
                    x: idxs.map((i) => umapData.x[i]),
                    y: idxs.map((i) => umapData.y[i]),
                    text: idxs.map((i) => umapData.text[i]),
                    type: "scattergl",
                    mode: "markers",
                    name: sev,
                    marker: {
                      color: SEV_COLORS[sev] || theme.GRAY,
                      size: 5,
                      opacity: 0.8,
                      line: { width: 0 },
                    },
                    hovertemplate: "<b>%{text}</b><extra></extra>",
                  };
                })}
                layout={{
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  xaxis: { visible: false },
                  yaxis: { visible: false },
                  height: 450,
                  margin: { l: 0, r: 0, t: 10, b: 0 },
                  legend: {
                    orientation: "h",
                    yanchor: "bottom",
                    y: -0.12,
                    font: { color: theme.GRAY, size: 11 },
                    bgcolor: "rgba(0,0,0,0)",
                  },
                  font: { color: theme.WHITE },
                }}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {/* Severity Distribution */}
          <div className="chart-section">
            <h4 className="section-heading">Severity Distribution</h4>
            <Plot
              data={[{
                x: sevOrder,
                y: sevOrder.map((s) => sevCounts[s] || 0),
                type: "bar",
                marker: {
                  color: sevColors,
                  line: { width: 0 },
                },
                text: sevOrder.map((s) => String(sevCounts[s] || 0)),
                textposition: "outside",
                textfont: { color: theme.WHITE, size: 12 },
              }]}
              layout={{
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                height: 280,
                margin: { l: 40, r: 20, t: 10, b: 40 },
                xaxis: { color: theme.GRAY, tickfont: { color: theme.GRAY } },
                yaxis: { color: theme.GRAY, gridcolor: "#1E293B", tickfont: { color: theme.GRAY_DIM } },
                font: { color: theme.WHITE },
                bargap: 0.3,
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Gemini Insight */}
          {results.gemini_insights?.dna_insight && (
            <div className="gemini-card">
              <div className="gemini-header">
                <span className="gemini-icon">✨</span>
                <span className="gemini-title">Gemini Insight: Run Profile Analysis</span>
              </div>
              <div className="gemini-text">"{results.gemini_insights.dna_insight}"</div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
