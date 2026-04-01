import React from "react";
import {
  Layers, Brain, FileInput, Cpu, Network, Shield,
  Fingerprint, GitBranch, TrendingUp, AudioLines, CheckCircle, Globe,
} from "lucide-react";
import { theme } from "../theme";

const pipelineLayers = [
  { icon: Globe, label: "Cross-Project Memory", color: theme.CYAN },
  { icon: FileInput, label: "Ingestion & Denoising", color: theme.GRAY },
  { icon: Brain, label: "Semantic Embedding (MiniLM)", color: theme.GRAY },
  { icon: Cpu, label: "UMAP + HDBSCAN Clustering", color: theme.GRAY },
  { icon: Network, label: "Topological RCA (DAG)", color: theme.GRAY },
  { icon: Shield, label: "Prioritization + XAI", color: theme.GRAY },
];

const features = [
  { icon: Fingerprint, label: "Failure DNA Fingerprint", color: theme.SANDISK_RED },
  { icon: GitBranch, label: "Git Blame Auto-Assignee", color: theme.PSG_LIGHT },
  { icon: TrendingUp, label: "Regression Trend Timeline", color: theme.WARNING },
  { icon: AudioLines, label: "Noise Confidence Scoring", color: theme.SUCCESS },
  { icon: CheckCircle, label: "Fix Verification Loop", color: "#E879F9" },
  { icon: Globe, label: "Cross-Project Memory", color: theme.CYAN },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-name">TEAM DYNAMITE</div>
        <div className="brand-sub">AI Debug Prioritization</div>
        <div className="brand-bar" />
      </div>
      <div className="sidebar-divider" />
      <div className="sidebar-section">
        <p className="section-title">Pipeline Layers</p>
        {pipelineLayers.map(({ icon: Icon, label, color }, i) => (
          <div key={i} className="section-item">
            <Icon size={14} style={{ color, flexShrink: 0, opacity: 0.7 }} />
            <span>L{i} &nbsp;{label}</span>
          </div>
        ))}
        <div style={{ marginTop: 20 }}>
          <p className="section-title">High-Impact Features</p>
          {features.map(({ icon: Icon, label, color }, i) => (
            <div key={i} className="feature-item" style={{ color }}>
              <Icon size={14} style={{ flexShrink: 0 }} />
              <span>{i + 1}. {label}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="sidebar-divider" />
      <div className="sidebar-footer">
        SanDisk  |  Western Digital<br />
        Hackathon 2024
      </div>
    </aside>
  );
}
