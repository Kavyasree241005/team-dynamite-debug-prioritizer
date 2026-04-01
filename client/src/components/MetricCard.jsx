import React from "react";

export default function MetricCard({ label, value, color = "#F8FAFC" }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={{ color }}>{value}</div>
    </div>
  );
}
