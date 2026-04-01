const API_BASE = "http://localhost:3001/api";

export async function runDemo() {
  const res = await fetch(`${API_BASE}/run-demo`, { method: "POST" });
  if (!res.ok) throw new Error((await res.json()).error || "Pipeline failed");
  return res.json();
}

export async function uploadFiles(files) {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
  if (!res.ok) throw new Error((await res.json()).error || "Upload failed");
  return res.json();
}

export async function suggestFix(signature, context = "") {
  const res = await fetch(`${API_BASE}/suggest-fix`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ signature, context }),
  });
  return res.json();
}

export async function markAsFixed(signature, runId = "ui_run") {
  const res = await fetch(`${API_BASE}/mark-fixed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ signature, runId }),
  });
  return res.json();
}

export async function getFixState() {
  const res = await fetch(`${API_BASE}/fix-state`);
  return res.json();
}

export async function getRegressionHistory() {
  const res = await fetch(`${API_BASE}/regression-history`);
  return res.json();
}

export async function simulateFix(clusterId, dag) {
  const res = await fetch(`${API_BASE}/simulate-fix`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clusterId, dag }),
  });
  return res.json();
}
