import React, { useState } from "react";
import {
  Upload, Target, GitBranch, Dna, TrendingUp, Search, Wrench,
} from "lucide-react";
import Sidebar from "./components/Sidebar";
import UploadTab from "./components/UploadTab";
import RankedTasksTab from "./components/RankedTasksTab";
import DagTab from "./components/DagTab";
import XaiTab from "./components/XaiTab";
import TimelineTab from "./components/TimelineTab";
import UniqueBugsTab from "./components/UniqueBugsTab";
import FixHistoryTab from "./components/FixHistoryTab";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { runDemo, uploadFiles } from "./api";
import "./index.css";

const TABS = [
  { id: "upload", label: "Upload & Run", icon: Upload },
  { id: "ranked", label: "Ranked Tasks", icon: Target },
  { id: "dag", label: "Root Cause DAG", icon: GitBranch },
  { id: "xai", label: "XAI + DNA", icon: Dna },
  { id: "timeline", label: "Timeline", icon: TrendingUp },
  { id: "unique", label: "Unique Bugs", icon: Search },
  { id: "fixes", label: "Fix History", icon: Wrench },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  const handleRunDemo = async () => {
    setLoading(true);
    setProgress(10);
    setError(null);
    try {
      const interval = setInterval(() => {
        setProgress((p) => Math.min(p + Math.random() * 12, 92));
      }, 1000);
      const data = await runDemo();
      clearInterval(interval);
      setProgress(100);
      setResults(data);
      setTimeout(() => {
        setProgress(0);
        setLoading(false);
      }, 600);
    } catch (err) {
      setError(err.message);
      setLoading(false);
      setProgress(0);
    }
  };

  const handleUpload = async (files) => {
    setLoading(true);
    setProgress(10);
    setError(null);
    try {
      const interval = setInterval(() => {
        setProgress((p) => Math.min(p + Math.random() * 12, 92));
      }, 1000);
      const data = await uploadFiles(files);
      clearInterval(interval);
      setProgress(100);
      setResults(data);
      setTimeout(() => {
        setProgress(0);
        setLoading(false);
      }, 600);
    } catch (err) {
      setError(err.message);
      setLoading(false);
      setProgress(0);
    }
  };

  const renderTab = () => {
    switch (activeTab) {
      case "upload":
        return <UploadTab results={results} onRunDemo={handleRunDemo} onUpload={handleUpload} loading={loading} progress={progress} error={error} />;
      case "ranked":
        return <RankedTasksTab results={results} />;
      case "dag":
        return <DagTab results={results} />;
      case "xai":
        return <XaiTab results={results} />;
      case "timeline":
        return <TimelineTab results={results} />;
      case "unique":
        return <UniqueBugsTab results={results} />;
      case "fixes":
        return <FixHistoryTab results={results} />;
      default:
        return null;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        {/* Tab bar */}
        <div className="tab-bar">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`tab-btn ${activeTab === tab.id ? "tab-active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <span className="tab-icon"><Icon size={16} /></span> {tab.label}
              </button>
            );
          })}
        </div>
        {/* Tab content */}
        <div className="tab-panel">
          <ErrorBoundary key={activeTab}>
            {renderTab()}
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
}
