import React from "react";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          background: "#0F172A",
          border: "1px solid rgba(244,63,94,0.3)",
          borderLeft: "3px solid #F43F5E",
          borderRadius: 12,
          padding: 24,
          margin: 16,
          color: "#F1F5F9",
        }}>
          <h3 style={{ color: "#F43F5E", marginBottom: 8, fontSize: "1rem", fontWeight: 700 }}>Component Error</h3>
          <p style={{ color: "#94A3B8", fontSize: "0.88rem", lineHeight: 1.6 }}>
            {this.state.error?.message || "An unexpected error occurred."}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: 14,
              background: "linear-gradient(135deg, #4F46E5, #6366F1)",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              padding: "10px 20px",
              cursor: "pointer",
              fontFamily: "inherit",
              fontWeight: 600,
              fontSize: "0.85rem",
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
