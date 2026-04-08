/**
 * server.js — Node.js API server for Team Dynamite Debug Prioritizer.
 *
 * Bridges the React frontend to the Python ML pipeline via child_process.
 * Runs on port 3001.
 */

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({
  origin: "*",
  methods: ["GET", "POST"],
}));
app.use(express.json({ limit: "50mb" }));

// Upload directory
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const upload = multer({ dest: uploadDir });

// Python executable — try common names
const PYTHON = process.platform === "win32" ? "python" : "python3";
const BRIDGE_SCRIPT = path.join(__dirname, "python_bridge.py");

/**
 * Run a Python bridge command and return parsed JSON result.
 */
function runPython(args, timeoutMs = 300000) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON, [BRIDGE_SCRIPT, ...args], {
      cwd: __dirname,
      env: { ...process.env },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
      // Print Python stderr to Node console for debugging
      process.stderr.write(data);
    });

    const timer = setTimeout(() => {
      proc.kill("SIGTERM");
      reject(new Error("Python process timed out"));
    }, timeoutMs);

    proc.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        return reject(new Error(stderr || `Python exited with code ${code}`));
      }

      // Find the last JSON object in stdout (pipeline may print logs before it)
      const lines = stdout.trim().split("\n");
      let jsonStr = "";
      // Walk backwards to find the JSON line
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith("{") || line.startsWith("[")) {
          // Collect from this line to the end
          jsonStr = lines.slice(i).join("\n");
          break;
        }
      }

      if (!jsonStr) {
        return reject(new Error("No JSON output from Python bridge"));
      }

      try {
        resolve(JSON.parse(jsonStr));
      } catch (e) {
        reject(new Error(`JSON parse error: ${e.message}\nRaw output: ${jsonStr.slice(0, 500)}`));
      }
    });

    proc.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

// ────────────────────────────────────────────────────────────────────────
// API Routes
// ────────────────────────────────────────────────────────────────────────

/**
 * POST /api/run-demo
 * Run the pipeline with sample log files.
 */
app.post("/api/run-demo", async (req, res) => {
  console.log("[API] Running demo pipeline...");
  try {
    const result = await runPython(["run-demo"], 300000);

    console.log("RESULT:", result); // 👈 ADD THIS

    if (result.error) {
      console.error("[API] Pipeline error:", result.error);
      return res.status(500).json(result);
    }

    res.json(result);
  } catch (err) {
    console.error("[API ERROR]:", err); // 👈 IMPORTANT

    res.status(500).json({
      error: err.message || "Pipeline failed",
    });
  }
});

/**
 * POST /api/upload
 * Upload log files and run the pipeline.
 */
app.post("/api/upload", upload.array("files", 20), async (req, res) => {
  console.log("[API] Processing uploaded files...");
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: "No files uploaded" });
    }

    const filePaths = req.files.map((f) => f.path);
    const result = await runPython(["upload", ...filePaths], 300000);

    // Clean up uploaded files
    for (const f of req.files) {
      try { fs.unlinkSync(f.path); } catch {}
    }

    if (result.error) {
      return res.status(500).json(result);
    }
    console.log("[API] Upload pipeline complete");
    res.json(result);
  } catch (err) {
    console.error("[API] Upload pipeline failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/suggest-fix
 * Get an AI-powered fix suggestion for a cluster signature.
 */
app.post("/api/suggest-fix", async (req, res) => {
  try {
    const { signature, context } = req.body;
    const result = await runPython(
      ["suggest-fix", signature || "", context || ""],
      30000
    );
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/mark-fixed
 * Mark a cluster signature as fixed.
 */
app.post("/api/mark-fixed", async (req, res) => {
  try {
    const { signature, runId } = req.body;
    const result = await runPython(
      ["mark-fixed", signature || "", runId || "ui_run"],
      10000
    );
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/fix-state
 * Get the current fix validation state.
 */
app.get("/api/fix-state", async (req, res) => {
  try {
    const result = await runPython(["fix-state"], 10000);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/regression-history
 * Get the regression trend history.
 */
app.get("/api/regression-history", async (req, res) => {
  try {
    const result = await runPython(["regression-history"], 10000);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/simulate-fix
 * Simulate fixing a root cause cluster and calculate downstream impact.
 */
app.post("/api/simulate-fix", async (req, res) => {
  try {
    const { clusterId, dag } = req.body;
    // Write DAG to a temp file since it can be too large for CLI args
    const tmpFile = path.join(uploadDir, `dag_${Date.now()}.json`);
    fs.writeFileSync(tmpFile, JSON.stringify(dag || { nodes: [], edges: [] }));
    try {
      const result = await runPython(
        ["simulate-fix", String(clusterId || 0), tmpFile],
        30000
      );
      res.json(result);
    } finally {
      try { fs.unlinkSync(tmpFile); } catch {}
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", service: "Team Dynamite Debug Prioritizer API" });
});

// ────────────────────────────────────────────────────────────────────────
// Start Server
// ────────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n  Team Dynamite API Server running on http://localhost:${PORT}\n`);
});
