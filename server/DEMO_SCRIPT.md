# Team Dynamite: 4-Minute Hackathon Demo Script

**Speaker Setup:** Have the dashboard (`streamlit run app.py`) already running in the browser on the **Upload & Run** tab. Ensure `project_memory.json` has at least one entry, or be ready to run it fresh. Have the Sample Logs folder ready.

---

## 0:00 - 0:30 | The Problem & The Solution (Upload & Run)
**Action:** Click **"Run Demo Mode"** (or drag-and-drop the 3 sample log files).
**Talk Track:**
*"Welcome to Team Dynamite. Today, SoC verification teams spend 60% of their time drowning in millions of lines of unstructured simulation logs, manually trying to group failures and figure out who should fix them. We built an AI-accelerated Triaging Engine that completely automates this.* 
*(Gesturing to the screen as progress bars complete)* 
*Watch as our engine ingests, embeds, and density-clusters 500+ log lines into actionable root causes in under 10 seconds. Let's see the results."*

---

## 0:30 - 1:15 | Feature 1: Failure DNA Fingerprint & Feature 2: Git Blame Auto-Assignee 
**Action:** Click the **"Ranked Debug Tasks"** tab.
**Talk Track:**
*"Here is our prioritized task list. Notice the colored stacked bars on each card? That's our **Failure DNA Fingerprint**. We extract a severity ratio vector—Fatal, Error, SVA, Warning—and fuse it directly into our LLM text embeddings. This forces bugs that look textually identical but behave differently (like an INFO message vs a FATAL crash) to separate into distinct clusters. It gives engineers instant visual context.*

*Next, look at the blue tags next to the DNA. This is our **Git Blame Auto-Assignee**. Our ingestion layer intelligently regex-matched the exact RTL or Testbench file and line number causing the failure, and mapped it to the likely owner. We just turned a raw log into an auto-assigned, Jira-ready ticket.*"

---

## 1:15 - 1:45 | Feature 3: Regression Trend Timeline
**Action:** Point to the red/green Sparklines on the cards, then click the **"Regression Timeline"** tab briefly, then go back.
**Talk Track:**
*"But failures don't happen in a vacuum—they evolve. Look at these inline sparklines. This is our persistent **Regression Trend Timeline**. We track the AI priority score of every unique failure signature across CI/CD runs. If a cascading failure is getting worse overnight, engineers get an immediate visual indicator (like a red 'Worsening' badge) so they know exactly what's degrading the build.*"

---

## 1:45 - 2:20 | Feature 4: Noise Confidence & Unique Bugs
**Action:** Click the **"Unique Bugs"** tab. 
**Talk Track:**
*"Standard clustering algorithms throw away 'noise' points. In verification, a noise point might be a critical zero-day bug! We built a custom **Novelty Scoring Engine** that fuses distance-to-centroid with HDBSCAN outlier probabilities.*
*(Point to the Hard/Medium/Soft noise tags)*
*We surface these unclustered logs as Hard, Medium, or Soft Noise. We guarantee that genuinely novel, one-off failures are never suppressed by the engine, giving designers confidence that nothing slips through the cracks.*"

---

## 2:20 - 3:00 | Feature 5: Fix Verification Loop (Live Demo)
**Action:** Go back to **"Ranked Debug Tasks"**. Find a cluster and click its blue **"Mark as Fixed"** button. The page reloads and shows it as green ("Fix Verified"). 
**Talk Track:**
*"How do we ensure a bug stays dead? Our **Fix Verification Loop**. When an engineer fixes a bug, they click this button to commit its signature to our persistent state index. Watch what happens: it turns green.* 
*(Imagine/Explain the next run)*
*If someone breaks the build tomorrow and this exact signature reappears in the logs, our engine instantly flags it with a critical red 'Regression' badge, stopping the broken code from merging.*"

---

## 3:00 - 3:45 | Feature 6: Cross-Project Memory
**Action:** Click the **"XAI + DNA Fingerprint"** tab and point to the "🧠 Memory Insight" callouts on the cards.
**Talk Track:**
*"Finally, our absolute killer feature: **Cross-Project Failure Memory**. We implemented a lightweight, CPU-friendly Vector Index. It permanently catalogs the centroids of every resolved cluster across the company.*
*(Point to the UI callout)*
*When a new project starts, if a bug surfaces that is >92% mathematically similar to a bug fixed three years ago on a different chip, the UI instantly surfaces: 'Memory Match! Seen before in Project Titan—previously fixed by editing axi_master.sv.' We are giving junior engineers the 20-year memory of a Principal Architect.*"

---

## 3:45 - 4:00 | Conclusion
**Action:** Click the **"Cluster Causal Graph"** tab to show the premium final UI.
**Talk Track:**
*"Combine all this with our brand new transition-probability Causal Graph that automatically extracts top failure chains, and you have a production-ready, zero-dependency, AI triage platform. Team Dynamite isn't just prioritizing debugs—we are fundamentally accelerating time-to-market for silicon. Thank you."*
