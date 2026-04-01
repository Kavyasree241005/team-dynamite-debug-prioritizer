import React from "react";
import { DNA_COLORS } from "../theme";

export default function DnaBar({ dna = [0, 0, 0, 0], height = 6 }) {
  const labels = ["FATAL", "ERROR", "SVA", "WARNING"];
  if (!dna || dna.every((v) => v === 0)) return null;
  
  return (
    <div className="dna-bar" style={{ height }}>
      {labels.map((lbl, i) =>
        dna[i] > 0 ? (
          <div
            key={lbl}
            style={{
              width: `${dna[i] * 100}%`,
              background: DNA_COLORS[lbl],
              height: "100%",
            }}
            title={`${lbl}: ${(dna[i] * 100).toFixed(1)}%`}
          />
        ) : null
      )}
    </div>
  );
}
