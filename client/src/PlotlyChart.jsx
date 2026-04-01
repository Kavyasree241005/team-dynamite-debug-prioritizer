import reactPlotly from "react-plotly.js";

// Handle CJS/ESM interop: react-plotly.js exports { default: PlotComponent }
const Plot = reactPlotly.default || reactPlotly;

export default Plot;
