// Simple module declaration for react-plotly.js
declare module 'react-plotly.js' {
  const Plot: any;
  export default Plot;
}

// Plotly global type definition
declare global {
  interface Window {
    Plotly: any;
  }
}

export {};
