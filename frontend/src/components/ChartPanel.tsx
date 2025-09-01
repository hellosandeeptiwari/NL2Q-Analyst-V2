import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

interface ChartPanelProps {
  plotlySpec?: any;
  response?: any;
}

function ChartPanel({ plotlySpec, response }: ChartPanelProps) {
  if (!plotlySpec) {
    return (
      <div style={{ textAlign: 'center', color: '#6c757d', padding: '2rem' }}>
        No visualization available. Execute a query to generate charts.
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ color: '#495057', marginBottom: '1rem', fontSize: '1.25rem' }}>Data Visualization</h2>
      <div style={{ border: '1px solid #e9ecef', borderRadius: '4px', padding: '1rem', background: '#fff' }}>
        <Plot 
          data={plotlySpec.data} 
          layout={plotlySpec.layout}
          style={{ width: '100%', height: '400px' }}
        />
      </div>
    </div>
  );
}

export default ChartPanel;
