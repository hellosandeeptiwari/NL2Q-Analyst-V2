import React from 'react';

function HelpPanel() {
  return (
    <div style={{ background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #e2e8f0', padding: '1rem' }}>
      <h2>Help & Documentation</h2>
      <p>Welcome to NL2Q Agent! Here's how to use it:</p>
      <ul>
        <li>Enter a natural language question in the query panel.</li>
        <li>View the generated SQL and results.</li>
        <li>Visualizations appear inline in the chat.</li>
        <li>Generate insights from stored data.</li>
        <li>Download CSVs or view history/analytics.</li>
      </ul>
      <h3>API Docs</h3>
      <p>See README.md for full API documentation.</p>
      <h3>Support</h3>
      <p>Report issues via the error reporting endpoint.</p>
    </div>
  );
}

export default HelpPanel;
