import React, { useState } from 'react';

interface DownloadButtonProps {
  jobId?: string;
  rows?: any[];
  disabled?: boolean;
}

function DownloadButton({ jobId = '', rows = [], disabled = true }: DownloadButtonProps) {
  const handleDownload = () => {
    if (!jobId || disabled) return;
    window.open(`http://localhost:8003/csv/${jobId}`);
  };

  const handleCSVDownload = () => {
    if (!rows.length) return;
    
    // Convert rows to CSV
    const headers = Object.keys(rows[0] || {});
    const csvContent = [
      headers.join(','),
      ...rows.map(row => headers.map(header => row[header] || '').join(','))
    ].join('\n');
    
    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query_results_${Date.now()}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div style={{ background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #e2e8f0', padding: '1rem' }}>
      <h2>Download CSV</h2>
      <div style={{ display: 'flex', gap: '1rem' }}>
        <button
          onClick={handleCSVDownload}
          disabled={disabled || !rows.length}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            backgroundColor: disabled || !rows.length ? '#e9ecef' : '#007bff',
            color: disabled || !rows.length ? '#6c757d' : '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: disabled || !rows.length ? 'not-allowed' : 'pointer'
          }}
        >
          Download Results ({rows.length} rows)
        </button>
      </div>
    </div>
  );
}

export default DownloadButton;
