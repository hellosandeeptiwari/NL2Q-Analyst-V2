import React, { useEffect, useState } from 'react';

interface ResultsGridProps {
  rows?: any[];
  columns?: string[];
  jobId?: string;
}

function ResultsGrid({ rows = [], columns = [], jobId = '' }: ResultsGridProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortColumn, setSortColumn] = useState('');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [dateFilter, setDateFilter] = useState('');

  const filteredRows = rows.filter(row =>
    Object.values(row).some(val =>
      String(val).toLowerCase().includes(searchTerm.toLowerCase())
    )
  );

  const sortedRows = [...filteredRows].sort((a, b) => {
    if (!sortColumn) return 0;
    const aVal = a[sortColumn];
    const bVal = b[sortColumn];
    if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
    if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
    return 0;
  });

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  if (!rows.length) {
    return (
      <div style={{ textAlign: 'center', color: '#6c757d', padding: '2rem' }}>
        No results yet. Execute a query to see data.
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ color: '#495057', marginBottom: '1rem', fontSize: '1.25rem' }}>Query Results</h2>

      {/* Filters */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Search..."
          value={searchTerm}
          onChange={e => setSearchTerm(e.target.value)}
          style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ddd' }}
        />
        <input
          type="date"
          value={dateFilter}
          onChange={e => setDateFilter(e.target.value)}
          style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ddd' }}
        />
      </div>

      <div style={{ overflowX: 'auto', border: '1px solid #e9ecef', borderRadius: '4px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead style={{ background: '#f8f9fa' }}>
            <tr>
              {Object.keys(rows[0] || {}).map(col => (
                <th
                  key={col}
                  onClick={() => handleSort(col)}
                  style={{
                    borderBottom: '2px solid #dee2e6',
                    padding: '0.75rem',
                    textAlign: 'left',
                    fontWeight: '600',
                    color: '#495057',
                    cursor: 'pointer',
                    userSelect: 'none'
                  }}
                >
                  {col}
                  {sortColumn === col && (
                    <span style={{ marginLeft: '0.5rem' }}>
                      {sortDirection === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8f9fa' }}>
                {Object.values(row).map((val, j) => (
                  <td key={j} style={{ padding: '0.75rem', borderBottom: '1px solid #e9ecef', color: '#495057' }}>
                    {String(val)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ResultsGrid;
