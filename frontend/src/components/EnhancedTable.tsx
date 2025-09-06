import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { FiSearch, FiDownload, FiChevronUp, FiChevronDown, FiMoreHorizontal } from 'react-icons/fi';
import './EnhancedTable.css';

interface EnhancedTableProps {
  data: any[];
  title?: string;
  description?: string;
  maxHeight?: string;
  onExport?: (data: any[], format: 'csv' | 'excel') => void;
}

interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}

const EnhancedTable: React.FC<EnhancedTableProps> = ({
  data,
  title = "Query Results",
  description = "Data retrieved from your query",
  maxHeight = "400px",
  onExport
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState<SortConfig | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [highlightedRow, setHighlightedRow] = useState<number | null>(null);
  const [columnWidths, setColumnWidths] = useState<{[key: string]: number}>({});
  const [isResizing, setIsResizing] = useState(false);
  
  const searchTimeoutRef = useRef<NodeJS.Timeout>();
  const tableRef = useRef<HTMLTableElement>(null);

  // Get column names
  const columns = useMemo(() => {
    if (!data || data.length === 0) return [];
    return Object.keys(data[0]);
  }, [data]);

  // Debounced search functionality
  const debouncedSearch = useCallback((term: string) => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    searchTimeoutRef.current = setTimeout(() => {
      setCurrentPage(1); // Reset to first page when searching
    }, 300);
  }, []);

  useEffect(() => {
    debouncedSearch(searchTerm);
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [searchTerm, debouncedSearch]);

  // Memoized filtered and sorted data
  const processedData = useMemo(() => {
    let filteredData = [...data];

    // Apply search filter
    if (searchTerm) {
      filteredData = filteredData.filter(row =>
        Object.values(row).some(value => 
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    // Apply sorting
    if (sortConfig) {
      filteredData.sort((a, b) => {
        const aValue = a[sortConfig.key];
        const bValue = b[sortConfig.key];
        
        // Handle different data types
        if (aValue === null || aValue === undefined) return 1;
        if (bValue === null || bValue === undefined) return -1;
        
        // Numeric comparison
        if (!isNaN(Number(aValue)) && !isNaN(Number(bValue))) {
          return sortConfig.direction === 'asc' 
            ? Number(aValue) - Number(bValue)
            : Number(bValue) - Number(aValue);
        }
        
        // String comparison
        const aStr = String(aValue).toLowerCase();
        const bStr = String(bValue).toLowerCase();
        
        if (sortConfig.direction === 'asc') {
          return aStr < bStr ? -1 : aStr > bStr ? 1 : 0;
        } else {
          return aStr > bStr ? -1 : aStr < bStr ? 1 : 0;
        }
      });
    }

    return filteredData;
  }, [data, searchTerm, sortConfig]);

  // Pagination
  const totalPages = Math.ceil(processedData.length / pageSize);
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize;
    return processedData.slice(startIndex, startIndex + pageSize);
  }, [processedData, currentPage, pageSize]);

  // Handle sorting
  const handleSort = useCallback((column: string) => {
    setSortConfig(prevConfig => {
      if (prevConfig?.key === column) {
        // Toggle direction
        return {
          key: column,
          direction: prevConfig.direction === 'asc' ? 'desc' : 'asc'
        };
      } else {
        // New column
        return { key: column, direction: 'asc' };
      }
    });
  }, []);

  // Export functionality
  const handleExport = useCallback((format: 'csv' | 'excel') => {
    if (onExport) {
      onExport(processedData, format);
      return;
    }

    // Default CSV export
    if (format === 'csv') {
      const csvContent = [
        columns.join(','),
        ...processedData.map(row => 
          columns.map(col => `"${String(row[col] || '')}"`).join(',')
        )
      ].join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.csv`;
      link.click();
      URL.revokeObjectURL(url);
    }
  }, [processedData, columns, title, onExport]);

  // Column resizing
  const handleMouseDown = useCallback((column: string, e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    
    const startX = e.clientX;
    const startWidth = columnWidths[column] || 150;

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = Math.max(80, startWidth + (e.clientX - startX));
      setColumnWidths(prev => ({ ...prev, [column]: newWidth }));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [columnWidths]);

  if (!data || data.length === 0) {
    return (
      <div className="enhanced-table-container">
        <div className="table-header">
          <h4 className="table-title">No Data Available</h4>
          <p className="table-description">No results to display</p>
        </div>
      </div>
    );
  }

  return (
    <div className="enhanced-table-container">
      {/* Table Header with Controls */}
      <div className="table-header">
        <div className="table-title-section">
          <h4 className="table-title">{title} ({processedData.length} rows)</h4>
          <p className="table-description">{description}</p>
        </div>
        
        <div className="table-controls">
          {/* Search */}
          <div className="search-container">
            <FiSearch className="search-icon" />
            <input
              type="text"
              placeholder="Search data..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          {/* Export Options */}
          <div className="export-controls">
            <button 
              onClick={() => handleExport('csv')}
              className="export-btn"
              title="Export as CSV"
            >
              <FiDownload size={14} />
              CSV
            </button>
          </div>

          {/* Page Size Selector */}
          <select 
            value={pageSize} 
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="page-size-select"
          >
            <option value={10}>10 rows</option>
            <option value={25}>25 rows</option>
            <option value={50}>50 rows</option>
            <option value={100}>100 rows</option>
          </select>
        </div>
      </div>

      {/* Table Container */}
      <div className="table-scroll-container" style={{ maxHeight }}>
        <table ref={tableRef} className="enhanced-results-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th 
                  key={column} 
                  className={`table-header-cell ${sortConfig?.key === column ? 'sorted' : ''}`}
                  style={{ width: columnWidths[column] || 'auto' }}
                  onClick={() => handleSort(column)}
                >
                  <div className="header-content">
                    <span className="header-text">{column}</span>
                    <div className="sort-indicators">
                      <FiChevronUp 
                        className={`sort-icon ${sortConfig?.key === column && sortConfig?.direction === 'asc' ? 'active' : ''}`}
                        size={12}
                      />
                      <FiChevronDown 
                        className={`sort-icon ${sortConfig?.key === column && sortConfig?.direction === 'desc' ? 'active' : ''}`}
                        size={12}
                      />
                    </div>
                    <div 
                      className="resize-handle"
                      onMouseDown={(e) => handleMouseDown(column, e)}
                    />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, rowIndex) => (
              <tr 
                key={rowIndex} 
                className={`table-row ${highlightedRow === rowIndex ? 'highlighted' : ''}`}
                onMouseEnter={() => setHighlightedRow(rowIndex)}
                onMouseLeave={() => setHighlightedRow(null)}
              >
                {columns.map((column, colIndex) => (
                  <td key={colIndex} className="table-cell">
                    {row[column] !== null && row[column] !== undefined ? String(row[column]) : '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="table-pagination">
          <div className="pagination-info">
            Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, processedData.length)} of {processedData.length} entries
            {searchTerm && ` (filtered from ${data.length} total entries)`}
          </div>
          
          <div className="pagination-controls">
            <button 
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
              className="pagination-btn"
            >
              First
            </button>
            <button 
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="pagination-btn"
            >
              Previous
            </button>
            
            <span className="page-info">
              Page {currentPage} of {totalPages}
            </span>
            
            <button 
              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages}
              className="pagination-btn"
            >
              Next
            </button>
            <button 
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage === totalPages}
              className="pagination-btn"
            >
              Last
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancedTable;
