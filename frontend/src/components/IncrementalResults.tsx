import React, { useState, useEffect } from 'react';
import { FiEye, FiEyeOff, FiChevronDown, FiChevronUp } from 'react-icons/fi';
import Plot from 'react-plotly.js';
import EnhancedTable from './EnhancedTable';
import './IncrementalResults.css';

interface PartialResult {
  id: string;
  stepId: string;
  stepName: string;
  type: 'query' | 'chart' | 'insight' | 'error';
  data?: any;
  timestamp: number;
  isComplete: boolean;
  preview?: boolean;
}

interface IncrementalResultsProps {
  results: PartialResult[];
  showPreviews?: boolean;
  maxPreviewRows?: number;
}

const IncrementalResults: React.FC<IncrementalResultsProps> = ({
  results,
  showPreviews = true,
  maxPreviewRows = 5
}) => {
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());
  const [hiddenResults, setHiddenResults] = useState<Set<string>>(new Set());

  // Auto-expand new results
  useEffect(() => {
    const latestResult = results[results.length - 1];
    if (latestResult && latestResult.isComplete) {
      setExpandedResults(prev => {
        const newSet = new Set(prev);
        newSet.add(latestResult.id);
        return newSet;
      });
    }
  }, [results]);

  const toggleExpanded = (resultId: string) => {
    setExpandedResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(resultId)) {
        newSet.delete(resultId);
      } else {
        newSet.add(resultId);
      }
      return newSet;
    });
  };

  const toggleHidden = (resultId: string) => {
    setHiddenResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(resultId)) {
        newSet.delete(resultId);
      } else {
        newSet.add(resultId);
      }
      return newSet;
    });
  };

  const renderPreview = (result: PartialResult) => {
    if (!showPreviews || !result.data) return null;

    switch (result.type) {
      case 'query':
        if (Array.isArray(result.data) && result.data.length > 0) {
          const previewData = result.data.slice(0, maxPreviewRows);
          return (
            <div className="result-preview">
              <div className="preview-label">
                Data Preview ({result.data.length} rows total)
              </div>
              <EnhancedTable 
                data={previewData}
                maxHeight="200px"
                showPagination={false}
                showExport={false}
                compact={true}
              />
              {result.data.length > maxPreviewRows && (
                <div className="preview-more">
                  +{result.data.length - maxPreviewRows} more rows
                </div>
              )}
            </div>
          );
        }
        break;

      case 'chart':
        if (result.data && result.data.data) {
          return (
            <div className="result-preview chart-preview">
              <div className="preview-label">Chart Preview</div>
              <div className="chart-preview-container">
                <Plot
                  data={result.data.data}
                  layout={{
                    ...result.data.layout,
                    width: 300,
                    height: 200,
                    margin: { t: 20, r: 10, b: 20, l: 30 },
                    showlegend: false
                  }}
                  config={{ 
                    displayModeBar: false,
                    responsive: false
                  }}
                />
              </div>
            </div>
          );
        }
        break;

      case 'insight':
        return (
          <div className="result-preview insight-preview">
            <div className="preview-label">Insight</div>
            <div className="insight-text">
              {typeof result.data === 'string' ? result.data : JSON.stringify(result.data)}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const renderFullResult = (result: PartialResult) => {
    if (!result.data) return null;

    switch (result.type) {
      case 'query':
        return (
          <div className="full-result">
            <EnhancedTable data={result.data} />
          </div>
        );

      case 'chart':
        return (
          <div className="full-result">
            <Plot
              data={result.data.data}
              layout={{
                ...result.data.layout,
                autosize: true,
                margin: { t: 40, r: 20, b: 40, l: 40 }
              }}
              config={{
                displayModeBar: true,
                responsive: true,
                displaylogo: false
              }}
              style={{ width: '100%', height: '400px' }}
            />
          </div>
        );

      case 'insight':
        return (
          <div className="full-result insight-full">
            <div className="insight-content">
              {typeof result.data === 'string' ? result.data : JSON.stringify(result.data, null, 2)}
            </div>
          </div>
        );

      default:
        return <div className="full-result">No preview available</div>;
    }
  };

  const getResultIcon = (type: string) => {
    switch (type) {
      case 'query': return '📊';
      case 'chart': return '📈';
      case 'insight': return '💡';
      case 'error': return '❌';
      default: return '📄';
    }
  };

  const getResultStatus = (result: PartialResult) => {
    if (result.type === 'error') return 'error';
    if (!result.isComplete) return 'processing';
    return 'complete';
  };

  return (
    <div className="incremental-results">
      <div className="results-header">
        <h3>Analysis Results</h3>
        <div className="results-stats">
          {results.length} step{results.length !== 1 ? 's' : ''} • 
          {results.filter(r => r.isComplete).length} completed
        </div>
      </div>

      <div className="results-list">
        {results.map((result) => {
          const isExpanded = expandedResults.has(result.id);
          const isHidden = hiddenResults.has(result.id);
          const status = getResultStatus(result);

          if (isHidden) {
            return (
              <div key={result.id} className="result-item hidden">
                <button 
                  className="unhide-button"
                  onClick={() => toggleHidden(result.id)}
                  title="Show result"
                >
                  <FiEye /> Show {result.stepName}
                </button>
              </div>
            );
          }

          return (
            <div key={result.id} className={`result-item ${status}`}>
              <div className="result-header">
                <div className="result-info">
                  <span className="result-icon">{getResultIcon(result.type)}</span>
                  <div className="result-details">
                    <div className="result-title">{result.stepName}</div>
                    <div className="result-meta">
                      <span className="result-type">{result.type}</span>
                      <span className="result-time">
                        {new Date(result.timestamp).toLocaleTimeString()}
                      </span>
                      {!result.isComplete && (
                        <span className="processing-indicator">Processing...</span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="result-actions">
                  <button
                    className="hide-button"
                    onClick={() => toggleHidden(result.id)}
                    title="Hide result"
                  >
                    <FiEyeOff />
                  </button>
                  
                  {result.isComplete && (
                    <button
                      className="expand-button"
                      onClick={() => toggleExpanded(result.id)}
                      title={isExpanded ? "Collapse" : "Expand"}
                    >
                      {isExpanded ? <FiChevronUp /> : <FiChevronDown />}
                    </button>
                  )}
                </div>
              </div>

              {/* Show preview for incomplete results or when collapsed */}
              {(!result.isComplete || !isExpanded) && renderPreview(result)}

              {/* Show full result when expanded and complete */}
              {result.isComplete && isExpanded && renderFullResult(result)}

              {/* Processing indicator */}
              {!result.isComplete && (
                <div className="processing-bar">
                  <div className="processing-fill"></div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {results.length === 0 && (
        <div className="no-results">
          <div className="no-results-icon">📊</div>
          <div className="no-results-text">Results will appear here as analysis progresses</div>
        </div>
      )}
    </div>
  );
};

export default IncrementalResults;
