import React, { useState, useEffect } from 'react';
import { VisualizationPlan, TemporalComparisonCard } from './types';
import KPICard from './KPICard';
import TimelineView from './TimelineView';
import ComparisonCard from './ComparisonCard';
import Plot from 'react-plotly.js';
import './AdaptiveLayout.css';

interface AdaptiveLayoutProps {
  plan: VisualizationPlan;
  data: any[];
}

const AdaptiveLayout: React.FC<AdaptiveLayoutProps> = ({ plan, data }) => {
  // State for active comparison card
  const [activeComparisonCard, setActiveComparisonCard] = useState<number | null>(null);
  const [filteredData, setFilteredData] = useState<any[]>(data);

  // Set the first comparison card as active by default
  useEffect(() => {
    if (plan.temporal_context?.enabled && plan.temporal_context.comparison_periods.length > 0) {
      setActiveComparisonCard(0);
    }
  }, [plan.temporal_context]);

  // Update filtered data when active comparison card changes
  useEffect(() => {
    if (activeComparisonCard !== null && plan.temporal_context?.comparison_periods) {
      const card = plan.temporal_context.comparison_periods[activeComparisonCard];
      
      // Extract filter criteria from the card's time_period
      const filterKey = card.time_period.toLowerCase();
      
      // Try to filter data based on the card's context
      // For "By PDRPFlag" or "By Totaltrx", extract the column name
      if (filterKey.includes('by ')) {
        const columnName = filterKey.replace('by ', '').trim();
        
        // Find matching column in data (case-insensitive)
        const dataKeys = data.length > 0 ? Object.keys(data[0]) : [];
        const matchingKey = dataKeys.find(key => 
          key.toLowerCase() === columnName || 
          key.toLowerCase().includes(columnName)
        );
        
        if (matchingKey) {
          // For contextual insights, show data grouped/highlighted by this column
          // For now, just show all data with the context (actual grouping happens in chart)
          console.log(`🔍 Filtering by: ${matchingKey}`, card);
          setFilteredData(data);
        } else {
          setFilteredData(data);
        }
      } else {
        // No specific filter criteria found, show all data
        setFilteredData(data);
      }
    } else {
      setFilteredData(data);
    }
  }, [activeComparisonCard, data, plan.temporal_context]);

  // Get display data (use filtered data if comparison card is active)
  const getDisplayData = () => {
    return filteredData.length > 0 ? filteredData : data;
  };

  // Calculate KPI values from data with optional filtering - ROBUST for any data structure
  const calculateKPIValue = (spec: any): number => {
    if (!data || data.length === 0) return 0;

    // Apply filter if specified
    let workingData = data;
    if (spec.filter_condition) {
      const { filter_column, filter_value } = spec.filter_condition;
      workingData = data.filter(row => {
        const rowValue = row[filter_column];
        
        // Handle various comparison types robustly
        if (rowValue === null || rowValue === undefined) return false;
        
        // Try exact match first
        if (rowValue === filter_value) return true;
        
        // Try case-insensitive string comparison
        const rowStr = String(rowValue).trim().toLowerCase();
        const filterStr = String(filter_value).trim().toLowerCase();
        if (rowStr === filterStr) return true;
        
        // Try numeric comparison if both are numbers
        const rowNum = Number(rowValue);
        const filterNum = Number(filter_value);
        if (!isNaN(rowNum) && !isNaN(filterNum) && rowNum === filterNum) return true;
        
        return false;
      });
      
      console.log(`🔍 KPI "${spec.title}" filtered by ${filter_column}=${filter_value}: ${data.length} → ${workingData.length} rows`);
      
      if (workingData.length === 0) {
        const uniqueValues = Array.from(new Set(data.map(r => r[filter_column])));
        console.warn(`⚠️ No data matched filter ${filter_column}=${filter_value}. Available values:`, uniqueValues);
      }
    }

    if (workingData.length === 0) return 0;

    // Handle different calculation types
    switch (spec.calculation) {
      case 'count':
        return workingData.length;
        
      case 'sum':
      case 'mean':
      case 'max':
      case 'min':
      case 'percentage_change':
        const values = workingData
          .map(row => {
            const val = row[spec.value_column];
            const num = Number(val);
            return isNaN(num) ? 0 : num;
          })
          .filter(v => v !== 0 || spec.calculation === 'sum'); // Keep zeros for sum
        
        if (values.length === 0) return 0;

        switch (spec.calculation) {
          case 'sum':
            return values.reduce((a, b) => a + b, 0);
          case 'mean':
            return values.reduce((a, b) => a + b, 0) / values.length;
          case 'max':
            return Math.max(...values);
          case 'min':
            return Math.min(...values);
          case 'percentage_change':
            if (values.length >= 2) {
              return ((values[values.length - 1] - values[0]) / values[0]) * 100;
            }
            return 0;
          default:
            return values[0] || 0;
        }
        
      default:
        return 0;
    }
  };

  // Render primary chart using Plotly
  const renderChart = () => {
    const displayData = getDisplayData();
    if (!plan.primary_chart || !displayData || displayData.length === 0) return null;

    const { primary_chart } = plan;
    const xValues = displayData.map(row => row[primary_chart.x_axis]);
    const yValues = displayData.map(row => Number(row[primary_chart.y_axis]) || 0);

    let plotData: any[] = [];

    switch (primary_chart.type) {
      case 'line':
        plotData = [{
          x: xValues,
          y: yValues,
          type: 'scatter',
          mode: 'lines+markers',
          fill: primary_chart.style === 'area_fill' ? 'tozeroy' : 'none',
          marker: { color: '#667eea', size: 8 },
          line: { color: '#667eea', width: 3 },
        }];
        break;

      case 'bar':
        plotData = [{
          x: xValues,
          y: yValues,
          type: 'bar',
          marker: {
            color: '#667eea',
            line: { width: 0 }
          },
        }];
        break;

      case 'pie':
        plotData = [{
          labels: xValues,
          values: yValues,
          type: 'pie',
          marker: {
            colors: [
              '#667eea', '#764ba2', '#f093fb', '#f5576c',
              '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
            ]
          },
        }];
        break;

      case 'scatter':
        plotData = [{
          x: xValues,
          y: yValues,
          type: 'scatter',
          mode: 'markers',
          marker: {
            color: '#667eea',
            size: 10,
            line: { color: 'white', width: 2 }
          },
        }];
        break;

      case 'area':
        plotData = [{
          x: xValues,
          y: yValues,
          type: 'scatter',
          fill: 'tozeroy',
          fillcolor: 'rgba(102, 126, 234, 0.3)',
          line: { color: '#667eea', width: 2 },
        }];
        break;

      default:
        plotData = [{
          x: xValues,
          y: yValues,
          type: 'bar',
          marker: { color: '#667eea' },
        }];
    }

    const layout = {
      title: {
        text: primary_chart.title,
        font: { size: 18, color: '#1f2937' }
      },
      xaxis: {
        title: primary_chart.x_axis.replace('_', ' '),
        gridcolor: '#f3f4f6',
      },
      yaxis: {
        title: primary_chart.y_axis.replace('_', ' '),
        gridcolor: '#f3f4f6',
      },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      margin: { t: 60, r: 40, b: 60, l: 60 },
      showlegend: false,
    };

    return (
      <div className="chart-container">
        <Plot
          data={plotData}
          layout={layout}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
          }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    );
  };

  // Render comparison cards sidebar
  const renderComparisonSidebar = () => {
    if (!plan.temporal_context?.enabled || !plan.temporal_context.comparison_periods.length) {
      return null;
    }

    const { temporal_context } = plan;
    const contextType = temporal_context.context_type || 'temporal';
    const sidebarTitle = contextType === 'temporal' ? '⏰ Time Periods' : '📊 Contextual Insights';
    const sidebarSubtitle = temporal_context.query_timeframe || '';

    return (
      <div className="comparison-sidebar">
        <div className="comparison-sidebar-header">
          <h3>{sidebarTitle}</h3>
          {sidebarSubtitle && (
            <p className="comparison-sidebar-subtitle">{sidebarSubtitle}</p>
          )}
          {temporal_context.insight_type && (
            <span className="insight-type-badge">{temporal_context.insight_type.replace('_', ' ')}</span>
          )}
        </div>
        
        <div className="comparison-cards-container">
          {temporal_context.comparison_periods.map((card, idx) => (
            <ComparisonCard
              key={idx}
              card={card}
              isActive={activeComparisonCard === idx}
              onClick={() => {
                console.log(`🎯 Clicked comparison card ${idx}:`, card.time_period);
                setActiveComparisonCard(idx);
              }}
              contextType={contextType}
            />
          ))}
        </div>

        <div className="comparison-sidebar-footer">
          <p className="comparison-hint">
            💡 Click to highlight dimension in chart
          </p>
        </div>
      </div>
    );
  };

  // Render layout based on type
  const renderLayout = () => {
    const { layout_type, kpis, primary_chart, timeline, breakdown, temporal_context } = plan;
    const hasComparisons = temporal_context?.enabled && temporal_context.comparison_periods.length > 0;
    const hasSidebar = timeline?.enabled || breakdown?.enabled || hasComparisons;

    return (
      <div className={`adaptive-layout ${layout_type} ${hasComparisons ? 'has-comparisons' : ''}`}>
        {/* Layout Type Badge */}
        <div className="layout-badge">
          <span className="layout-type">{layout_type.replace('_', ' ')}</span>
          {plan.metadata?.llm_reasoning && (
            <span className="layout-reasoning" title={plan.metadata.llm_reasoning}>
              AI-Planned
            </span>
          )}
          {hasComparisons && (
            <span className="comparison-badge">
              {temporal_context.context_type === 'temporal' ? '⏰ Temporal' : '📊 Contextual'}
            </span>
          )}
        </div>

        {/* KPI Row */}
        {kpis && kpis.length > 0 && (
          <div className="kpi-row" style={{ gridTemplateColumns: `repeat(${Math.min(kpis.length, 4)}, 1fr)` }}>
            {kpis.map((kpi, idx) => (
              <KPICard
                key={idx}
                spec={kpi}
                value={calculateKPIValue(kpi)}
                data={getDisplayData()}
              />
            ))}
          </div>
        )}

        {/* Main Content Area */}
        <div className={`main-content ${hasSidebar ? 'with-sidebar' : 'full-width'}`}>
          {/* Primary Chart */}
          <div className="main-chart-area">
            {renderChart()}
            {activeComparisonCard !== null && temporal_context?.comparison_periods && (
              <div className="active-filter-badge">
                Showing: {temporal_context.comparison_periods[activeComparisonCard].time_period}
              </div>
            )}
          </div>

          {/* Sidebar for Comparisons, Timeline or Breakdown */}
          {hasSidebar && (
            <div className="sidebar">
              {/* Priority 1: Comparison Cards */}
              {hasComparisons && renderComparisonSidebar()}
              
              {/* Priority 2: Timeline */}
              {timeline?.enabled && (
                <TimelineView spec={timeline} data={getDisplayData()} />
              )}
              
              {/* Priority 3: Breakdown - Hidden until implemented */}
              {false && breakdown?.enabled && (
                <div className="breakdown-section">
                  <h3>Breakdown</h3>
                  <p>Coming soon...</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer with metadata */}
        {plan.metadata?.llm_reasoning && (
          <div className="layout-footer">
            <div className="reasoning-box">
              <strong>AI Analysis:</strong> {plan.metadata.llm_reasoning}
            </div>
          </div>
        )}
      </div>
    );
  };

  return renderLayout();
};

export default AdaptiveLayout;
