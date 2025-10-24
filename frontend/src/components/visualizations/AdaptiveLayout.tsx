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

  console.log('🔄 AdaptiveLayout rendering with data:', data?.length, 'rows');

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

      console.log(`🎯 Active comparison card changed to ${activeComparisonCard}:`, JSON.stringify(card, null, 2));

      // Check if the card has KPIs with filter conditions
      if (card.kpis && card.kpis.length > 0 && card.kpis[0].filter_condition) {
        const { filter_column, filter_value } = card.kpis[0].filter_condition;

        console.log(`🔍 Filtering chart data by ${filter_column}=${filter_value}`);

        // Find actual column name (case-insensitive)
        const dataKeys = data.length > 0 ? Object.keys(data[0]) : [];
        const actualColumn = dataKeys.find(key =>
          key.toLowerCase() === filter_column.toLowerCase() ||
          key.toLowerCase().replace(/\s+/g, '') === filter_column.toLowerCase().replace(/\s+/g, '') ||
          key.toLowerCase().includes(filter_column.toLowerCase())
        ) || filter_column;

        console.log(`📊 Using column: "${actualColumn}" (original: "${filter_column}")`);

        const filtered = data.filter(row => {
          const rowValue = row[actualColumn];
          if (rowValue === null || rowValue === undefined) return false;

          // Try multiple comparison methods
          if (rowValue === filter_value) return true;

          const rowStr = String(rowValue).trim().toLowerCase();
          const filterStr = String(filter_value).trim().toLowerCase();
          if (rowStr === filterStr) return true;

          const rowNum = Number(rowValue);
          const filterNum = Number(filter_value);
          if (!isNaN(rowNum) && !isNaN(filterNum) && rowNum === filterNum) return true;

          return false;
        });

        console.log(`✅ Filtered chart data: ${data.length} → ${filtered.length} rows`);

        if (filtered.length > 0) {
          setFilteredData(filtered);
        } else {
          console.warn(`⚠️ No data matched filter, showing all data`);
          setFilteredData(data);
        }
      } else {
        // No filter condition, show all data
        console.log(`ℹ️ No filter condition in comparison card, showing all data`);
        setFilteredData(data);
      }
    } else {
      setFilteredData(data);
    }
  }, [activeComparisonCard, data, plan.temporal_context]);

  // Get display data (use filtered data if comparison card is active)
  const getDisplayData = () => {
    // If no active card, show all data
    if (activeComparisonCard === null || !plan.temporal_context?.comparison_periods) {
      return filteredData.length > 0 ? filteredData : data;
    }

    const activeCard = plan.temporal_context.comparison_periods[activeComparisonCard];
    const cardLabel = activeCard.time_period.toLowerCase();

    // For statistical insight cards, filter data based on the statistic type
    if (plan.temporal_context.context_type === 'contextual' ||
      plan.temporal_context.insight_type === 'statistical_overview') {

      const displayData = filteredData.length > 0 ? filteredData : data;
      if (!displayData || displayData.length === 0) return displayData;

      // Get the y-axis column for calculations
      const yColumn = plan.primary_chart?.y_axis;
      if (!yColumn) return displayData;

      // Extract numeric values
      const values = displayData.map(row => Number(row[yColumn]) || 0);

      if (cardLabel.includes('max')) {
        // Show only the row with maximum value
        const maxValue = Math.max(...values);
        console.log(`🔍 Filtering to MAX value: ${maxValue}`);
        return displayData.filter(row => Number(row[yColumn]) === maxValue);
      }
      else if (cardLabel.includes('min')) {
        // Show only the row with minimum value
        const minValue = Math.min(...values);
        console.log(`🔍 Filtering to MIN value: ${minValue}`);
        return displayData.filter(row => Number(row[yColumn]) === minValue);
      }
      else if (cardLabel.includes('average') || cardLabel.includes('mean')) {
        // Show all data (average line will be added to chart separately)
        console.log(`📊 Showing all data for AVERAGE comparison`);
        return displayData;
      }
      else if (cardLabel.includes('total') || cardLabel.includes('count')) {
        // Show all data (this is the total view)
        console.log(`📊 Showing all data for TOTAL view`);
        return displayData;
      }
    }

    // For other types of cards (temporal, regional, etc.), use filtered data
    return filteredData.length > 0 ? filteredData : data;
  };

  // Calculate KPI values from data with optional filtering - ROBUST for any data structure
  const calculateKPIValue = (spec: any): number => {
    console.log(`💳 Calculating KPI "${spec.title}":`, JSON.stringify({
      value_column: spec.value_column,
      calculation: spec.calculation,
      filter_condition: spec.filter_condition
    }, null, 2));

    if (!data || data.length === 0) return 0;

    // Apply filter if specified
    let workingData = data;
    if (spec.filter_condition) {
      const { filter_column, filter_value } = spec.filter_condition;

      console.log(`🔍 Attempting to filter KPI "${spec.title}" by ${filter_column}=${filter_value}`);
      console.log(`📊 Available columns in data:`, data[0] ? Object.keys(data[0]) : []);

      // Find the actual column name (case-insensitive match)
      const dataKeys = data.length > 0 ? Object.keys(data[0]) : [];
      const actualFilterColumn = dataKeys.find(key =>
        key.toLowerCase() === filter_column.toLowerCase() ||
        key.toLowerCase().replace(/\s+/g, '') === filter_column.toLowerCase().replace(/\s+/g, '') ||
        key.toLowerCase().includes(filter_column.toLowerCase())
      ) || filter_column;

      console.log(`🎯 Using column: "${actualFilterColumn}" (original: "${filter_column}")`);

      workingData = data.filter(row => {
        const rowValue = row[actualFilterColumn];

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

      console.log(`✅ KPI "${spec.title}" filtered: ${data.length} → ${workingData.length} rows`);

      if (workingData.length === 0) {
        const uniqueValues = Array.from(new Set(data.map(r => r[actualFilterColumn]))).slice(0, 10);
        console.warn(`⚠️ No data matched filter ${actualFilterColumn}=${filter_value}. Sample values:`, uniqueValues);
        return 0;
      }
    }

    if (workingData.length === 0) return 0;

    // 🔧 CRITICAL FIX: For aggregated data (already grouped), just read the value from the filtered row
    // Check if data is already aggregated (only 1 row after filtering)
    if (workingData.length === 1 && spec.filter_condition) {
      // Data is pre-aggregated, just read the value directly
      const value = workingData[0][spec.value_column];
      const num = Number(value);
      console.log(`📊 Using pre-aggregated value for "${spec.title}":`, num);
      return isNaN(num) ? 0 : num;
    }

    // Handle different calculation types for detail data
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
    if (!plan.primary_chart || !displayData || displayData.length === 0) {
      console.log('⚠️ No chart to render:', {
        hasPrimaryChart: !!plan.primary_chart,
        hasData: !!displayData,
        dataLength: displayData?.length
      });
      return null;
    }

    const { primary_chart } = plan;

    // Check if x_axis and y_axis columns exist in the data
    const dataKeys = displayData.length > 0 ? Object.keys(displayData[0]) : [];
    if (!dataKeys.includes(primary_chart.x_axis) || !dataKeys.includes(primary_chart.y_axis)) {
      console.warn('⚠️ Chart axes not found in data:', {
        requestedX: primary_chart.x_axis,
        requestedY: primary_chart.y_axis,
        availableColumns: dataKeys
      });
      return null;
    }

    const xValues = displayData.map(row => row[primary_chart.x_axis]);
    const yValues = displayData.map(row => Number(row[primary_chart.y_axis]) || 0);

    let plotData: any[] = [];

    // Check if we should add average line (when Average card is active)
    const shouldShowAverageLine = activeComparisonCard !== null &&
      plan.temporal_context?.comparison_periods &&
      plan.temporal_context.comparison_periods[activeComparisonCard]?.time_period.toLowerCase().includes('average');

    // Calculate average from ALL data (not just filtered)
    const allYValues = data.map(row => Number(row[primary_chart.y_axis]) || 0);
    const averageValue = allYValues.reduce((a, b) => a + b, 0) / allYValues.length;

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
          name: primary_chart.y_axis
        }];

        // Add average line if needed
        if (shouldShowAverageLine) {
          plotData.push({
            x: xValues,
            y: Array(xValues.length).fill(averageValue),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ef4444', width: 2, dash: 'dash' },
            name: `Average (${averageValue.toFixed(1)})`,
            showlegend: true
          });
        }
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
          name: primary_chart.y_axis
        }];

        // Add average line if needed
        if (shouldShowAverageLine) {
          plotData.push({
            x: xValues,
            y: Array(xValues.length).fill(averageValue),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ef4444', width: 3, dash: 'dash' },
            name: `Average (${averageValue.toFixed(1)})`,
            showlegend: true
          });
        }
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
        font: { size: 16, color: '#1f2937' }
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
      margin: { t: 40, r: 20, b: 50, l: 60 },
      showlegend: shouldShowAverageLine, // Show legend when average line is present
      legend: {
        orientation: 'h',
        yanchor: 'bottom',
        y: 1.02,
        xanchor: 'right',
        x: 1
      },
      autosize: true,
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
          useResizeHandler={true}
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
          {/* Primary Chart - Only show if there's actually data to chart */}
          {primary_chart && getDisplayData() && getDisplayData().length > 0 && (
            <div className="main-chart-area">
              {renderChart()}
              {activeComparisonCard !== null && temporal_context?.comparison_periods && (
                <div className="active-filter-badge">
                  Showing: {temporal_context.comparison_periods[activeComparisonCard].time_period}
                </div>
              )}
            </div>
          )}
          
          {/* Show helpful message when no chart is available */}
          {(!primary_chart || !getDisplayData() || getDisplayData().length === 0) && (
            <div className="main-chart-area" style={{
              background: '#f8fafc',
              border: '2px dashed #e5e7eb',
              borderRadius: '8px',
              padding: '32px',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#6b7280',
              height: '420px'
            }}>
              <div style={{ fontSize: '48px', marginBottom: '16px' }}>📊</div>
              <div style={{ fontSize: '16px', fontWeight: '600', marginBottom: '8px', color: '#374151' }}>
                Single Value Result
              </div>
              <div style={{ fontSize: '14px' }}>
                This query returned a single aggregate value. No chart visualization available.
              </div>
            </div>
          )}          {/* Sidebar for Comparisons, Timeline or Breakdown */}
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
