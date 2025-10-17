import React from 'react';
import { KPISpec } from './types';
import { 
  FiTrendingUp, 
  FiTrendingDown, 
  FiActivity, 
  FiDollarSign, 
  FiUsers, 
  FiTarget,
  FiCheckCircle,
  FiAlertCircle 
} from 'react-icons/fi';
import './KPICard.css';

interface KPICardProps {
  spec: KPISpec;
  value: number | string;
  previousValue?: number;
  data?: any[];
}

const KPICard: React.FC<KPICardProps> = ({ spec, value, previousValue, data }) => {
  // Calculate trend if needed
  const trendValue = spec.trend && previousValue 
    ? ((Number(value) - previousValue) / previousValue * 100).toFixed(1)
    : null;

  const isPositiveTrend = trendValue ? Number(trendValue) > 0 : null;

  // Format value based on spec
  const formatValue = (val: number | string): string => {
    const numVal = Number(val);
    if (isNaN(numVal)) return String(val);

    switch (spec.format) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0,
        }).format(numVal);
      case 'percentage':
        return `${numVal.toFixed(1)}%`;
      case 'number':
      default:
        return new Intl.NumberFormat('en-US', {
          minimumFractionDigits: 0,
          maximumFractionDigits: 0,
        }).format(numVal);
    }
  };

  // Icon mapping
  const getIcon = () => {
    const iconMap: { [key: string]: React.ReactElement } = {
      'activity': <FiActivity />,
      'trendingUp': <FiTrendingUp />,
      'trendingDown': <FiTrendingDown />,
      'dollarSign': <FiDollarSign />,
      'users': <FiUsers />,
      'target': <FiTarget />,
      'checkCircle': <FiCheckCircle />,
      'alertCircle': <FiAlertCircle />,
    };
    return spec.icon ? iconMap[spec.icon] || <FiActivity /> : <FiActivity />;
  };

  // Simple sparkline (if enabled and data available)
  const renderSparkline = () => {
    if (!spec.sparkline || !data || data.length < 2) return null;

    const values = data.map(d => Number(d[spec.value_column || '']) || 0);
    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = max - min || 1;

    const points = values.map((val, idx) => {
      const x = (idx / (values.length - 1)) * 100;
      const y = 100 - ((val - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg className="kpi-sparkline" viewBox="0 0 100 30" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    );
  };

  return (
    <div className="kpi-card">
      <div className="kpi-header">
        <div className="kpi-icon">{getIcon()}</div>
        <div className="kpi-title">{spec.title}</div>
      </div>
      
      <div className="kpi-value-section">
        <div className="kpi-value">{formatValue(value)}</div>
        
        {spec.trend && trendValue !== null && (
          <div className={`kpi-trend ${isPositiveTrend ? 'positive' : 'negative'}`}>
            {isPositiveTrend ? <FiTrendingUp /> : <FiTrendingDown />}
            <span>{Math.abs(Number(trendValue))}%</span>
          </div>
        )}
      </div>

      {renderSparkline()}

      {/* Show time period if available */}
      {spec.time_period && (
        <div className="kpi-time-period">
          {spec.time_period}
        </div>
      )}

      {/* Show comparison text if available */}
      {spec.comparison_text && (
        <div className="kpi-comparison-text">
          {spec.comparison_text}
        </div>
      )}

      {/* Legacy trend comparison */}
      {!spec.comparison_text && spec.trend_comparison && (
        <div className="kpi-comparison">
          vs {spec.trend_comparison.replace('_', ' ')}
        </div>
      )}
    </div>
  );
};

export default KPICard;
