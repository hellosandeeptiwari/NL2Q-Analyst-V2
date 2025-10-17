import React, { useState } from 'react';
import { TemporalComparisonCard } from './types';
import { 
  FiCalendar, 
  FiTrendingUp, 
  FiUsers, 
  FiBarChart2,
  FiPieChart,
  FiActivity,
  FiTarget,
  FiLayers
} from 'react-icons/fi';
import './ComparisonCard.css';

interface ComparisonCardProps {
  card: TemporalComparisonCard;
  isActive: boolean;
  onClick: () => void;
  contextType: 'temporal' | 'contextual';
}

const ComparisonCard: React.FC<ComparisonCardProps> = ({ 
  card, 
  isActive, 
  onClick,
  contextType 
}) => {
  const [isHovered, setIsHovered] = useState(false);

  // Get icon based on metric type and label
  const getIcon = () => {
    if (contextType === 'temporal') {
      return <FiCalendar />;
    }

    // For contextual insights, determine icon by time_period text
    const label = card.time_period.toLowerCase();
    
    if (label.includes('specialty') || label.includes('region') || label.includes('territory')) {
      return <FiPieChart />;
    } else if (label.includes('prescriber') || label.includes('user')) {
      return <FiUsers />;
    } else if (label.includes('average') || label.includes('mean')) {
      return <FiActivity />;
    } else if (label.includes('top') || label.includes('performer')) {
      return <FiTrendingUp />;
    } else if (label.includes('distribution') || label.includes('breakdown')) {
      return <FiLayers />;
    } else if (label.includes('target') || label.includes('goal')) {
      return <FiTarget />;
    } else {
      return <FiBarChart2 />;
    }
  };

  // Get professional blue color scheme based on relative offset (priority)
  const getColorScheme = () => {
    const priority = Math.abs(card.relative_offset);
    
    // Professional blue/cyan shades only - no pink
    const colors = [
      { bg: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)', border: '#3b82f6' },
      { bg: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)', border: '#0ea5e9' },
      { bg: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)', border: '#8b5cf6' },
      { bg: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)', border: '#06b6d4' },
      { bg: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)', border: '#6366f1' },
      { bg: 'linear-gradient(135deg, #0891b2 0%, #0e7490 100%)', border: '#0891b2' },
    ];
    return colors[(priority - 1) % colors.length];
  };

  const colorScheme = getColorScheme();

  return (
    <div 
      className={`comparison-card ${isActive ? 'active' : ''} ${isHovered ? 'hovered' : ''}`}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        background: isActive ? colorScheme.bg : 'white',
        borderColor: isActive ? colorScheme.border : '#e5e7eb',
        color: isActive ? 'white' : '#1f2937'
      }}
    >
      <div className="comparison-card-icon" style={{ 
        color: isActive ? 'white' : colorScheme.border 
      }}>
        {getIcon()}
      </div>
      
      <div className="comparison-card-content">
        <div className="comparison-card-label">{card.time_period}</div>
        {card.summary_text && (
          <div className="comparison-card-description">{card.summary_text}</div>
        )}
      </div>

      {/* Priority badge */}
      <div className={`comparison-card-priority ${isActive ? 'active' : ''}`}>
        {Math.abs(card.relative_offset)}
      </div>

      {/* Active indicator */}
      {isActive && (
        <div className="comparison-card-active-indicator">
          <div className="pulse-dot"></div>
        </div>
      )}
    </div>
  );
};

export default ComparisonCard;
