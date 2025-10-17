import React from 'react';
import { TimelineSpec } from './types';
import { FiClock, FiMail, FiPhone, FiCalendar, FiCheckCircle } from 'react-icons/fi';
import './TimelineView.css';

interface TimelineViewProps {
  spec: TimelineSpec;
  data: any[];
}

interface TimelineItem {
  date: Date;
  label: string;
  value?: number | string;
  type?: string;
}

const TimelineView: React.FC<TimelineViewProps> = ({ spec, data }) => {
  // Process data into timeline items
  const processTimelineData = (): TimelineItem[] => {
    if (!data || data.length === 0) return [];

    const items = data.map(row => {
      const dateValue = row[spec.time_column];
      const date = dateValue instanceof Date ? dateValue : new Date(dateValue);
      
      return {
        date,
        label: formatLabel(row),
        value: row.value || row.count || '',
        type: row.type || row.activity_type || 'activity'
      };
    });

    // Sort by date (most recent first)
    items.sort((a, b) => b.date.getTime() - a.date.getTime());

    // Limit items
    return items.slice(0, spec.max_items || 10);
  };

  const formatLabel = (row: any): string => {
    // Try to build a meaningful label from the row data
    const keys = Object.keys(row).filter(k => k !== spec.time_column && k !== 'value' && k !== 'count');
    if (keys.length > 0) {
      return String(row[keys[0]]);
    }
    return 'Activity';
  };

  const formatDate = (date: Date): string => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const getIcon = (type: string) => {
    const iconMap: { [key: string]: React.ReactElement } = {
      'email': <FiMail />,
      'call': <FiPhone />,
      'meeting': <FiCalendar />,
      'completed': <FiCheckCircle />,
      'activity': <FiClock />,
    };
    return iconMap[type.toLowerCase()] || <FiClock />;
  };

  const timelineItems = processTimelineData();

  if (timelineItems.length === 0) {
    return (
      <div className="timeline-view empty">
        <p>No timeline data available</p>
      </div>
    );
  }

  return (
    <div className="timeline-view">
      <div className="timeline-header">
        <FiClock />
        <h3>Activity Timeline</h3>
        {spec.show_labels && (
          <span className="timeline-group-by">Grouped by {spec.group_by}</span>
        )}
      </div>

      <div className="timeline-list">
        {timelineItems.map((item, idx) => (
          <div key={idx} className="timeline-item">
            <div className="timeline-icon">
              {getIcon(item.type || 'activity')}
            </div>
            <div className="timeline-content">
              <div className="timeline-label">{item.label}</div>
              <div className="timeline-date">{formatDate(item.date)}</div>
              {item.value && (
                <div className="timeline-value">{item.value}</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TimelineView;
