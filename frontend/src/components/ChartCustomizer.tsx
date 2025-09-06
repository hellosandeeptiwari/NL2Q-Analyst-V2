import React, { useState } from 'react';
import { FiSettings, FiDownload, FiBarChart, FiPieChart, FiTrendingUp } from 'react-icons/fi';
import './ChartCustomizer.css';

interface ChartCustomizerProps {
  chartData: any;
  onChartTypeChange: (type: string) => void;
  onColorSchemeChange: (colors: string[]) => void;
  onDownloadChart: (format: 'png' | 'pdf' | 'svg') => void;
  currentChartType: string;
  currentColors: string[];
}

const ChartCustomizer: React.FC<ChartCustomizerProps> = ({
  chartData,
  onChartTypeChange,
  onColorSchemeChange,
  onDownloadChart,
  currentChartType,
  currentColors
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'type' | 'colors' | 'download'>('type');

  const chartTypes = [
    { id: 'bar', name: 'Bar Chart', icon: FiBarChart, description: 'Compare categories' },
    { id: 'line', name: 'Line Chart', icon: FiTrendingUp, description: 'Show trends over time' },
    { id: 'pie', name: 'Pie Chart', icon: FiPieChart, description: 'Show proportions' },
    { id: 'scatter', name: 'Scatter Plot', icon: FiBarChart, description: 'Show correlations' },
    { id: 'histogram', name: 'Histogram', icon: FiBarChart, description: 'Show distributions' }
  ];

  const colorSchemes = [
    {
      name: 'Pharma Blue',
      colors: ['#3B82F6', '#1E40AF', '#60A5FA', '#93C5FD', '#DBEAFE']
    },
    {
      name: 'Medical Green',
      colors: ['#10B981', '#059669', '#34D399', '#6EE7B7', '#D1FAE5']
    },
    {
      name: 'Clinical Red',
      colors: ['#EF4444', '#DC2626', '#F87171', '#FCA5A5', '#FEE2E2']
    },
    {
      name: 'Research Purple',
      colors: ['#8B5CF6', '#7C3AED', '#A78BFA', '#C4B5FD', '#EDE9FE']
    },
    {
      name: 'Analytics Orange',
      colors: ['#F59E0B', '#D97706', '#FBBF24', '#FCD34D', '#FEF3C7']
    },
    {
      name: 'Gradient Rainbow',
      colors: ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6']
    }
  ];

  const downloadFormats = [
    { id: 'png', name: 'PNG Image', description: 'High quality image format', icon: '🖼️' },
    { id: 'pdf', name: 'PDF Document', description: 'Printable document format', icon: '📄' },
    { id: 'svg', name: 'SVG Vector', description: 'Scalable vector format', icon: '📊' }
  ];

  return (
    <div className="chart-customizer">
      <button 
        className="customizer-toggle"
        onClick={() => setIsOpen(!isOpen)}
        title="Customize Chart"
      >
        <FiSettings size={16} />
        Customize
      </button>

      {isOpen && (
        <div className="customizer-panel">
          <div className="customizer-header">
            <h3>Chart Customization</h3>
            <button 
              className="close-btn"
              onClick={() => setIsOpen(false)}
            >
              ✕
            </button>
          </div>

          <div className="customizer-tabs">
            <button 
              className={`tab ${activeTab === 'type' ? 'active' : ''}`}
              onClick={() => setActiveTab('type')}
            >
              <FiBarChart size={14} />
              Chart Type
            </button>
            <button 
              className={`tab ${activeTab === 'colors' ? 'active' : ''}`}
              onClick={() => setActiveTab('colors')}
            >
              🎨
              Colors
            </button>
            <button 
              className={`tab ${activeTab === 'download' ? 'active' : ''}`}
              onClick={() => setActiveTab('download')}
            >
              <FiDownload size={14} />
              Download
            </button>
          </div>

          <div className="customizer-content">
            {activeTab === 'type' && (
              <div className="chart-types">
                <h4>Choose Chart Type</h4>
                <div className="chart-type-grid">
                  {chartTypes.map((type) => (
                    <button
                      key={type.id}
                      className={`chart-type-option ${currentChartType === type.id ? 'selected' : ''}`}
                      onClick={() => onChartTypeChange(type.id)}
                    >
                      <type.icon size={20} />
                      <span className="type-name">{type.name}</span>
                      <span className="type-description">{type.description}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'colors' && (
              <div className="color-schemes">
                <h4>Color Scheme</h4>
                <div className="color-scheme-grid">
                  {colorSchemes.map((scheme, index) => (
                    <button
                      key={index}
                      className={`color-scheme-option ${
                        JSON.stringify(currentColors) === JSON.stringify(scheme.colors) ? 'selected' : ''
                      }`}
                      onClick={() => onColorSchemeChange(scheme.colors)}
                    >
                      <div className="scheme-name">{scheme.name}</div>
                      <div className="color-preview">
                        {scheme.colors.map((color, colorIndex) => (
                          <div
                            key={colorIndex}
                            className="color-swatch"
                            style={{ backgroundColor: color }}
                          />
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'download' && (
              <div className="download-options">
                <h4>Download Chart</h4>
                <div className="download-format-grid">
                  {downloadFormats.map((format) => (
                    <button
                      key={format.id}
                      className="download-format-option"
                      onClick={() => onDownloadChart(format.id as 'png' | 'pdf' | 'svg')}
                    >
                      <span className="format-icon">{format.icon}</span>
                      <div className="format-info">
                        <span className="format-name">{format.name}</span>
                        <span className="format-description">{format.description}</span>
                      </div>
                    </button>
                  ))}
                </div>
                
                <div className="download-settings">
                  <h5>Download Settings</h5>
                  <div className="setting-group">
                    <label>
                      <input type="checkbox" defaultChecked />
                      Include title and labels
                    </label>
                    <label>
                      <input type="checkbox" defaultChecked />
                      High resolution (300 DPI)
                    </label>
                    <label>
                      <input type="checkbox" />
                      Transparent background
                    </label>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ChartCustomizer;
