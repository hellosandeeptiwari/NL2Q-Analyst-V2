// Chart Download Utilities

// Simple download function that works without external dependencies
export const downloadChartAsPNG = (chartElement: HTMLElement, filename: string = 'chart') => {
  try {
    // Try to find canvas element within the chart
    const canvas = chartElement.querySelector('canvas');
    
    if (canvas) {
      // If we find a canvas, download it directly
      const link = document.createElement('a');
      link.download = `${filename}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } else {
      // Fallback: use html2canvas if available
      if (typeof window !== 'undefined' && (window as any).html2canvas) {
        (window as any).html2canvas(chartElement, {
          background: '#ffffff',
          scale: 2,
          useCORS: true
        }).then((canvas: HTMLCanvasElement) => {
          const link = document.createElement('a');
          link.download = `${filename}.png`;
          link.href = canvas.toDataURL('image/png');
          link.click();
        });
      } else {
        fallbackDownload(chartElement, filename, 'png');
      }
    }
  } catch (error) {
    console.error('PNG download failed:', error);
    fallbackDownload(chartElement, filename, 'png');
  }
};

export const downloadChartAsPDF = (chartElement: HTMLElement, filename: string = 'chart') => {
  try {
    // Try to find canvas element within the chart
    const canvas = chartElement.querySelector('canvas');
    
    if (canvas && typeof window !== 'undefined' && (window as any).jsPDF) {
      const jsPDF = (window as any).jsPDF;
      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'mm',
        format: 'a4'
      });
      
      const imgWidth = 280;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      
      pdf.addImage(
        canvas.toDataURL('image/png'),
        'PNG',
        10,
        10,
        imgWidth,
        imgHeight
      );
      
      pdf.save(`${filename}.pdf`);
    } else {
      // Fallback to PNG if PDF generation is not available
      downloadChartAsPNG(chartElement, filename);
    }
  } catch (error) {
    console.error('PDF download failed:', error);
    fallbackDownload(chartElement, filename, 'pdf');
  }
};

export const downloadChartAsSVG = (chartElement: HTMLElement, filename: string = 'chart') => {
  try {
    // Try to find SVG element within the chart
    const svgElement = chartElement.querySelector('svg');
    
    if (svgElement) {
      const svgData = new XMLSerializer().serializeToString(svgElement);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
      const link = document.createElement('a');
      link.download = `${filename}.svg`;
      link.href = URL.createObjectURL(svgBlob);
      link.click();
      URL.revokeObjectURL(link.href);
    } else {
      // Fallback to PNG if no SVG found
      downloadChartAsPNG(chartElement, filename);
    }
  } catch (error) {
    console.error('SVG download failed:', error);
    fallbackDownload(chartElement, filename, 'svg');
  }
};

const fallbackDownload = (chartElement: HTMLElement, filename: string, format: string) => {
  // Simple fallback using canvas.toDataURL for older browsers
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    const rect = chartElement.getBoundingClientRect();
    canvas.width = rect.width * 2; // High resolution
    canvas.height = rect.height * 2;
    
    // Scale context for high resolution
    ctx.scale(2, 2);
    
    // Fill white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    // Add fallback text
    ctx.fillStyle = '#333333';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Chart Export', rect.width / 2, rect.height / 2);
    ctx.fillText(`(${format.toUpperCase()} format)`, rect.width / 2, rect.height / 2 + 25);
    
    // Download
    const link = document.createElement('a');
    link.download = `${filename}.${format === 'pdf' ? 'png' : format}`;
    link.href = canvas.toDataURL('image/png');
    link.click();
    
  } catch (error) {
    console.error('Fallback download failed:', error);
    alert('Download failed. Please try again or contact support.');
  }
};

// Chart customization utilities
export const applyChartColors = (chartData: any, colors: string[]) => {
  if (!chartData || !colors) return chartData;
  
  try {
    // For Plotly charts
    if (chartData.data && Array.isArray(chartData.data)) {
      return {
        ...chartData,
        data: chartData.data.map((trace: any, index: number) => ({
          ...trace,
          marker: {
            ...trace.marker,
            color: colors[index % colors.length]
          }
        }))
      };
    }
    
    // For other chart libraries, return as-is with colors for manual application
    return {
      ...chartData,
      colors: colors
    };
  } catch (error) {
    console.error('Error applying chart colors:', error);
    return chartData;
  }
};

export const convertChartType = (chartData: any, newType: string) => {
  if (!chartData) return chartData;
  
  try {
    // For Plotly charts - modify the data traces
    if (chartData.data && Array.isArray(chartData.data)) {
      const convertedData = chartData.data.map((trace: any) => {
        const newTrace = { ...trace };
        
        // Map our UI chart types to Plotly types
        switch (newType) {
          case 'bar':
            newTrace.type = 'bar';
            // Remove line-specific properties
            delete newTrace.mode;
            break;
            
          case 'line':
            newTrace.type = 'scatter';
            newTrace.mode = 'lines+markers';
            break;
            
          case 'pie':
            newTrace.type = 'pie';
            // For pie charts, we need values and labels
            if (newTrace.x && newTrace.y) {
              newTrace.values = newTrace.y;
              newTrace.labels = newTrace.x;
              delete newTrace.x;
              delete newTrace.y;
            }
            delete newTrace.mode;
            break;
            
          case 'scatter':
            newTrace.type = 'scatter';
            newTrace.mode = 'markers';
            break;
            
          case 'histogram':
            newTrace.type = 'histogram';
            // For histogram, use x data as the values
            if (newTrace.y && !newTrace.x) {
              newTrace.x = newTrace.y;
              delete newTrace.y;
            }
            delete newTrace.mode;
            break;
            
          default:
            newTrace.type = 'bar'; // Default fallback
        }
        
        return newTrace;
      });
      
      // Update layout based on chart type
      const newLayout = { ...chartData.layout };
      
      // Adjust layout for specific chart types
      if (newType === 'pie') {
        // Pie charts don't need axes
        delete newLayout.xaxis;
        delete newLayout.yaxis;
      } else {
        // Ensure axes exist for non-pie charts
        if (!newLayout.xaxis) newLayout.xaxis = {};
        if (!newLayout.yaxis) newLayout.yaxis = {};
      }
      
      return {
        ...chartData,
        data: convertedData,
        layout: newLayout
      };
    }
    
    // For other chart types, return as-is with type metadata
    return {
      ...chartData,
      type: newType
    };
  } catch (error) {
    console.error('Error converting chart type:', error);
    return chartData;
  }
};

const mapChartType = (type: string): string => {
  const typeMap: { [key: string]: string } = {
    'bar': 'bar',
    'line': 'scatter',
    'pie': 'pie',
    'scatter': 'scatter',
    'histogram': 'histogram'
  };
  
  return typeMap[type] || 'bar';
};

// Install required dependencies function
export const installChartDependencies = () => {
  console.log('Installing chart dependencies...');
  console.log('Run: npm install html2canvas jspdf');
  
  // Show user-friendly message if dependencies are missing
  const showInstallMessage = () => {
    const message = `
Chart download requires additional packages:
- html2canvas (for PNG exports)
- jsPDF (for PDF exports)

Run: npm install html2canvas jspdf

Then restart your development server.
    `;
    console.warn(message);
    alert('Chart download packages not installed. Check console for instructions.');
  };
  
  return showInstallMessage;
};

// Helper function to get optimal chart type based on data structure
export const suggestChartType = (data: any) => {
  if (!data || !data.data || !Array.isArray(data.data)) {
    return 'bar';
  }
  
  const firstTrace = data.data[0];
  if (!firstTrace) return 'bar';
  
  // If data has categorical labels and numeric values, suggest bar/pie
  if (firstTrace.x && firstTrace.y) {
    const xValues = Array.isArray(firstTrace.x) ? firstTrace.x : [];
    const yValues = Array.isArray(firstTrace.y) ? firstTrace.y : [];
    
    // If x values are strings (categories) and y values are numbers
    const hasCategories = xValues.some((val: any) => typeof val === 'string');
    const hasNumbers = yValues.some((val: any) => typeof val === 'number');
    
    if (hasCategories && hasNumbers) {
      // For small datasets, pie chart might work well
      if (xValues.length <= 8) return 'pie';
      return 'bar';
    }
    
    // If both x and y are numeric, scatter plot is good
    const xNumeric = xValues.every((val: any) => typeof val === 'number');
    const yNumeric = yValues.every((val: any) => typeof val === 'number');
    
    if (xNumeric && yNumeric) {
      return 'scatter';
    }
  }
  
  return 'bar'; // Safe default
};

// Helper to validate if chart type is compatible with data
export const isChartTypeCompatible = (chartData: any, chartType: string): boolean => {
  if (!chartData || !chartData.data || !Array.isArray(chartData.data)) {
    return false;
  }
  
  const firstTrace = chartData.data[0];
  if (!firstTrace) return false;
  
  switch (chartType) {
    case 'pie':
      // Pie charts need either x,y data or values,labels
      return (firstTrace.x && firstTrace.y) || (firstTrace.values && firstTrace.labels);
      
    case 'scatter':
    case 'line':
      // Scatter and line charts need x,y coordinates
      return firstTrace.x && firstTrace.y;
      
    case 'histogram':
      // Histograms need at least x or y data
      return firstTrace.x || firstTrace.y;
      
    case 'bar':
    default:
      // Bar charts are most flexible
      return true;
  }
};
