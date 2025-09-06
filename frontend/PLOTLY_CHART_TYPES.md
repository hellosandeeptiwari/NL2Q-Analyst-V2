# Plotly Chart Type Changes

## How Chart Types Work in Plotly

Plotly charts are defined by **traces** in the `data` array. Each trace has a `type` property that determines how the data is visualized.

## Available Chart Types

### 1. Bar Chart (`type: 'bar'`)
```javascript
const barChart = {
  data: [{
    x: ['Product A', 'Product B', 'Product C'],
    y: [20, 14, 23],
    type: 'bar'
  }],
  layout: {
    title: 'Sales by Product'
  }
};
```

### 2. Line Chart (`type: 'scatter'` with `mode: 'lines+markers'`)
```javascript
const lineChart = {
  data: [{
    x: ['Jan', 'Feb', 'Mar', 'Apr'],
    y: [20, 14, 23, 25],
    type: 'scatter',
    mode: 'lines+markers'
  }],
  layout: {
    title: 'Monthly Trend'
  }
};
```

### 3. Scatter Plot (`type: 'scatter'` with `mode: 'markers'`)
```javascript
const scatterChart = {
  data: [{
    x: [1, 2, 3, 4, 5],
    y: [10, 11, 12, 13, 14],
    type: 'scatter',
    mode: 'markers'
  }],
  layout: {
    title: 'Correlation Analysis'
  }
};
```

### 4. Pie Chart (`type: 'pie'`)
```javascript
const pieChart = {
  data: [{
    values: [16, 15, 12, 6, 5, 4, 42],
    labels: ['US', 'China', 'European Union', 'Russian Federation', 'Brazil', 'India', 'Rest of World'],
    type: 'pie'
  }],
  layout: {
    title: 'Market Share'
  }
};
```

### 5. Histogram (`type: 'histogram'`)
```javascript
const histogramChart = {
  data: [{
    x: [1, 2, 1, 3, 3, 2, 4, 5, 5, 4, 3, 2, 1],
    type: 'histogram'
  }],
  layout: {
    title: 'Distribution'
  }
};
```

## How Our Chart Converter Works

Our `convertChartType` function in `chartUtils.ts` handles the conversion between chart types:

1. **Bar to Line**: Changes `type: 'bar'` to `type: 'scatter'` and adds `mode: 'lines+markers'`
2. **Bar to Pie**: Changes structure from x/y arrays to values/labels format
3. **Line to Scatter**: Keeps `type: 'scatter'` but changes `mode` to `'markers'`
4. **Any to Histogram**: Uses the y-values as x-values for distribution

## Usage in the UI

When you click "Customize" on a chart:

1. The ChartCustomizer component shows available chart types
2. When you select a new type, it calls `onChartTypeChange`
3. This function uses `convertChartType` to transform the Plotly data structure
4. The chart re-renders with the new type

## Data Compatibility

- **Bar/Line/Scatter**: Need x and y arrays
- **Pie**: Needs values and labels (converted from x/y)
- **Histogram**: Needs just x array (distribution of values)

The system automatically handles data transformation between these formats!
