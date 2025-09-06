"""
Plotly Visualization Generator
Generates Plotly specifications for data visualization
"""
import json
from typing import Dict, List, Any, Optional
import pandas as pd

class PlotlyGenerator:
    """Generate Plotly chart specifications from data and query context"""
    
    def __init__(self):
        pass
    
    def generate_plotly_spec(self, data: List[Dict], query: str, sql: str = "") -> Dict[str, Any]:
        """
        Generate a Plotly specification based on data and query context
        
        Args:
            data: List of dictionaries containing the query results
            query: Natural language query
            sql: SQL query used to generate the data
            
        Returns:
            Dictionary containing Plotly specification
        """
        if not data:
            return {}
        
        try:
            df = pd.DataFrame(data)
            
            # Analyze query to determine best chart type
            chart_recommendation = self._recommend_chart_type(query, df)
            
            # Generate Plotly spec based on recommendation
            plotly_spec = self._create_plotly_spec(df, chart_recommendation, query)
            
            return plotly_spec
            
        except Exception as e:
            print(f"❌ Failed to generate Plotly spec: {e}")
            return {}
    
    def _recommend_chart_type(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Recommend the best chart type based on query and data structure"""
        query_lower = query.lower()
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Chart type determination logic
        if any(word in query_lower for word in ['frequency', 'count', 'distribution', 'recommended']):
            if len(text_cols) >= 1 and len(numeric_cols) >= 1:
                return {
                    "type": "bar",
                    "x_col": text_cols[0],
                    "y_col": numeric_cols[0] if numeric_cols else "count",
                    "reasoning": "Bar chart for frequency/count analysis"
                }
            elif len(text_cols) >= 1:
                return {
                    "type": "bar", 
                    "x_col": text_cols[0],
                    "y_col": "count",
                    "reasoning": "Bar chart with count for categorical data"
                }
        
        elif any(word in query_lower for word in ['trend', 'time', 'over time', 'daily', 'monthly']):
            if len(df.columns) >= 2:
                return {
                    "type": "line",
                    "x_col": columns[0],
                    "y_col": columns[1] if len(columns) > 1 else columns[0],
                    "reasoning": "Line chart for trend analysis"
                }
        
        elif any(word in query_lower for word in ['scatter', 'correlation', 'relationship']):
            if len(numeric_cols) >= 2:
                return {
                    "type": "scatter",
                    "x_col": numeric_cols[0],
                    "y_col": numeric_cols[1],
                    "reasoning": "Scatter plot for correlation analysis"
                }
        
        elif any(word in query_lower for word in ['pie', 'proportion', 'percentage', 'share']):
            if len(text_cols) >= 1:
                return {
                    "type": "pie",
                    "labels_col": text_cols[0],
                    "values_col": numeric_cols[0] if numeric_cols else "count",
                    "reasoning": "Pie chart for proportional data"
                }
        
        # Default fallback logic
        if len(text_cols) >= 1 and len(numeric_cols) >= 1:
            return {
                "type": "bar",
                "x_col": text_cols[0],
                "y_col": numeric_cols[0],
                "reasoning": "Default bar chart for mixed data types"
            }
        elif len(numeric_cols) >= 2:
            return {
                "type": "scatter",
                "x_col": numeric_cols[0], 
                "y_col": numeric_cols[1],
                "reasoning": "Default scatter plot for numeric data"
            }
        else:
            return {
                "type": "table",
                "reasoning": "Table view for complex or non-visual data"
            }
    
    def _create_plotly_spec(self, df: pd.DataFrame, chart_rec: Dict, query: str) -> Dict[str, Any]:
        """Create the actual Plotly specification"""
        chart_type = chart_rec.get("type", "table")
        
        # Generate title from query
        title = self._generate_title(query, chart_type)
        
        # Base plotly configuration
        config = {
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "displaylogo": False,
            "responsive": True
        }
        
        layout = {
            "title": {"text": title, "x": 0.5},
            "margin": {"l": 50, "r": 50, "t": 60, "b": 50},
            "height": 400,
            "showlegend": True
        }
        
        if chart_type == "bar":
            return self._create_bar_chart(df, chart_rec, layout, config)
        elif chart_type == "line":
            return self._create_line_chart(df, chart_rec, layout, config)
        elif chart_type == "scatter":
            return self._create_scatter_chart(df, chart_rec, layout, config)
        elif chart_type == "pie":
            return self._create_pie_chart(df, chart_rec, layout, config)
        else:
            return self._create_table_view(df, layout, config)
    
    def _create_bar_chart(self, df: pd.DataFrame, chart_rec: Dict, layout: Dict, config: Dict) -> Dict:
        """Create bar chart specification"""
        x_col = chart_rec.get("x_col")
        y_col = chart_rec.get("y_col")
        
        # If y_col is "count", we need to aggregate
        if y_col == "count" and x_col in df.columns:
            value_counts = df[x_col].value_counts()
            x_data = value_counts.index.tolist()
            y_data = value_counts.values.tolist()
        else:
            x_data = df[x_col].tolist() if x_col in df.columns else list(range(len(df)))
            y_data = df[y_col].tolist() if y_col in df.columns else [1] * len(df)
        
        # Limit data points for better visualization
        if len(x_data) > 20:
            x_data = x_data[:20]
            y_data = y_data[:20]
        
        layout.update({
            "xaxis": {"title": x_col if x_col else "Categories"},
            "yaxis": {"title": y_col if y_col else "Count"}
        })
        
        return {
            "data": [{
                "type": "bar",
                "x": x_data,
                "y": y_data,
                "marker": {"color": "#1f77b4"},
                "name": y_col if y_col else "Count"
            }],
            "layout": layout,
            "config": config
        }
    
    def _create_line_chart(self, df: pd.DataFrame, chart_rec: Dict, layout: Dict, config: Dict) -> Dict:
        """Create line chart specification"""
        x_col = chart_rec.get("x_col")
        y_col = chart_rec.get("y_col")
        
        x_data = df[x_col].tolist() if x_col in df.columns else list(range(len(df)))
        y_data = df[y_col].tolist() if y_col in df.columns else df.iloc[:, 0].tolist()
        
        layout.update({
            "xaxis": {"title": x_col if x_col else "X-axis"},
            "yaxis": {"title": y_col if y_col else "Y-axis"}
        })
        
        return {
            "data": [{
                "type": "scatter",
                "mode": "lines+markers",
                "x": x_data,
                "y": y_data,
                "line": {"color": "#1f77b4"},
                "name": y_col if y_col else "Values"
            }],
            "layout": layout,
            "config": config
        }
    
    def _create_scatter_chart(self, df: pd.DataFrame, chart_rec: Dict, layout: Dict, config: Dict) -> Dict:
        """Create scatter chart specification"""
        x_col = chart_rec.get("x_col")
        y_col = chart_rec.get("y_col")
        
        x_data = df[x_col].tolist() if x_col in df.columns else list(range(len(df)))
        y_data = df[y_col].tolist() if y_col in df.columns else df.iloc[:, -1].tolist()
        
        layout.update({
            "xaxis": {"title": x_col if x_col else "X-axis"},
            "yaxis": {"title": y_col if y_col else "Y-axis"}
        })
        
        return {
            "data": [{
                "type": "scatter",
                "mode": "markers",
                "x": x_data,
                "y": y_data,
                "marker": {"color": "#1f77b4", "size": 8},
                "name": "Data Points"
            }],
            "layout": layout,
            "config": config
        }
    
    def _create_pie_chart(self, df: pd.DataFrame, chart_rec: Dict, layout: Dict, config: Dict) -> Dict:
        """Create pie chart specification"""
        labels_col = chart_rec.get("labels_col")
        values_col = chart_rec.get("values_col")
        
        if values_col == "count" and labels_col in df.columns:
            value_counts = df[labels_col].value_counts()
            labels = value_counts.index.tolist()
            values = value_counts.values.tolist()
        else:
            labels = df[labels_col].tolist() if labels_col in df.columns else [f"Item {i}" for i in range(len(df))]
            values = df[values_col].tolist() if values_col in df.columns else [1] * len(df)
        
        # Limit to top 10 slices
        if len(labels) > 10:
            labels = labels[:10]
            values = values[:10]
        
        return {
            "data": [{
                "type": "pie",
                "labels": labels,
                "values": values,
                "hole": 0.3
            }],
            "layout": layout,
            "config": config
        }
    
    def _create_table_view(self, df: pd.DataFrame, layout: Dict, config: Dict) -> Dict:
        """Create table view specification"""
        # Limit rows for display
        display_df = df.head(100)
        
        return {
            "data": [{
                "type": "table",
                "header": {
                    "values": list(display_df.columns),
                    "fill": {"color": "#f0f0f0"},
                    "align": "left"
                },
                "cells": {
                    "values": [display_df[col].tolist() for col in display_df.columns],
                    "fill": {"color": "#ffffff"},
                    "align": "left"
                }
            }],
            "layout": layout,
            "config": config
        }
    
    def _generate_title(self, query: str, chart_type: str) -> str:
        """Generate an appropriate title for the chart"""
        # Extract key terms from query
        key_terms = []
        important_words = ["recommended", "messages", "NBA", "marketing", "frequency", "count", "analysis"]
        
        for word in important_words:
            if word.lower() in query.lower():
                key_terms.append(word)
        
        if key_terms:
            return f"{' '.join(key_terms).title()} - {chart_type.title()} Chart"
        else:
            return f"Data Visualization - {chart_type.title()} Chart"
