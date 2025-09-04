"""
Chart Builder - Intelligent visualization generation
Automatically selects optimal chart types based on data characteristics
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
from collections import Counter

@dataclass
class ChartRecommendation:
    chart_type: str
    confidence: float
    reasoning: str
    config: Dict[str, Any]
    alternative_charts: List[Dict[str, Any]] = None

@dataclass
class ChartSpec:
    chart_type: str
    title: str
    x_axis: Dict[str, Any]
    y_axis: Dict[str, Any]
    data_config: Dict[str, Any]
    layout_config: Dict[str, Any]
    interactive_features: List[str]
    accessibility_config: Dict[str, Any]

class ChartBuilder:
    """
    Intelligent chart builder that analyzes data and recommends optimal visualizations
    """
    
    def __init__(self):
        # Chart type mappings based on data characteristics
        self.chart_rules = {
            'bar': {
                'conditions': ['categorical_x', 'numeric_y', 'moderate_categories'],
                'use_cases': ['comparing categories', 'showing rankings', 'discrete comparisons']
            },
            'line': {
                'conditions': ['temporal_x', 'numeric_y', 'continuous_data'],
                'use_cases': ['trends over time', 'continuous relationships', 'progression']
            },
            'scatter': {
                'conditions': ['numeric_x', 'numeric_y', 'correlation_analysis'],
                'use_cases': ['relationships', 'correlations', 'outlier detection']
            },
            'pie': {
                'conditions': ['categorical_x', 'numeric_y', 'few_categories', 'parts_of_whole'],
                'use_cases': ['proportions', 'composition', 'market share']
            },
            'histogram': {
                'conditions': ['numeric_x', 'distribution_analysis'],
                'use_cases': ['data distribution', 'frequency analysis', 'statistical overview']
            },
            'box': {
                'conditions': ['categorical_x', 'numeric_y', 'distribution_comparison'],
                'use_cases': ['distribution comparison', 'outlier detection', 'statistical summary']
            },
            'heatmap': {
                'conditions': ['matrix_data', 'correlation_matrix', 'density_visualization'],
                'use_cases': ['correlation analysis', 'density maps', 'pattern recognition']
            },
            'treemap': {
                'conditions': ['hierarchical_data', 'size_comparison'],
                'use_cases': ['hierarchical proportions', 'nested categories', 'size visualization']
            }
        }
        
        # Color palettes for different use cases
        self.color_palettes = {
            'categorical': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd'],
            'diverging': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a'],
            'temporal': ['#1f77b4', '#ff7f0e', '#2ca02c'],
            'performance': ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Yellow, Red
        }
    
    async def analyze_and_recommend(
        self, 
        data: List[Dict[str, Any]], 
        query_context: Dict[str, Any]
    ) -> ChartRecommendation:
        """
        Analyze data characteristics and recommend optimal chart type
        """
        
        if not data:
            return ChartRecommendation(
                chart_type='table',
                confidence=1.0,
                reasoning="No data available for visualization",
                config={"message": "No data to display"}
            )
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        
        # Analyze data characteristics
        data_profile = await self._profile_data(df, query_context)
        
        # Get chart recommendations
        recommendations = await self._get_chart_recommendations(data_profile)
        
        # Select best recommendation
        best_recommendation = recommendations[0] if recommendations else None
        
        if not best_recommendation:
            return ChartRecommendation(
                chart_type='table',
                confidence=0.5,
                reasoning="Could not determine optimal chart type",
                config=await self._create_table_config(df)
            )
        
        # Generate chart configuration
        chart_config = await self._create_chart_config(
            df, 
            best_recommendation['chart_type'], 
            data_profile,
            query_context
        )
        
        return ChartRecommendation(
            chart_type=best_recommendation['chart_type'],
            confidence=best_recommendation['confidence'],
            reasoning=best_recommendation['reasoning'],
            config=chart_config,
            alternative_charts=recommendations[1:3]  # Top 2 alternatives
        )
    
    async def create_chart_spec(
        self, 
        data: List[Dict[str, Any]], 
        chart_type: str,
        query_context: Dict[str, Any],
        custom_config: Optional[Dict[str, Any]] = None
    ) -> ChartSpec:
        """
        Create detailed chart specification for rendering
        """
        
        df = pd.DataFrame(data)
        data_profile = await self._profile_data(df, query_context)
        
        # Generate title from query context
        title = await self._generate_chart_title(query_context, data_profile)
        
        # Configure axes
        x_axis_config = await self._configure_x_axis(df, data_profile, chart_type)
        y_axis_config = await self._configure_y_axis(df, data_profile, chart_type)
        
        # Configure data representation
        data_config = await self._configure_data_representation(df, chart_type, data_profile)
        
        # Configure layout
        layout_config = await self._configure_layout(chart_type, data_profile, custom_config)
        
        # Configure interactivity
        interactive_features = await self._configure_interactivity(chart_type, data_profile)
        
        # Configure accessibility
        accessibility_config = await self._configure_accessibility(chart_type, data_profile)
        
        return ChartSpec(
            chart_type=chart_type,
            title=title,
            x_axis=x_axis_config,
            y_axis=y_axis_config,
            data_config=data_config,
            layout_config=layout_config,
            interactive_features=interactive_features,
            accessibility_config=accessibility_config
        )
    
    async def _profile_data(self, df: pd.DataFrame, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Profile data to understand its characteristics
        """
        
        profile = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': {},
            'relationships': {},
            'data_quality': {},
            'query_intent': query_context.get('intent', 'unknown')
        }
        
        # Analyze each column
        for col in df.columns:
            col_data = df[col]
            col_profile = {
                'dtype': str(col_data.dtype),
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
                'cardinality': 'high' if col_data.nunique() > len(df) * 0.8 else 'low'
            }
            
            # Detect data type semantics
            if pd.api.types.is_numeric_dtype(col_data):
                col_profile['semantic_type'] = 'numeric'
                col_profile['min'] = col_data.min()
                col_profile['max'] = col_data.max()
                col_profile['mean'] = col_data.mean()
                
                # Check if it's a count/id field
                if all(col_data >= 0) and all(col_data == col_data.astype(int)):
                    if 'id' in col.lower() or col_data.nunique() == len(df):
                        col_profile['semantic_type'] = 'identifier'
                    elif col_data.min() >= 0:
                        col_profile['semantic_type'] = 'count'
                        
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_profile['semantic_type'] = 'temporal'
                col_profile['min_date'] = col_data.min()
                col_profile['max_date'] = col_data.max()
                col_profile['date_range_days'] = (col_data.max() - col_data.min()).days
                
            else:
                col_profile['semantic_type'] = 'categorical'
                col_profile['categories'] = col_data.value_counts().to_dict()
                
                # Check if it's actually a temporal string
                if await self._is_temporal_string(col_data):
                    col_profile['semantic_type'] = 'temporal_string'
            
            profile['columns'][col] = col_profile
        
        # Analyze relationships between columns
        if len(df.columns) >= 2:
            profile['relationships'] = await self._analyze_relationships(df)
        
        # Assess data quality
        profile['data_quality'] = await self._assess_data_quality(df)
        
        return profile
    
    async def _get_chart_recommendations(self, data_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate chart recommendations based on data profile
        """
        
        recommendations = []
        columns = data_profile['columns']
        
        # Get column types
        numeric_cols = [col for col, info in columns.items() if info['semantic_type'] == 'numeric']
        categorical_cols = [col for col, info in columns.items() if info['semantic_type'] == 'categorical']
        temporal_cols = [col for col, info in columns.items() if info['semantic_type'] in ['temporal', 'temporal_string']]
        
        row_count = data_profile['row_count']
        
        # Rule-based recommendations
        
        # Time series data
        if temporal_cols and numeric_cols:
            recommendations.append({
                'chart_type': 'line',
                'confidence': 0.9,
                'reasoning': f"Time series data with {temporal_cols[0]} (temporal) and {numeric_cols[0]} (numeric) - ideal for line chart"
            })
        
        # Categorical vs Numeric
        if categorical_cols and numeric_cols:
            unique_categories = columns[categorical_cols[0]]['unique_count']
            
            if unique_categories <= 10:
                if unique_categories <= 5 and len(numeric_cols) == 1:
                    # Consider pie chart for parts-of-whole
                    total_check = sum(data_profile.get('sample_data', {}).get(numeric_cols[0], [0]))
                    if total_check > 0:  # Positive values suggest parts of whole
                        recommendations.append({
                            'chart_type': 'pie',
                            'confidence': 0.8,
                            'reasoning': f"Few categories ({unique_categories}) with positive numeric values - good for pie chart"
                        })
                
                recommendations.append({
                    'chart_type': 'bar',
                    'confidence': 0.85,
                    'reasoning': f"Categorical data ({categorical_cols[0]}) with numeric values - excellent for bar chart"
                })
            else:
                recommendations.append({
                    'chart_type': 'bar',
                    'confidence': 0.7,
                    'reasoning': f"Many categories ({unique_categories}) - bar chart with scrolling"
                })
        
        # Two numeric columns
        if len(numeric_cols) >= 2:
            recommendations.append({
                'chart_type': 'scatter',
                'confidence': 0.8,
                'reasoning': f"Two numeric columns ({numeric_cols[0]}, {numeric_cols[1]}) - good for correlation analysis"
            })
        
        # Single numeric column
        if len(numeric_cols) == 1 and not categorical_cols and not temporal_cols:
            recommendations.append({
                'chart_type': 'histogram',
                'confidence': 0.75,
                'reasoning': f"Single numeric column ({numeric_cols[0]}) - histogram shows distribution"
            })
        
        # Large dataset considerations
        if row_count > 1000:
            recommendations.append({
                'chart_type': 'heatmap',
                'confidence': 0.6,
                'reasoning': f"Large dataset ({row_count} rows) - heatmap can show patterns"
            })
        
        # Default to table for complex data
        if not recommendations:
            recommendations.append({
                'chart_type': 'table',
                'confidence': 0.5,
                'reasoning': "Complex data structure - table provides comprehensive view"
            })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations
    
    async def _create_chart_config(
        self, 
        df: pd.DataFrame, 
        chart_type: str, 
        data_profile: Dict[str, Any],
        query_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create chart configuration based on chart type and data
        """
        
        base_config = {
            'type': chart_type,
            'responsive': True,
            'data': df.to_dict('records'),
            'height': 400,
            'margin': {'t': 40, 'l': 60, 'r': 20, 'b': 60}
        }
        
        columns = list(df.columns)
        
        if chart_type == 'bar':
            config = await self._create_bar_config(df, data_profile, base_config)
        elif chart_type == 'line':
            config = await self._create_line_config(df, data_profile, base_config)
        elif chart_type == 'scatter':
            config = await self._create_scatter_config(df, data_profile, base_config)
        elif chart_type == 'pie':
            config = await self._create_pie_config(df, data_profile, base_config)
        elif chart_type == 'histogram':
            config = await self._create_histogram_config(df, data_profile, base_config)
        elif chart_type == 'heatmap':
            config = await self._create_heatmap_config(df, data_profile, base_config)
        else:
            config = await self._create_table_config(df)
        
        # Add common enhancements
        config.update({
            'title': await self._generate_chart_title(query_context, data_profile),
            'colors': await self._select_color_palette(chart_type, data_profile),
            'annotations': await self._generate_annotations(df, chart_type, data_profile)
        })
        
        return config
    
    async def _create_bar_config(self, df: pd.DataFrame, data_profile: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create bar chart configuration"""
        
        columns = list(df.columns)
        categorical_cols = [col for col, info in data_profile['columns'].items() if info['semantic_type'] == 'categorical']
        numeric_cols = [col for col, info in data_profile['columns'].items() if info['semantic_type'] == 'numeric']
        
        x_col = categorical_cols[0] if categorical_cols else columns[0]
        y_col = numeric_cols[0] if numeric_cols else columns[1] if len(columns) > 1 else columns[0]
        
        config = base_config.copy()
        config.update({
            'x_column': x_col,
            'y_column': y_col,
            'orientation': 'vertical',
            'show_values': len(df) <= 20,  # Show values on bars if not too many
            'sort_by': y_col,
            'sort_order': 'desc'
        })
        
        return config
    
    async def _create_line_config(self, df: pd.DataFrame, data_profile: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create line chart configuration"""
        
        columns = list(df.columns)
        temporal_cols = [col for col, info in data_profile['columns'].items() if info['semantic_type'] in ['temporal', 'temporal_string']]
        numeric_cols = [col for col, info in data_profile['columns'].items() if info['semantic_type'] == 'numeric']
        
        x_col = temporal_cols[0] if temporal_cols else columns[0]
        y_col = numeric_cols[0] if numeric_cols else columns[1] if len(columns) > 1 else columns[0]
        
        config = base_config.copy()
        config.update({
            'x_column': x_col,
            'y_column': y_col,
            'mode': 'lines+markers',
            'line_smoothing': 0.3,
            'show_trend': len(df) > 10,
            'fill_area': False
        })
        
        return config
    
    async def _create_scatter_config(self, df: pd.DataFrame, data_profile: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create scatter plot configuration"""
        
        columns = list(df.columns)
        numeric_cols = [col for col, info in data_profile['columns'].items() if info['semantic_type'] == 'numeric']
        
        x_col = numeric_cols[0] if len(numeric_cols) > 0 else columns[0]
        y_col = numeric_cols[1] if len(numeric_cols) > 1 else columns[1] if len(columns) > 1 else columns[0]
        
        config = base_config.copy()
        config.update({
            'x_column': x_col,
            'y_column': y_col,
            'size_column': None,
            'color_column': None,
            'show_trend_line': True,
            'opacity': 0.7
        })
        
        return config
    
    async def _create_pie_config(self, df: pd.DataFrame, data_profile: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create pie chart configuration"""
        
        columns = list(df.columns)
        categorical_cols = [col for col, info in data_profile['columns'].items() if info['semantic_type'] == 'categorical']
        numeric_cols = [col for col, info in data_profile['columns'].items() if info['semantic_type'] == 'numeric']
        
        label_col = categorical_cols[0] if categorical_cols else columns[0]
        value_col = numeric_cols[0] if numeric_cols else columns[1] if len(columns) > 1 else columns[0]
        
        config = base_config.copy()
        config.update({
            'label_column': label_col,
            'value_column': value_col,
            'show_percentages': True,
            'donut_mode': False,
            'pull_largest': True
        })
        
        return config
    
    async def _create_histogram_config(self, df: pd.DataFrame, data_profile: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create histogram configuration"""
        
        numeric_cols = [col for col, info in data_profile['columns'].items() if info['semantic_type'] == 'numeric']
        x_col = numeric_cols[0] if numeric_cols else list(df.columns)[0]
        
        config = base_config.copy()
        config.update({
            'x_column': x_col,
            'bins': 'auto',
            'show_distribution_curve': True,
            'show_statistics': True
        })
        
        return config
    
    async def _create_heatmap_config(self, df: pd.DataFrame, data_profile: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create heatmap configuration"""
        
        config = base_config.copy()
        config.update({
            'z_values': 'correlation',  # or 'values'
            'color_scale': 'RdYlBu',
            'show_values': len(df) <= 100
        })
        
        return config
    
    async def _create_table_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create table configuration"""
        
        return {
            'type': 'table',
            'data': df.to_dict('records'),
            'columns': [{'header': col, 'accessor': col} for col in df.columns],
            'pagination': len(df) > 50,
            'sorting': True,
            'filtering': len(df) > 20,
            'export_options': ['csv', 'excel']
        }
    
    async def _generate_chart_title(self, query_context: Dict[str, Any], data_profile: Dict[str, Any]) -> str:
        """Generate appropriate chart title"""
        
        intent = query_context.get('intent', '')
        entities = query_context.get('entities', [])
        
        if intent and entities:
            return f"{intent.title()} - {', '.join(entities)}"
        elif intent:
            return intent.title()
        else:
            # Generate from data characteristics
            columns = list(data_profile['columns'].keys())
            if len(columns) >= 2:
                return f"{columns[1]} by {columns[0]}"
            else:
                return f"Analysis of {columns[0]}" if columns else "Data Analysis"
    
    async def _select_color_palette(self, chart_type: str, data_profile: Dict[str, Any]) -> List[str]:
        """Select appropriate color palette"""
        
        if chart_type in ['line', 'scatter']:
            return self.color_palettes['categorical'][:3]
        elif chart_type == 'heatmap':
            return self.color_palettes['diverging']
        elif 'temporal' in str(data_profile):
            return self.color_palettes['temporal']
        else:
            return self.color_palettes['categorical']
    
    async def _generate_annotations(self, df: pd.DataFrame, chart_type: str, data_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate helpful annotations for the chart"""
        
        annotations = []
        
        # Add data quality warnings
        null_counts = df.isnull().sum()
        if null_counts.any():
            high_null_cols = null_counts[null_counts > len(df) * 0.1].index.tolist()
            if high_null_cols:
                annotations.append({
                    'type': 'warning',
                    'text': f"High missing data in: {', '.join(high_null_cols)}",
                    'position': 'top-right'
                })
        
        # Add insights for specific chart types
        if chart_type == 'scatter' and len(df.select_dtypes(include=[np.number]).columns) >= 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
            correlation = df[numeric_cols[0]].corr(df[numeric_cols[1]])
            if abs(correlation) > 0.7:
                annotations.append({
                    'type': 'insight',
                    'text': f"Strong correlation detected: {correlation:.2f}",
                    'position': 'bottom-left'
                })
        
        return annotations
    
    async def _analyze_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between columns"""
        
        relationships = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            relationships['correlations'] = corr_matrix.to_dict()
            
            # Find strong correlations
            strong_corrs = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append({
                            'col1': col1,
                            'col2': col2,
                            'correlation': corr_val
                        })
            
            relationships['strong_correlations'] = strong_corrs
        
        return relationships
    
    async def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        
        quality = {
            'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'outliers': {}
        }
        
        # Detect outliers in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            quality['outliers'][col] = len(outliers)
        
        return quality
    
    async def _is_temporal_string(self, series: pd.Series) -> bool:
        """Check if string series contains temporal data"""
        
        sample_values = series.dropna().head(10).tolist()
        temporal_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\w+ \d{1,2}, \d{4}',  # Month DD, YYYY
        ]
        
        for value in sample_values:
            if isinstance(value, str):
                for pattern in temporal_patterns:
                    if re.search(pattern, value):
                        return True
        
        return False
    
    async def _configure_x_axis(self, df: pd.DataFrame, data_profile: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """Configure X-axis settings"""
        return {"title": "X Axis", "type": "category"}
    
    async def _configure_y_axis(self, df: pd.DataFrame, data_profile: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """Configure Y-axis settings"""
        return {"title": "Y Axis", "type": "linear"}
    
    async def _configure_data_representation(self, df: pd.DataFrame, chart_type: str, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Configure how data is represented"""
        return {"encoding": "standard"}
    
    async def _configure_layout(self, chart_type: str, data_profile: Dict[str, Any], custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Configure chart layout"""
        return {"responsive": True, "padding": 20}
    
    async def _configure_interactivity(self, chart_type: str, data_profile: Dict[str, Any]) -> List[str]:
        """Configure interactive features"""
        return ["tooltip", "zoom", "pan"]
    
    async def _configure_accessibility(self, chart_type: str, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Configure accessibility features"""
        return {"alt_text": f"Chart showing {chart_type} visualization", "keyboard_navigation": True}
