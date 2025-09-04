"""
Inline Results Renderer with Rich Visualizations
Renders query results with charts, tables, insights, and download options
"""

import asyncio
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import base64
import io
import zipfile
import tempfile
import os

@dataclass
class TableRenderConfig:
    """Configuration for table rendering"""
    max_preview_rows: int = 200
    include_pagination: bool = True
    show_column_types: bool = True
    highlight_key_columns: bool = True
    enable_sorting: bool = True
    enable_filtering: bool = True

@dataclass
class ChartConfig:
    """Configuration for chart rendering"""
    chart_type: str  # bar, line, pie, scatter, heatmap, treemap
    title: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_by: Optional[str] = None
    size_by: Optional[str] = None
    facet_by: Optional[str] = None
    
@dataclass
class RenderedResult:
    """Complete rendered result with multiple formats"""
    table_html: str
    visualizations: List[Dict[str, Any]]
    summary_insights: List[str]
    download_links: Dict[str, str]
    metadata: Dict[str, Any]
    performance_stats: Dict[str, Any]

class InlineRenderer:
    """
    Advanced results renderer for pharmaceutical analytics
    Creates rich inline visualizations and downloadable formats
    """
    
    def __init__(self):
        # Pharmaceutical color palettes
        self.pharma_colors = {
            "primary": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "performance": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
            "outcomes": ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
            "therapeutic": ["#8ECAE6", "#219EBC", "#023047", "#FFB703", "#FB8500"]
        }
        
        # Chart templates for pharmaceutical analytics
        self.pharma_chart_templates = {
            "prescription_trends": {
                "type": "line",
                "title": "Prescription Trends Over Time",
                "config": {"showlegend": True, "hovermode": "x unified"}
            },
            "market_share": {
                "type": "pie",
                "title": "Market Share Analysis", 
                "config": {"showlegend": True, "hole": 0.3}
            },
            "hcp_performance": {
                "type": "bar",
                "title": "HCP Performance Metrics",
                "config": {"showlegend": True, "orientation": "v"}
            },
            "patient_outcomes": {
                "type": "scatter",
                "title": "Patient Outcomes Analysis",
                "config": {"showlegend": True, "hovermode": "closest"}
            },
            "territory_heatmap": {
                "type": "heatmap",
                "title": "Territory Performance Heatmap",
                "config": {"showlegend": True, "colorscale": "Blues"}
            }
        }
    
    async def format_table_data(
        self,
        data: List[Dict[str, Any]],
        config: Optional[TableRenderConfig] = None
    ) -> Dict[str, Any]:
        """
        Format tabular data with pagination and sorting
        """
        
        if not config:
            config = TableRenderConfig()
        
        if not data:
            return {
                "html": "<div class='no-data'>No data available</div>",
                "metadata": {"total_rows": 0, "columns": []},
                "pagination": None
            }
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Calculate pagination
        total_rows = len(df)
        preview_rows = min(config.max_preview_rows, total_rows)
        has_more = total_rows > config.max_preview_rows
        
        # Get preview data
        preview_df = df.head(preview_rows)
        
        # Generate HTML table
        table_html = self._generate_enhanced_table_html(preview_df, config)
        
        # Create metadata
        metadata = {
            "total_rows": total_rows,
            "preview_rows": preview_rows,
            "columns": list(df.columns),
            "column_types": {col: str(df[col].dtype) for col in df.columns},
            "has_more": has_more,
            "summary_stats": self._calculate_summary_stats(df)
        }
        
        # Pagination info
        pagination = None
        if config.include_pagination and has_more:
            pagination = {
                "current_page": 1,
                "total_pages": (total_rows + config.max_preview_rows - 1) // config.max_preview_rows,
                "page_size": config.max_preview_rows,
                "total_rows": total_rows
            }
        
        return {
            "html": table_html,
            "metadata": metadata,
            "pagination": pagination,
            "full_data": data  # Keep for further processing
        }
    
    async def build_pharma_visualizations(
        self,
        data: List[Dict[str, Any]],
        pharma_templates: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Build pharmaceutical-specific visualizations
        """
        
        if not data:
            return []
        
        df = pd.DataFrame(data)
        visualizations = []
        
        # Auto-detect chart opportunities
        chart_opportunities = self._detect_chart_opportunities(df)
        
        for opportunity in chart_opportunities:
            try:
                chart = await self._create_chart(df, opportunity, pharma_templates)
                if chart:
                    visualizations.append(chart)
            except Exception as e:
                # Log error but continue with other charts
                print(f"Chart creation failed: {str(e)}")
                continue
        
        return visualizations
    
    def _generate_enhanced_table_html(
        self, 
        df: pd.DataFrame, 
        config: TableRenderConfig
    ) -> str:
        """Generate enhanced HTML table with features"""
        
        # Start building HTML
        html = ['<div class="enhanced-table-container">']
        
        # Add table controls if enabled
        if config.enable_sorting or config.enable_filtering:
            html.append('<div class="table-controls">')
            if config.enable_sorting:
                html.append('<button class="sort-button" data-sort="asc">Sort ↑</button>')
                html.append('<button class="sort-button" data-sort="desc">Sort ↓</button>')
            if config.enable_filtering:
                html.append('<input type="text" class="table-filter" placeholder="Filter results...">')
            html.append('</div>')
        
        # Start table
        html.append('<table class="enhanced-table">')
        
        # Table header
        html.append('<thead><tr>')
        for col in df.columns:
            col_class = ""
            if config.highlight_key_columns and self._is_key_column(col):
                col_class = "key-column"
            
            col_type = ""
            if config.show_column_types:
                col_type = f"<span class='column-type'>{str(df[col].dtype)}</span>"
            
            html.append(f'<th class="{col_class}">{col} {col_type}</th>')
        
        html.append('</tr></thead>')
        
        # Table body
        html.append('<tbody>')
        for _, row in df.iterrows():
            html.append('<tr>')
            for col in df.columns:
                value = row[col]
                formatted_value = self._format_cell_value(value, col)
                cell_class = self._get_cell_class(value, col)
                html.append(f'<td class="{cell_class}">{formatted_value}</td>')
            html.append('</tr>')
        html.append('</tbody>')
        
        html.append('</table>')
        html.append('</div>')
        
        return '\n'.join(html)
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the dataset"""
        
        stats = {
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "null_counts": {},
            "unique_counts": {}
        }
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            
            # Categorize columns
            if col_type in ['int64', 'float64', 'int32', 'float32']:
                stats["numeric_columns"].append({
                    "name": col,
                    "mean": df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "median": df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "min": df[col].min(),
                    "max": df[col].max()
                })
            elif 'datetime' in col_type:
                stats["date_columns"].append({
                    "name": col,
                    "earliest": df[col].min(),
                    "latest": df[col].max(),
                    "range_days": (df[col].max() - df[col].min()).days if pd.api.types.is_datetime64_any_dtype(df[col]) else None
                })
            else:
                stats["categorical_columns"].append({
                    "name": col,
                    "unique_values": df[col].nunique(),
                    "most_common": df[col].mode().iloc[0] if not df[col].mode().empty else None
                })
            
            # Null and unique counts
            stats["null_counts"][col] = int(df[col].isnull().sum())
            stats["unique_counts"][col] = int(df[col].nunique())
        
        return stats
    
    def _detect_chart_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Auto-detect visualization opportunities"""
        
        opportunities = []
        
        # Get column info
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Time series opportunities
        if date_cols and numeric_cols:
            for date_col in date_cols[:1]:  # Limit to first date column
                for numeric_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                    opportunities.append({
                        "type": "line",
                        "title": f"{numeric_col} Over Time",
                        "x": date_col,
                        "y": numeric_col,
                        "template": "prescription_trends"
                    })
        
        # Categorical breakdown opportunities
        if categorical_cols and numeric_cols:
            for cat_col in categorical_cols[:2]:  # Limit categories
                if df[cat_col].nunique() <= 10:  # Reasonable number of categories
                    for numeric_col in numeric_cols[:1]:  # First numeric column
                        
                        # Bar chart
                        opportunities.append({
                            "type": "bar",
                            "title": f"{numeric_col} by {cat_col}",
                            "x": cat_col,
                            "y": numeric_col,
                            "template": "hcp_performance"
                        })
                        
                        # Pie chart if suitable
                        if df[cat_col].nunique() <= 6:
                            opportunities.append({
                                "type": "pie",
                                "title": f"{numeric_col} Distribution by {cat_col}",
                                "labels": cat_col,
                                "values": numeric_col,
                                "template": "market_share"
                            })
        
        # Scatter plot opportunities
        if len(numeric_cols) >= 2:
            opportunities.append({
                "type": "scatter",
                "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "template": "patient_outcomes"
            })
        
        # Heatmap opportunities (if we have multiple numeric columns and categories)
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 2:
            opportunities.append({
                "type": "heatmap",
                "title": f"{numeric_cols[0]} Heatmap",
                "template": "territory_heatmap"
            })
        
        return opportunities[:6]  # Limit to 6 charts maximum
    
    async def _create_chart(
        self,
        df: pd.DataFrame,
        opportunity: Dict[str, Any],
        use_pharma_templates: bool
    ) -> Optional[Dict[str, Any]]:
        """Create individual chart based on opportunity"""
        
        chart_type = opportunity["type"]
        title = opportunity["title"]
        
        try:
            # Prepare data for chart
            if chart_type == "line":
                fig = self._create_line_chart(df, opportunity)
            elif chart_type == "bar":
                fig = self._create_bar_chart(df, opportunity)
            elif chart_type == "pie":
                fig = self._create_pie_chart(df, opportunity)
            elif chart_type == "scatter":
                fig = self._create_scatter_chart(df, opportunity)
            elif chart_type == "heatmap":
                fig = self._create_heatmap_chart(df, opportunity)
            else:
                return None
            
            if fig is None:
                return None
            
            # Apply pharma styling
            if use_pharma_templates:
                self._apply_pharma_styling(fig, opportunity.get("template", "default"))
            
            # Convert to JSON
            chart_json = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
            return {
                "id": f"chart_{hash(title)}",
                "type": chart_type,
                "title": title,
                "config": chart_json,
                "insights": self._generate_chart_insights(df, opportunity),
                "download_formats": ["png", "svg", "pdf", "html"]
            }
            
        except Exception as e:
            print(f"Error creating {chart_type} chart: {str(e)}")
            return None
    
    def _create_line_chart(self, df: pd.DataFrame, opportunity: Dict[str, Any]) -> Optional[go.Figure]:
        """Create line chart"""
        
        x_col = opportunity.get("x")
        y_col = opportunity.get("y")
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return None
        
        # Aggregate data if needed
        if df[x_col].dtype == 'object':
            # If x is categorical, aggregate
            chart_data = df.groupby(x_col)[y_col].sum().reset_index()
        else:
            chart_data = df.sort_values(x_col)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data[x_col],
            y=chart_data[y_col],
            mode='lines+markers',
            name=y_col,
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=opportunity["title"],
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=400
        )
        
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, opportunity: Dict[str, Any]) -> Optional[go.Figure]:
        """Create bar chart"""
        
        x_col = opportunity.get("x")
        y_col = opportunity.get("y")
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return None
        
        # Aggregate data
        chart_data = df.groupby(x_col)[y_col].sum().reset_index()
        chart_data = chart_data.sort_values(y_col, ascending=False).head(10)  # Top 10
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_data[x_col],
            y=chart_data[y_col],
            name=y_col,
            marker=dict(
                color=chart_data[y_col],
                colorscale='Blues',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=opportunity["title"],
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=400
        )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, opportunity: Dict[str, Any]) -> Optional[go.Figure]:
        """Create pie chart"""
        
        labels_col = opportunity.get("labels")
        values_col = opportunity.get("values")
        
        if not labels_col or not values_col or labels_col not in df.columns or values_col not in df.columns:
            return None
        
        # Aggregate data
        chart_data = df.groupby(labels_col)[values_col].sum().reset_index()
        chart_data = chart_data.sort_values(values_col, ascending=False).head(8)  # Top 8 slices
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=chart_data[labels_col],
            values=chart_data[values_col],
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        ))
        
        fig.update_layout(
            title=opportunity["title"],
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _create_scatter_chart(self, df: pd.DataFrame, opportunity: Dict[str, Any]) -> Optional[go.Figure]:
        """Create scatter chart"""
        
        x_col = opportunity.get("x")
        y_col = opportunity.get("y")
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return None
        
        # Sample data if too many points
        chart_data = df.sample(min(1000, len(df))) if len(df) > 1000 else df
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data[x_col],
            y=chart_data[y_col],
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.7,
                color=chart_data[y_col] if pd.api.types.is_numeric_dtype(chart_data[y_col]) else None,
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=opportunity["title"],
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=400
        )
        
        return fig
    
    def _create_heatmap_chart(self, df: pd.DataFrame, opportunity: Dict[str, Any]) -> Optional[go.Figure]:
        """Create heatmap chart"""
        
        # Create a simple correlation heatmap for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        correlation_matrix = df[numeric_cols].corr()
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=opportunity["title"],
            height=400
        )
        
        return fig
    
    def _apply_pharma_styling(self, fig: go.Figure, template: str) -> None:
        """Apply pharmaceutical industry styling to charts"""
        
        template_config = self.pharma_chart_templates.get(template, {})
        colors = self.pharma_colors["primary"]
        
        # Update layout with pharma styling
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            colorway=colors,
            **template_config.get("config", {})
        )
        
        # Update axes styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='gray'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='gray'
        )
    
    def _generate_chart_insights(self, df: pd.DataFrame, opportunity: Dict[str, Any]) -> List[str]:
        """Generate key insights for the chart"""
        
        insights = []
        chart_type = opportunity["type"]
        
        if chart_type == "line":
            y_col = opportunity.get("y")
            if y_col and y_col in df.columns:
                trend = "increasing" if df[y_col].iloc[-1] > df[y_col].iloc[0] else "decreasing"
                insights.append(f"{y_col.replace('_', ' ').title()} shows an overall {trend} trend")
        
        elif chart_type == "bar":
            x_col = opportunity.get("x")
            y_col = opportunity.get("y")
            if x_col and y_col and both in df.columns:
                top_category = df.groupby(x_col)[y_col].sum().idxmax()
                insights.append(f"Highest performance: {top_category}")
        
        elif chart_type == "pie":
            labels_col = opportunity.get("labels")
            values_col = opportunity.get("values")
            if labels_col and values_col and both in df.columns:
                top_slice = df.groupby(labels_col)[values_col].sum().idxmax()
                percentage = (df.groupby(labels_col)[values_col].sum().max() / df[values_col].sum()) * 100
                insights.append(f"Largest segment: {top_slice} ({percentage:.1f}%)")
        
        return insights
    
    def _is_key_column(self, column_name: str) -> bool:
        """Determine if column should be highlighted as key column"""
        
        key_indicators = [
            'id', 'key', 'name', 'date', 'time', 'count', 'total', 
            'amount', 'value', 'score', 'rate', 'percent'
        ]
        
        column_lower = column_name.lower()
        return any(indicator in column_lower for indicator in key_indicators)
    
    def _format_cell_value(self, value: Any, column_name: str) -> str:
        """Format cell value based on data type and column name"""
        
        if pd.isna(value):
            return '<span class="null-value">—</span>'
        
        column_lower = column_name.lower()
        
        # Currency formatting
        if any(word in column_lower for word in ['amount', 'cost', 'price', 'revenue', 'value']):
            if isinstance(value, (int, float)):
                return f"${value:,.2f}"
        
        # Percentage formatting
        elif any(word in column_lower for word in ['percent', 'rate', 'ratio', 'share']):
            if isinstance(value, (int, float)):
                return f"{value:.1f}%"
        
        # Count formatting
        elif any(word in column_lower for word in ['count', 'total', 'number']):
            if isinstance(value, (int, float)):
                return f"{value:,.0f}"
        
        # Date formatting
        elif isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        
        # Large number formatting
        elif isinstance(value, (int, float)) and abs(value) >= 1000:
            return f"{value:,.0f}"
        
        # Default formatting
        return str(value)
    
    def _get_cell_class(self, value: Any, column_name: str) -> str:
        """Get CSS class for cell based on value"""
        
        classes = []
        
        if pd.isna(value):
            classes.append("null-value")
        
        elif isinstance(value, (int, float)):
            if value > 0:
                classes.append("positive-value")
            elif value < 0:
                classes.append("negative-value")
            else:
                classes.append("zero-value")
        
        column_lower = column_name.lower()
        if any(word in column_lower for word in ['id', 'key']):
            classes.append("id-column")
        
        return " ".join(classes)
    
    async def create_download_links(
        self,
        data: List[Dict[str, Any]],
        query_id: str
    ) -> Dict[str, str]:
        """Create downloadable files in multiple formats"""
        
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        download_links = {}
        
        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        
        try:
            # CSV format
            csv_path = os.path.join(temp_dir, f"results_{query_id}.csv")
            df.to_csv(csv_path, index=False)
            download_links["csv"] = csv_path
            
            # Excel format
            excel_path = os.path.join(temp_dir, f"results_{query_id}.xlsx")
            df.to_excel(excel_path, index=False, sheet_name="Results")
            download_links["excel"] = excel_path
            
            # JSON format
            json_path = os.path.join(temp_dir, f"results_{query_id}.json")
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            download_links["json"] = json_path
            
            # Create summary report
            summary_path = os.path.join(temp_dir, f"summary_{query_id}.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Query Results Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Total Records: {len(data):,}\n")
                f.write(f"Columns: {len(df.columns)}\n")
                f.write(f"Column Names: {', '.join(df.columns)}\n\n")
                
                # Add basic statistics
                for col in df.select_dtypes(include=['number']).columns:
                    f.write(f"{col}:\n")
                    f.write(f"  Mean: {df[col].mean():.2f}\n")
                    f.write(f"  Median: {df[col].median():.2f}\n")
                    f.write(f"  Min: {df[col].min()}\n")
                    f.write(f"  Max: {df[col].max()}\n\n")
            
            download_links["summary"] = summary_path
            
        except Exception as e:
            print(f"Error creating download files: {str(e)}")
        
        return download_links

# Global instance
inline_renderer = InlineRenderer()
