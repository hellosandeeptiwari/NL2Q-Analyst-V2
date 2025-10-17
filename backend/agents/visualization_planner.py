"""
Intelligent Visualization Planner - LLM-Driven Adaptive Visualization System
Analyzes user queries and data to dynamically plan comprehensive visualizations

This system:
1. Detects query intent using LLM
2. Profiles data characteristics
3. Plans optimal visualization layouts
4. Generates component specifications
5. Adapts to different query types and data patterns
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class KPISpec:
    """Specification for a KPI card"""
    title: str
    value_column: Optional[str] = None
    calculation: str = "sum"  # sum, mean, count, percentage_change, etc.
    trend: bool = False
    trend_comparison: Optional[str] = None  # "previous_period", "target", etc.
    status_thresholds: Optional[Dict[str, float]] = None
    format: str = "number"  # number, currency, percentage
    icon: Optional[str] = None
    sparkline: bool = False
    time_period: Optional[str] = None  # "Last month", "Last quarter", etc.
    comparison_text: Optional[str] = None  # Additional context text for the card

@dataclass
class ChartSpec:
    """Specification for a chart"""
    type: str  # bar, line, pie, scatter, area, etc.
    title: str
    x_axis: str
    y_axis: str
    style: Optional[str] = None
    aggregation: Optional[str] = None
    color_scheme: Optional[str] = None
    interactive_features: List[str] = None

@dataclass
class TimelineSpec:
    """Specification for timeline view"""
    enabled: bool
    time_column: str
    group_by: str = "month"  # day, week, month, quarter
    show_labels: bool = True
    max_items: int = 10

@dataclass
class BreakdownSpec:
    """Specification for metric breakdown"""
    enabled: bool
    category_column: str
    value_column: str
    layout: str = "grid"  # grid, list, compact
    show_icons: bool = True
    max_categories: int = 10

@dataclass
class TemporalComparisonCard:
    """Specification for temporal comparison cards (right side panel)"""
    time_period: str  # "Last month", "Last quarter", "2 months ago", etc.
    relative_offset: int  # -1 for last period, -2 for 2 periods ago, etc.
    summary_text: str  # Brief description of activity in that period
    kpis: List[Dict[str, Any]]  # Mini-KPIs for this time period
    
@dataclass
class TemporalContextSpec:
    """Specification for temporal/contextual comparison context"""
    enabled: bool
    time_granularity: Optional[str] = None  # "day", "week", "month", "quarter", "year" (temporal only)
    comparison_periods: List[TemporalComparisonCard] = None  # Cards to show
    query_timeframe: str = ""  # "last_6_months", "top_prescribers", "categorical_breakdown", etc.
    context_type: str = "temporal"  # "temporal" or "contextual"
    insight_type: Optional[str] = None  # "categorical_breakdown", "statistical_summary", etc. (contextual only)

@dataclass
class LayoutRow:
    """Specification for a layout row"""
    row: int
    type: str  # kpi_row, main_chart, timeline, breakdown, comparison
    columns: int
    height: Optional[str] = None
    components: List[Dict[str, Any]] = None

@dataclass
class VisualizationPlan:
    """Complete visualization plan"""
    layout_type: str  # dashboard, trend_analysis, comparison, activity_stream, performance
    query_type: str
    kpis: List[KPISpec]
    primary_chart: ChartSpec
    timeline: Optional[TimelineSpec] = None
    breakdown: Optional[BreakdownSpec] = None
    temporal_context: Optional[TemporalContextSpec] = None  # NEW: Temporal comparison cards
    layout_structure: List[LayoutRow] = None
    metadata: Dict[str, Any] = None

class VisualizationPlanner:
    """
    LLM-driven visualization planner that adapts to user queries and data
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.fast_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        # Query pattern detection (quick pre-filtering)
        self.query_patterns = {
            'trend': ['trend', 'over time', 'progression', 'growth', 'change', 'historical', 'evolution'],
            'comparison': ['compare', 'vs', 'versus', 'difference', 'between', 'year-over-year', 'month-over-month'],
            'summary': ['summary', 'overview', 'total', 'breakdown', 'distribution', 'composition'],
            'activity': ['recent', 'activities', 'events', 'timeline', 'history', 'latest', 'last'],
            'performance': ['performance', 'metrics', 'kpi', 'scorecard', 'dashboard', 'goals', 'targets']
        }
    
    async def plan_visualization(
        self,
        query: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VisualizationPlan:
        """
        Main entry point: Plan comprehensive visualization based on query and data
        
        Args:
            query: User's natural language query
            data: Query results as DataFrame
            metadata: Optional metadata about query execution
            
        Returns:
            VisualizationPlan with complete specifications
        """
        
        print(f"📊 Planning visualization for query: '{query[:50]}...'")
        
        # Step 1: Quick pattern-based query type detection
        query_type = self._detect_query_type_simple(query)
        print(f"🔍 Detected query type: {query_type}")
        
        # Step 2: Profile data characteristics
        data_profile = self._profile_data(data)
        print(f"📈 Data profile: {data_profile['row_count']} rows, {data_profile['column_count']} columns")
        print(f"   Temporal data: {data_profile['has_temporal']}")
        print(f"   Numeric columns: {len(data_profile['numeric_columns'])}")
        
        # Step 3: Detect temporal/contextual insights for comparison cards
        temporal_context = self._detect_temporal_context(query, data_profile)
        if temporal_context['enabled']:
            context_type = temporal_context.get('context_type', 'temporal')
            if context_type == 'temporal':
                print(f"⏰ Temporal context detected: {temporal_context['query_timeframe']}")
                print(f"   Granularity: {temporal_context.get('time_granularity')}")
                print(f"   Comparison periods: {temporal_context['comparison_periods']}")
            else:
                print(f"📊 Contextual insights detected: {temporal_context['query_timeframe']}")
                print(f"   Insight type: {temporal_context.get('insight_type')}")
                print(f"   Comparison cards: {temporal_context['comparison_periods']}")
        
        # Step 4: LLM-based detailed planning
        try:
            viz_plan = await self._llm_plan_visualization(
                query, query_type, data_profile, metadata or {}, temporal_context
            )
            print(f"✅ LLM planning successful: {viz_plan['layout_type']}")
            
            # Convert to structured plan
            structured_plan = self._convert_to_structured_plan(viz_plan, data_profile, temporal_context)
            return structured_plan
            
        except Exception as e:
            print(f"⚠️ LLM planning failed: {e}, using fallback")
            # Fallback to rule-based planning
            return self._create_fallback_plan(query, query_type, data_profile, metadata or {}, temporal_context)
    
    def _detect_query_type_simple(self, query: str) -> str:
        """Quick pattern matching for initial query type detection"""
        query_lower = query.lower()
        
        scores = {}
        for qtype, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            scores[qtype] = score
        
        # Return type with highest score, default to 'summary'
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'summary'
    
    def _profile_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Profile data characteristics for intelligent planning"""
        
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        
        # Detect temporal columns
        temporal_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['date', 'time', 'year', 'month', 'quarter', 'day']):
                temporal_cols.append(col)
        
        # Analyze value distributions
        value_distributions = {}
        for col in categorical_cols[:5]:  # Limit to first 5 categorical
            unique_count = df[col].nunique()
            value_distributions[col] = {
                'unique_count': unique_count,
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'temporal_columns': temporal_cols,
            'has_temporal': len(temporal_cols) > 0,
            'has_numeric': len(numeric_cols) > 0,
            'has_categorical': len(categorical_cols) > 0,
            'data_density': len(df) * len(df.columns),
            'value_distributions': value_distributions
        }
    
    def _detect_temporal_context(self, query: str, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect temporal context from query to determine appropriate comparison periods
        If no temporal aspect, generate contextual insights (categorical/statistical)
        
        Returns dict with:
        - enabled: True if temporal OR contextual insights available
        - context_type: "temporal" or "contextual"
        - time_granularity: "day", "week", "month", "quarter", "year" (temporal only)
        - query_timeframe: "last_week", "top_performers", "categorical_breakdown", etc.
        - comparison_periods: List of period/category labels to show
        """
        query_lower = query.lower()
        
        # Detect timeframe from query
        timeframe_patterns = {
            'last_week': ['last week', 'past week', 'this week'],
            'last_month': ['last month', 'past month', 'this month'],
            'last_quarter': ['last quarter', 'past quarter', 'this quarter', 'quarterly'],
            'last_6_months': ['last 6 months', 'past 6 months', 'last six months', '6 months'],
            'last_year': ['last year', 'past year', 'this year', 'yearly', 'annual'],
            'ytd': ['year to date', 'ytd'],
            'mtd': ['month to date', 'mtd']
        }
        
        detected_timeframe = None
        for timeframe, patterns in timeframe_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_timeframe = timeframe
                break
        
        # Default to monthly if temporal data exists but no specific timeframe detected
        if not detected_timeframe and data_profile.get('has_temporal', False):
            detected_timeframe = 'last_month'
        
        # Map timeframe to granularity and comparison periods
        temporal_config = {
            'last_week': {
                'granularity': 'day',
                'comparison_periods': ['Yesterday', '2 days ago', '3 days ago', 'Last week']
            },
            'last_month': {
                'granularity': 'week',
                'comparison_periods': ['Last week', '2 weeks ago', '3 weeks ago', 'Last month']
            },
            'last_quarter': {
                'granularity': 'month',
                'comparison_periods': ['Last month', '2 months ago', '3 months ago', 'Last quarter']
            },
            'last_6_months': {
                'granularity': 'month',
                'comparison_periods': ['Last month', 'Last quarter', '4 months ago', '6 months ago']
            },
            'last_year': {
                'granularity': 'quarter',
                'comparison_periods': ['Last quarter', '2 quarters ago', '3 quarters ago', 'Last year']
            },
            'ytd': {
                'granularity': 'quarter',
                'comparison_periods': ['This quarter', 'Last quarter', '2 quarters ago', 'Year start']
            },
            'mtd': {
                'granularity': 'week',
                'comparison_periods': ['This week', 'Last week', '2 weeks ago', 'Month start']
            }
        }
        
        if detected_timeframe and detected_timeframe in temporal_config:
            config = temporal_config[detected_timeframe]
            return {
                'enabled': True,
                'context_type': 'temporal',
                'query_timeframe': detected_timeframe,
                'time_granularity': config['granularity'],
                'comparison_periods': config['comparison_periods']
            }
        
        # 🎯 NEW: If no temporal context, generate CONTEXTUAL INSIGHTS
        # Analyze query intent and data to provide relevant breakdowns
        contextual_insights = self._generate_contextual_insights(query_lower, data_profile)
        
        if contextual_insights['enabled']:
            return contextual_insights
        
        return {
            'enabled': False,
            'context_type': None,
            'query_timeframe': None,
            'time_granularity': 'month',
            'comparison_periods': []
        }
    
    def _generate_contextual_insights(self, query_lower: str, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate contextual insights for non-temporal queries
        Provides categorical breakdowns, statistical insights, or performance metrics
        """
        
        # Detect query patterns for contextual insights
        top_patterns = ['top', 'highest', 'best', 'leading', 'most', 'largest']
        bottom_patterns = ['bottom', 'lowest', 'worst', 'least', 'smallest']
        comparison_patterns = ['by', 'per', 'across', 'breakdown']
        
        is_top_query = any(pattern in query_lower for pattern in top_patterns)
        is_bottom_query = any(pattern in query_lower for pattern in bottom_patterns)
        is_comparison_query = any(pattern in query_lower for pattern in comparison_patterns)
        
        # Get categorical columns for potential breakdowns
        categorical_cols = data_profile.get('categorical_columns', [])
        
        # Strategy 1: TOP/BOTTOM queries → Show statistical breakdowns
        if is_top_query or is_bottom_query:
            # For prescriber queries, show specialty, region, territory breakdowns
            if any(term in query_lower for term in ['prescriber', 'doctor', 'physician', 'provider']):
                return {
                    'enabled': True,
                    'context_type': 'contextual_categorical',
                    'query_timeframe': 'top_prescribers',
                    'insight_type': 'categorical_breakdown',
                    'comparison_periods': ['By Specialty', 'By Region', 'By Territory', 'By Product']
                }
            
            # For product/sales queries, show product mix and performance
            elif any(term in query_lower for term in ['product', 'sales', 'revenue', 'nrx', 'trx']):
                return {
                    'enabled': True,
                    'context_type': 'contextual_statistical',
                    'query_timeframe': 'top_products',
                    'insight_type': 'statistical_summary',
                    'comparison_periods': ['Average', 'Top Performer', 'Distribution', 'Target %']
                }
            
            # Generic top query
            else:
                return {
                    'enabled': True,
                    'context_type': 'contextual_statistical',
                    'query_timeframe': 'top_items',
                    'insight_type': 'statistical_summary',
                    'comparison_periods': ['Total', 'Average', 'Top Item', 'Distribution']
                }
        
        # Strategy 2: Categorical breakdown queries → Show sub-categories
        elif is_comparison_query and len(categorical_cols) >= 2:
            # Intelligently select breakdown dimensions
            breakdown_dims = []
            
            # Prioritize common healthcare dimensions
            priority_columns = ['specialty', 'region', 'territory', 'product', 'category', 'type', 'status']
            
            for col in categorical_cols:
                col_lower = col.lower()
                if any(priority in col_lower for priority in priority_columns):
                    breakdown_dims.append(col)
                    if len(breakdown_dims) >= 4:
                        break
            
            # Fallback to first 4 categorical columns
            if len(breakdown_dims) < 2:
                breakdown_dims = categorical_cols[:4]
            
            if breakdown_dims:
                # Create friendly labels
                labels = [f"By {col.replace('_', ' ').title()}" for col in breakdown_dims]
                
                return {
                    'enabled': True,
                    'context_type': 'contextual_categorical',
                    'query_timeframe': 'categorical_analysis',
                    'insight_type': 'categorical_breakdown',
                    'comparison_periods': labels[:4],  # Max 4 cards
                    'breakdown_columns': breakdown_dims[:4]
                }
        
        # Strategy 3: Generic numeric query → Show statistical summary
        elif data_profile.get('has_numeric', False):
            return {
                'enabled': True,
                'context_type': 'contextual_statistical',
                'query_timeframe': 'statistical_overview',
                'insight_type': 'statistical_summary',
                'comparison_periods': ['Total Count', 'Average Value', 'Max Value', 'Min Value']
            }
        
        # No contextual insights available
        return {
            'enabled': False,
            'context_type': None
        }
    
    async def _llm_plan_visualization(
        self,
        query: str,
        query_type: str,
        data_profile: Dict[str, Any],
        metadata: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to create detailed, adaptive visualization plan
        This is where the magic happens - LLM decides everything based on context
        """
        
        # Build comprehensive context for LLM
        columns_summary = ', '.join(data_profile['columns'][:15])
        if len(data_profile['columns']) > 15:
            columns_summary += f" ... and {len(data_profile['columns']) - 15} more"
        
        numeric_summary = ', '.join(data_profile['numeric_columns'][:10])
        categorical_summary = ', '.join(data_profile['categorical_columns'][:5])
        temporal_summary = ', '.join(data_profile['temporal_columns']) if data_profile['temporal_columns'] else "None"
        
        # Distribution insights
        distribution_insights = ""
        if data_profile['value_distributions']:
            distribution_insights = "\n\nCATEGORICAL VALUE DISTRIBUTIONS:"
            for col, dist in list(data_profile['value_distributions'].items())[:3]:
                distribution_insights += f"\n- {col}: {dist['unique_count']} unique values"
                top_vals = list(dist['top_values'].items())[:3]
                distribution_insights += f"\n  Top: {', '.join([f'{k}({v})' for k, v in top_vals])}"
        
        # Add temporal/contextual insights to prompt
        context_text = ""
        if temporal_context.get('enabled', False):
            context_type = temporal_context.get('context_type', 'temporal')
            
            if context_type == 'temporal':
                context_text = f"""

TEMPORAL CONTEXT DETECTED:
- Query timeframe: {temporal_context['query_timeframe']}
- Time granularity: {temporal_context.get('time_granularity', 'month')}
- Comparison periods to show: {', '.join(temporal_context['comparison_periods'])}

IMPORTANT: Include temporal comparison cards showing metrics for these periods!
- Create KPI cards for EACH comparison period (e.g., "Last month", "2 months ago")
- Each card should have time_period and comparison_text fields
- This provides context for how metrics changed over time"""
            
            elif context_type in ['contextual_categorical', 'contextual_statistical']:
                insight_type = temporal_context.get('insight_type', 'breakdown')
                context_text = f"""

CONTEXTUAL INSIGHTS DETECTED (Non-temporal query):
- Query pattern: {temporal_context['query_timeframe']}
- Insight type: {insight_type}
- Comparison cards to show: {', '.join(temporal_context['comparison_periods'])}

IMPORTANT: Include contextual comparison cards for deeper insights!
- For "Top prescribers" → Show breakdowns by Specialty, Region, Territory, Product
- For "Sales/Product" queries → Show statistical summaries (Average, Top, Distribution)
- For categorical queries → Show sub-category breakdowns
- Each card should have time_period (repurposed as category label) and comparison_text
- Provide percentage breakdowns, totals, or key statistics per card
- This gives users multi-dimensional understanding of their data"""

        prompt = f"""You are an Expert Visualization Planner for healthcare analytics dashboards.

USER QUERY: "{query}"
INITIAL DETECTION: {query_type}

DATA CHARACTERISTICS:
- Total rows: {data_profile['row_count']}
- Total columns: {data_profile['column_count']}
- Columns: {columns_summary}
- Numeric columns: {numeric_summary or 'None'}
- Categorical columns: {categorical_summary or 'None'}
- Temporal columns: {temporal_summary}
- Has temporal data: {data_profile['has_temporal']}{distribution_insights}{context_text}

YOUR TASK:
Analyze this query and data to plan a COMPREHENSIVE, ADAPTIVE visualization that provides maximum insight.

AVAILABLE LAYOUT TYPES:
1. **dashboard** - Multi-metric overview with KPIs, multiple charts, comprehensive view
2. **trend_analysis** - Time-based analysis with temporal charts, timeline, trend indicators
3. **comparison** - Side-by-side comparison with delta cards, comparison charts
4. **activity_stream** - Timeline-focused with activity breakdown, chronological view
5. **performance** - Scorecard with goal indicators, performance metrics

PLANNING GUIDELINES:
- Choose layout type that BEST matches user intent and data characteristics
- For COMPARISON queries: Create KPIs for EACH category being compared (e.g., Enabled vs Disabled, Top vs Bottom)
- Include ALL metrics mentioned in the query (e.g., if query mentions TRX, NRX, and LunchLearn, create KPIs for all three)
- Select PRIMARY CHART that tells the main story
- Add timeline if temporal data exists and is relevant
- Add breakdown if categorical analysis would provide insight
- Consider what the user REALLY wants to know

CRITICAL: Adapt to the data you see:
- If temporal columns exist → Consider trend_analysis layout
- If comparison keywords in query (compare, vs, enabled/disabled, top/bottom) → Use comparison layout with KPIs per category
- If activity/recent keywords → Use activity_stream layout
- If multiple metrics → Use dashboard layout
- If performance keywords → Use performance layout

COMPARISON QUERY SPECIAL RULES:
- When query asks to "compare A vs B" or has categories (enabled/disabled, top/bottom, region A vs region B):
  * Identify the comparison dimension (the column being compared, e.g., PDRPFlag, Region, Specialty)
  * Identify ALL unique values in that dimension from the data
  * Create SEPARATE KPIs for EACH value (e.g., if PDRPFlag has YES/NO, create KPIs for both)
  * Use filter_condition: {{"filter_column": "ComparisonColumn", "filter_value": "SpecificValue"}}
- Include ALL numeric metrics mentioned in query (TRX, NRX, Sales, Calls, Samples, etc.)
- Example: Query "compare sales by region" with 3 regions → Create KPIs for each region × each metric

OUTPUT FORMAT: Valid JSON only, no explanations.

REQUIRED STRUCTURE:
{{
  "layout_type": "trend_analysis|comparison|dashboard|activity_stream|performance",
  "reasoning": "Brief explanation of why this layout",
  "kpis": [
    {{
      "title": "KPI display name",
      "value_column": "actual_column_name_from_data",
      "calculation": "sum|mean|count|max|percentage_change",
      "trend": true|false,
      "trend_comparison": "previous_period|target|baseline",
      "format": "number|currency|percentage",
      "sparkline": true|false,
      "icon": "activity|trendingUp|users|dollarSign|etc",
      "time_period": "Last month|Last quarter|etc (OPTIONAL for temporal comparison cards)",
      "comparison_text": "Additional context text (OPTIONAL)",
      "filter_condition": {{"filter_column": "column_name", "filter_value": "value"}} (OPTIONAL - use for comparison queries)
    }}
  ],
  "primary_chart": {{
    "type": "line|bar|pie|scatter|area|heatmap",
    "title": "Chart title",
    "x_axis": "column_name",
    "y_axis": "column_name",
    "style": "area_fill|stacked|grouped|etc",
    "aggregation": "sum|mean|count|etc",
    "color_scheme": "categorical|sequential|diverging"
  }},
  "timeline": {{
    "enabled": true|false,
    "time_column": "column_name_if_enabled",
    "group_by": "day|week|month|quarter",
    "show_labels": true|false
  }},
  "breakdown": {{
    "enabled": true|false,
    "category_column": "column_name_if_enabled",
    "value_column": "column_name_if_enabled",
    "layout": "grid|list|compact"
  }},
  "layout_structure": [
    {{"row": 1, "type": "kpi_row", "columns": 3, "height": "120px"}},
    {{"row": 2, "type": "main_chart", "columns": 1, "height": "400px"}},
    {{"row": 3, "type": "timeline|breakdown", "columns": 1, "height": "300px"}}
  ]
}}

EXAMPLES (for reference):

EXAMPLE 1 - Comparison Query:
Query: "Compare sales by region"
Data has: Region (East, West, North), Sales, Orders
{{
  "layout_type": "comparison",
  "reasoning": "User wants to compare performance across regions",
  "kpis": [
    {{"title": "Sales (East)", "value_column": "Sales", "calculation": "sum", "format": "currency", "filter_condition": {{"filter_column": "Region", "filter_value": "East"}}}},
    {{"title": "Sales (West)", "value_column": "Sales", "calculation": "sum", "format": "currency", "filter_condition": {{"filter_column": "Region", "filter_value": "West"}}}},
    {{"title": "Sales (North)", "value_column": "Sales", "calculation": "sum", "format": "currency", "filter_condition": {{"filter_column": "Region", "filter_value": "North"}}}},
    {{"title": "Orders (East)", "value_column": "Orders", "calculation": "sum", "format": "number", "filter_condition": {{"filter_column": "Region", "filter_value": "East"}}}},
    {{"title": "Orders (West)", "value_column": "Orders", "calculation": "sum", "format": "number", "filter_condition": {{"filter_column": "Region", "filter_value": "West"}}}},
    {{"title": "Orders (North)", "value_column": "Orders", "calculation": "sum", "format": "number", "filter_condition": {{"filter_column": "Region", "filter_value": "North"}}}}
  ],
  "primary_chart": {{"type": "bar", "title": "Sales by Region", "x_axis": "Region", "y_axis": "Sales"}},
  "breakdown": {{"enabled": false}}
}}

EXAMPLE 2 - Trend Analysis:
Query: "Show prescription trends"
{{
  "layout_type": "trend_analysis",
  "reasoning": "User asks for trends over time",
  "kpis": [
    {{"title": "Total Prescriptions", "value_column": "prescription_count", "calculation": "sum", "format": "number"}},
    {{"title": "Growth Rate", "value_column": "prescription_count", "calculation": "percentage_change", "format": "percentage"}}
  ],
  "primary_chart": {{"type": "line", "title": "Trends", "x_axis": "date", "y_axis": "prescription_count"}},
  "timeline": {{"enabled": true, "time_column": "date", "group_by": "month"}}
}}

NOW GENERATE THE PLAN FOR THIS QUERY. Think carefully about what the user wants to see."""

        try:
            response = await self.client.chat.completions.create(
                model=self.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistent planning
                max_tokens=3000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
            
            viz_plan = json.loads(content)
            
            # Validate plan
            if not self._validate_plan(viz_plan, data_profile):
                print("⚠️ LLM plan validation failed, using fallback")
                raise ValueError("Invalid plan structure")
            
            return viz_plan
            
        except Exception as e:
            print(f"❌ LLM planning error: {e}")
            raise
    
    def _validate_plan(self, plan: Dict[str, Any], data_profile: Dict[str, Any]) -> bool:
        """Validate that LLM-generated plan is structurally correct"""
        required_keys = ['layout_type', 'kpis', 'primary_chart', 'layout_structure']
        
        # Check required keys
        if not all(key in plan for key in required_keys):
            return False
        
        # Validate column names exist in data
        all_columns = data_profile['columns']
        
        # Check KPIs reference valid columns
        for kpi in plan.get('kpis', []):
            if kpi.get('value_column') and kpi['value_column'] not in all_columns:
                print(f"⚠️ KPI references invalid column: {kpi['value_column']}")
                # Don't fail, just warn - LLM might use calculated fields
        
        # Check chart references valid columns
        chart = plan.get('primary_chart', {})
        if chart.get('x_axis') and chart['x_axis'] not in all_columns:
            print(f"⚠️ Chart x_axis references invalid column: {chart['x_axis']}")
        if chart.get('y_axis') and chart['y_axis'] not in all_columns:
            print(f"⚠️ Chart y_axis references invalid column: {chart['y_axis']}")
        
        return True
    
    def _convert_to_structured_plan(
        self,
        llm_plan: Dict[str, Any],
        data_profile: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> VisualizationPlan:
        """Convert LLM JSON plan to structured VisualizationPlan object"""
        
        # Convert KPIs
        kpis = []
        for kpi_data in llm_plan.get('kpis', []):
            kpis.append(KPISpec(
                title=kpi_data.get('title', 'Metric'),
                value_column=kpi_data.get('value_column'),
                calculation=kpi_data.get('calculation', 'sum'),
                trend=kpi_data.get('trend', False),
                trend_comparison=kpi_data.get('trend_comparison'),
                status_thresholds=kpi_data.get('status_thresholds'),
                format=kpi_data.get('format', 'number'),
                icon=kpi_data.get('icon'),
                sparkline=kpi_data.get('sparkline', False),
                time_period=kpi_data.get('time_period'),  # NEW: temporal comparison
                comparison_text=kpi_data.get('comparison_text')  # NEW: additional context
            ))
        
        # Convert chart
        chart_data = llm_plan.get('primary_chart', {})
        primary_chart = ChartSpec(
            type=chart_data.get('type', 'bar'),
            title=chart_data.get('title', 'Chart'),
            x_axis=chart_data.get('x_axis', data_profile['columns'][0] if data_profile['columns'] else 'x'),
            y_axis=chart_data.get('y_axis', data_profile['numeric_columns'][0] if data_profile['numeric_columns'] else 'y'),
            style=chart_data.get('style'),
            aggregation=chart_data.get('aggregation'),
            color_scheme=chart_data.get('color_scheme'),
            interactive_features=chart_data.get('interactive_features', ['hover', 'zoom'])
        )
        
        # Convert timeline
        timeline_data = llm_plan.get('timeline', {})
        timeline = None
        if timeline_data.get('enabled', False):
            timeline = TimelineSpec(
                enabled=True,
                time_column=timeline_data.get('time_column', data_profile['temporal_columns'][0] if data_profile['temporal_columns'] else ''),
                group_by=timeline_data.get('group_by', 'month'),
                show_labels=timeline_data.get('show_labels', True),
                max_items=timeline_data.get('max_items', 10)
            )
        
        # Convert breakdown
        breakdown_data = llm_plan.get('breakdown', {})
        breakdown = None
        if breakdown_data.get('enabled', False):
            breakdown = BreakdownSpec(
                enabled=True,
                category_column=breakdown_data.get('category_column', data_profile['categorical_columns'][0] if data_profile['categorical_columns'] else ''),
                value_column=breakdown_data.get('value_column', data_profile['numeric_columns'][0] if data_profile['numeric_columns'] else ''),
                layout=breakdown_data.get('layout', 'grid'),
                show_icons=breakdown_data.get('show_icons', True),
                max_categories=breakdown_data.get('max_categories', 10)
            )
        
        # Convert layout structure
        layout_structure = []
        for row_data in llm_plan.get('layout_structure', []):
            layout_structure.append(LayoutRow(
                row=row_data.get('row', 1),
                type=row_data.get('type', 'main_chart'),
                columns=row_data.get('columns', 1),
                height=row_data.get('height'),
                components=row_data.get('components')
            ))
        
        # Build temporal context spec if enabled
        temporal_context_spec = None
        if temporal_context.get('enabled', False):
            comparison_cards = []
            for i, period_label in enumerate(temporal_context['comparison_periods']):
                comparison_cards.append(TemporalComparisonCard(
                    time_period=period_label,
                    relative_offset=-(i+1),  # -1, -2, -3, etc.
                    summary_text=f"Activity summary for {period_label}",
                    kpis=[]  # Will be populated dynamically by frontend/backend
                ))
            
            temporal_context_spec = TemporalContextSpec(
                enabled=True,
                time_granularity=temporal_context.get('time_granularity'),
                comparison_periods=comparison_cards,
                query_timeframe=temporal_context.get('query_timeframe', ''),
                context_type=temporal_context.get('context_type', 'temporal'),
                insight_type=temporal_context.get('insight_type')
            )
        
        return VisualizationPlan(
            layout_type=llm_plan.get('layout_type', 'dashboard'),
            query_type=llm_plan.get('reasoning', ''),
            kpis=kpis,
            primary_chart=primary_chart,
            timeline=timeline,
            breakdown=breakdown,
            temporal_context=temporal_context_spec,  # NEW: Temporal comparison cards
            layout_structure=layout_structure,
            metadata={
                'data_profile': data_profile,
                'llm_reasoning': llm_plan.get('reasoning', ''),
                'temporal_context': temporal_context  # Keep raw context for reference
            }
        )
    
    def _create_fallback_plan(
        self,
        query: str,
        query_type: str,
        data_profile: Dict[str, Any],
        metadata: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> VisualizationPlan:
        """
        Rule-based fallback plan when LLM planning fails
        Still adaptive to data characteristics
        """
        
        print(f"📋 Creating rule-based fallback plan for: {query_type}")
        
        # Determine layout based on query type
        layout_type_mapping = {
            'trend': 'trend_analysis',
            'comparison': 'comparison',
            'summary': 'dashboard',
            'activity': 'activity_stream',
            'performance': 'performance'
        }
        layout_type = layout_type_mapping.get(query_type, 'dashboard')
        
        # Create basic KPIs from numeric columns
        kpis = []
        numeric_cols = data_profile['numeric_columns'][:3]  # First 3 numeric columns
        for i, col in enumerate(numeric_cols):
            kpis.append(KPISpec(
                title=col.replace('_', ' ').title(),
                value_column=col,
                calculation='sum' if i == 0 else 'mean',
                trend=i == 0,  # First KPI gets trend
                format='number',
                sparkline=i == 0
            ))
        
        # Create chart based on data
        x_col = data_profile['temporal_columns'][0] if data_profile['temporal_columns'] else \
                data_profile['categorical_columns'][0] if data_profile['categorical_columns'] else \
                data_profile['columns'][0]
        
        y_col = data_profile['numeric_columns'][0] if data_profile['numeric_columns'] else \
                data_profile['columns'][1] if len(data_profile['columns']) > 1 else \
                data_profile['columns'][0]
        
        chart_type = 'line' if data_profile['has_temporal'] else 'bar'
        
        primary_chart = ChartSpec(
            type=chart_type,
            title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
            x_axis=x_col,
            y_axis=y_col,
            style='area_fill' if chart_type == 'line' else None,
            interactive_features=['hover', 'zoom']
        )
        
        # Timeline if temporal data
        timeline = None
        if data_profile['has_temporal']:
            timeline = TimelineSpec(
                enabled=True,
                time_column=data_profile['temporal_columns'][0],
                group_by='month',
                show_labels=True
            )
        
        # Basic layout structure
        layout_structure = [
            LayoutRow(row=1, type='kpi_row', columns=len(kpis), height='120px'),
            LayoutRow(row=2, type='main_chart', columns=1, height='400px')
        ]
        
        if timeline:
            layout_structure.append(LayoutRow(row=3, type='timeline', columns=1, height='250px'))
        
        # Build temporal context spec if enabled (same as in _convert_llm_to_plan)
        temporal_context_spec = None
        if temporal_context.get('enabled', False):
            comparison_cards = []
            for i, period_label in enumerate(temporal_context['comparison_periods']):
                comparison_cards.append(TemporalComparisonCard(
                    time_period=period_label,
                    relative_offset=-(i+1),
                    summary_text=f"Activity summary for {period_label}",
                    kpis=[]
                ))
            
            temporal_context_spec = TemporalContextSpec(
                enabled=True,
                time_granularity=temporal_context.get('time_granularity', 'month'),
                comparison_periods=comparison_cards,
                query_timeframe=temporal_context.get('query_timeframe', ''),
                context_type=temporal_context.get('context_type', 'temporal'),
                insight_type=temporal_context.get('insight_type')
            )
        
        return VisualizationPlan(
            layout_type=layout_type,
            query_type=query_type,
            kpis=kpis,
            primary_chart=primary_chart,
            timeline=timeline,
            breakdown=None,
            temporal_context=temporal_context_spec,  # FIX: Include temporal context in fallback
            layout_structure=layout_structure,
            metadata={
                'data_profile': data_profile,
                'fallback_used': True,
                'temporal_context': temporal_context  # Keep raw context for reference
            }
        )
    
    def plan_to_dict(self, plan: VisualizationPlan) -> Dict[str, Any]:
        """Convert VisualizationPlan to dictionary for JSON serialization"""
        # Convert temporal_context to dict if present
        temporal_context_dict = None
        if plan.temporal_context:
            temporal_context_dict = {
                'enabled': plan.temporal_context.enabled,
                'context_type': plan.temporal_context.context_type,
                'query_timeframe': plan.temporal_context.query_timeframe,
                'time_granularity': plan.temporal_context.time_granularity,
                'insight_type': plan.temporal_context.insight_type,
                'comparison_periods': [asdict(card) for card in plan.temporal_context.comparison_periods] if plan.temporal_context.comparison_periods else []
            }
        
        return {
            'layout_type': plan.layout_type,
            'query_type': plan.query_type,
            'kpis': [asdict(kpi) for kpi in plan.kpis],
            'primary_chart': asdict(plan.primary_chart),
            'timeline': asdict(plan.timeline) if plan.timeline else None,
            'breakdown': asdict(plan.breakdown) if plan.breakdown else None,
            'temporal_context': temporal_context_dict,  # FIX: Include temporal context
            'layout_structure': [asdict(row) for row in plan.layout_structure] if plan.layout_structure else [],
            'metadata': plan.metadata
        }


# Example usage
async def main():
    """Example usage of VisualizationPlanner"""
    import pandas as pd
    
    # Sample data
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=6, freq='M'),
        'prescription_count': [1100, 1150, 1200, 1280, 1350, 1405],
        'provider_count': [45, 47, 48, 50, 52, 54],
        'region': ['Midwest'] * 6
    })
    
    planner = VisualizationPlanner()
    
    query = "Show me prescription trends for the last 6 months"
    plan = await planner.plan_visualization(query, data)
    
    print("\n📊 VISUALIZATION PLAN:")
    print(f"Layout Type: {plan.layout_type}")
    print(f"KPIs: {len(plan.kpis)}")
    for kpi in plan.kpis:
        print(f"  - {kpi.title} ({kpi.calculation})")
    print(f"Chart: {plan.primary_chart.type} - {plan.primary_chart.title}")
    print(f"Timeline: {plan.timeline.enabled if plan.timeline else False}")
    print(f"Layout rows: {len(plan.layout_structure)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
