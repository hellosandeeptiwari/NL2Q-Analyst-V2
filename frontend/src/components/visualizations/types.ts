// Type definitions for Intelligent Visualization System
// Matches backend/agents/visualization_planner.py structures

export interface KPISpec {
  title: string;
  value_column?: string;
  calculation: 'sum' | 'mean' | 'count' | 'percentage_change' | 'max' | 'min';
  trend: boolean;
  trend_comparison?: 'previous_period' | 'target' | 'baseline';
  status_thresholds?: { [key: string]: number };
  format: 'number' | 'currency' | 'percentage';
  icon?: string;
  sparkline: boolean;
  time_period?: string;  // NEW: e.g., "Q1 2024", "Last Month"
  comparison_text?: string;  // NEW: e.g., "vs Q4 2023", "vs previous month"
  filter_condition?: {  // NEW: For filtering data before calculation
    filter_column: string;
    filter_value: string | number;
  };
}

export interface ChartSpec {
  type: 'bar' | 'line' | 'pie' | 'scatter' | 'area' | 'heatmap' | 'treemap';
  title: string;
  x_axis: string;
  y_axis: string;
  style?: string;
  aggregation?: string;
  color_scheme?: 'categorical' | 'sequential' | 'diverging';
  interactive_features?: string[];
}

export interface TimelineSpec {
  enabled: boolean;
  time_column: string;
  group_by: 'day' | 'week' | 'month' | 'quarter';
  show_labels: boolean;
  max_items?: number;
}

export interface BreakdownSpec {
  enabled: boolean;
  category_column: string;
  value_column: string;
  layout: 'grid' | 'list' | 'compact';
  show_icons: boolean;
  max_categories?: number;
}

export interface LayoutRow {
  row: number;
  type: 'kpi_row' | 'main_chart' | 'timeline' | 'breakdown' | 'comparison';
  columns: number;
  height?: string;
  components?: any[];
}

// NEW: Temporal/Contextual Comparison Card
export interface TemporalComparisonCard {
  time_period: string;  // e.g., "Last Quarter", "By Specialty", "Average"
  relative_offset: number;  // -1 for last period, -2 for 2 periods ago, etc.
  summary_text: string;  // Brief description of activity in that period
  kpis: Array<{ [key: string]: any }>;  // Mini-KPIs for this time period
}

// NEW: Temporal/Contextual Context
export interface TemporalContextSpec {
  enabled: boolean;
  context_type: 'temporal' | 'contextual';  // NEW: distinguish between temporal and contextual
  query_timeframe?: string;  // e.g., "Last Month", "YTD", "Top 10 Prescribers"
  time_granularity?: 'day' | 'week' | 'month' | 'quarter' | 'year';
  comparison_periods: TemporalComparisonCard[];
  insight_type?: string;  // NEW: e.g., "top_bottom", "prescriber_analysis", "statistical_summary"
}

export interface VisualizationPlan {
  layout_type: 'dashboard' | 'trend_analysis' | 'comparison' | 'activity_stream' | 'performance';
  query_type: string;
  kpis: KPISpec[];
  primary_chart: ChartSpec;
  timeline?: TimelineSpec;
  breakdown?: BreakdownSpec;
  temporal_context?: TemporalContextSpec;  // NEW: Temporal/Contextual insights
  layout_structure: LayoutRow[];
  metadata?: {
    data_profile?: any;
    llm_reasoning?: string;
  };
}

export interface IntelligentVisualizationResult {
  status: 'completed' | 'skipped' | 'failed';
  visualization_plan?: VisualizationPlan;
  summary?: string;
  error?: string;
  reason?: string;
  fallback?: string;
}
