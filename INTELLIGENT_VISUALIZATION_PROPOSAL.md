# 🎨 Intelligent Adaptive Visualization System
## Comprehensive Display Enhancement Proposal for NL2Q Analyst

### 📋 Executive Summary

Based on the screenshot example and your current Plotly-based visualization system, I propose creating an **Intelligent Adaptive Visualization System** that dynamically generates comprehensive, context-aware visualizations with:

1. **Smart KPI Cards** with trend indicators
2. **Multi-chart dashboard layouts** based on query type
3. **Timeline/Activity views** for temporal data
4. **Metric breakdowns** with visual hierarchies
5. **LLM-driven layout planning** that adapts to user questions

---

## 🎯 Current State Analysis

### What You Have Now
- ✅ Basic Plotly chart generation (`chart_builder.py`)
- ✅ Chart type recommendations (bar, line, pie, scatter, etc.)
- ✅ Chart customizer UI component
- ✅ Data profiling and analysis
- ❌ **Limited to single chart per query**
- ❌ **No comprehensive dashboard layouts**
- ❌ **No KPI card visualizations**
- ❌ **No timeline/activity views**
- ❌ **No intelligent layout planning**

### What the Screenshot Shows
1. **KPI Summary Cards** (Top section)
   - Metrics with values (40 total)
   - Trend indicators (100% this week, 0% trend)
   - Visual bar charts embedded in cards

2. **Activity Breakdown** (Bottom left)
   - Categorical metrics (20 Total, 0 Action Item, 2 Email, etc.)
   - Clean icon-based representation

3. **Timeline View** (Right sidebar)
   - Chronological activity list
   - Status indicators
   - Time-based grouping (Last month, 3 months ago, etc.)

4. **Visual Hierarchy**
   - Primary metrics highlighted
   - Secondary details visible but not dominant
   - Color-coded status indicators

---

## 🚀 Proposed Solution: 4-Tier Visualization Architecture

### Tier 1: **Smart Visualization Planner** (LLM-Driven)
Analyzes user query to determine optimal display layout

```python
class VisualizationPlanner:
    """
    LLM-driven planner that analyzes query intent and data characteristics
    to generate comprehensive visualization layouts
    """
    
    def analyze_query_intent(self, query: str, data: pd.DataFrame) -> VisualizationPlan:
        """
        Detects query type and plans visualization strategy:
        - Trend query → Timeline + Line chart + KPI cards
        - Comparison query → Bar charts + Comparison cards + Delta indicators
        - Summary query → KPI cards + Pie chart + Breakdown table
        - Activity query → Timeline + Activity breakdown + Status cards
        - Performance query → Scorecard + Trend chart + Goal indicators
        """
```

**Query Type Detection Examples:**

| User Query | Detected Type | Visualization Plan |
|------------|---------------|-------------------|
| "Show prescription trends for last 6 months" | Trend Analysis | KPI cards (total, avg, change) + Line chart + Timeline view |
| "Compare Q1 vs Q2 sales" | Period Comparison | Delta cards + Side-by-side bar chart + Percentage change indicators |
| "Healthcare provider performance summary" | Performance Dashboard | Scorecard + Multi-metric cards + Performance gauge |
| "Recent prescription activities" | Activity Stream | Timeline view + Activity breakdown + Status indicators |
| "Top 10 prescribers by volume" | Ranking Analysis | Leaderboard cards + Bar chart + Trend sparklines |

### Tier 2: **Component Library** (Rich UI Elements)

#### 2.1 KPI Card Component
```typescript
interface KPICardProps {
  title: string;
  value: number | string;
  unit?: string;
  trend?: {
    direction: 'up' | 'down' | 'neutral';
    percentage: number;
    label: string; // "vs last month"
  };
  comparison?: {
    current: number;
    previous: number;
    label: string;
  };
  sparkline?: number[]; // Mini trend chart
  icon?: React.ReactNode;
  status?: 'success' | 'warning' | 'danger' | 'info';
}

<KPICard
  title="Total Prescriptions"
  value={1247}
  trend={{ direction: 'up', percentage: 15.3, label: 'vs last month' }}
  sparkline={[100, 120, 115, 130, 145, 150]}
  status="success"
/>
```

#### 2.2 Timeline/Activity View Component
```typescript
interface TimelineItem {
  id: string;
  timestamp: Date;
  title: string;
  description?: string;
  type: 'action' | 'email' | 'call' | 'meeting' | 'note';
  status?: 'completed' | 'pending' | 'missed';
  metadata?: Record<string, any>;
}

<TimelineView
  items={activities}
  groupBy="relative" // "relative" | "date" | "week" | "month"
  showStatus={true}
  interactive={true}
/>
```

#### 2.3 Metric Breakdown Component
```typescript
interface MetricBreakdown {
  categories: Array<{
    label: string;
    value: number;
    icon?: React.ReactNode;
    percentage?: number;
  }>;
  total: number;
  layout: 'grid' | 'list' | 'compact';
}

<MetricBreakdown
  categories={[
    { label: 'Total', value: 20, icon: <CheckIcon /> },
    { label: 'Action Item', value: 0, icon: <TaskIcon /> },
    { label: 'Email', value: 2, icon: <MailIcon /> },
    { label: 'Call', value: 1, icon: <PhoneIcon /> }
  ]}
  total={23}
  layout="grid"
/>
```

#### 2.4 Comparison Card Component
```typescript
interface ComparisonCardProps {
  metric: string;
  periods: Array<{
    label: string;
    value: number;
    trend?: number;
  }>;
  visualizationType: 'bar' | 'line' | 'area';
  showDelta: boolean;
}

<ComparisonCard
  metric="Prescription Volume"
  periods={[
    { label: 'Q1 2024', value: 1500, trend: 10 },
    { label: 'Q2 2024', value: 1750, trend: 16.7 }
  ]}
  visualizationType="bar"
  showDelta={true}
/>
```

#### 2.5 Performance Scorecard
```typescript
interface ScorecardProps {
  metrics: Array<{
    label: string;
    value: number;
    target?: number;
    unit?: string;
    format?: 'number' | 'percentage' | 'currency';
  }>;
  layout: '2x2' | '3x2' | '4x1';
  theme: 'light' | 'dark' | 'pharma';
}

<PerformanceScorecard
  metrics={[
    { label: 'Total Rx', value: 1247, target: 1200, format: 'number' },
    { label: 'Growth Rate', value: 15.3, target: 10, format: 'percentage' },
    { label: 'Revenue', value: 125000, target: 100000, format: 'currency' },
    { label: 'Satisfaction', value: 4.5, target: 4.0, format: 'number' }
  ]}
  layout="2x2"
  theme="pharma"
/>
```

### Tier 3: **Layout Engine** (Adaptive Grid System)

```python
class AdaptiveLayoutEngine:
    """
    Generates responsive layouts based on visualization plan
    Supports multiple layout patterns
    """
    
    LAYOUT_PATTERNS = {
        'dashboard': {
            'structure': [
                {'type': 'kpi_row', 'columns': 4, 'height': '120px'},
                {'type': 'chart_row', 'columns': 2, 'height': '400px'},
                {'type': 'detail_row', 'columns': 1, 'height': 'auto'}
            ]
        },
        'trend_analysis': {
            'structure': [
                {'type': 'kpi_row', 'columns': 3, 'height': '100px'},
                {'type': 'main_chart', 'columns': 1, 'height': '450px'},
                {'type': 'timeline', 'columns': 1, 'height': '300px'}
            ]
        },
        'comparison': {
            'structure': [
                {'type': 'comparison_cards', 'columns': 2, 'height': '150px'},
                {'type': 'chart_grid', 'columns': 2, 'height': '350px'},
                {'type': 'delta_table', 'columns': 1, 'height': 'auto'}
            ]
        },
        'activity_stream': {
            'structure': [
                {'type': 'summary_row', 'columns': 3, 'height': '100px'},
                {'type': 'split_view', 'left': 'breakdown', 'right': 'timeline', 'height': '500px'}
            ]
        }
    }
```

### Tier 4: **Backend Enhancement** (Data Transformation)

```python
class ComprehensiveVisualizationGenerator:
    """
    Generates complete visualization specifications from query results
    """
    
    async def generate_comprehensive_viz(
        self,
        query: str,
        data: pd.DataFrame,
        query_metadata: Dict[str, Any]
    ) -> VisualizationSpec:
        """
        1. Analyze query intent (trend, comparison, summary, activity)
        2. Profile data characteristics
        3. Generate KPI metrics
        4. Create visualization components
        5. Plan layout structure
        6. Return comprehensive spec
        """
        
        # Step 1: Detect query type
        query_type = await self.detect_query_type(query, data)
        
        # Step 2: Extract KPIs
        kpis = await self.extract_kpis(data, query_type)
        
        # Step 3: Generate charts
        charts = await self.generate_charts(data, query_type)
        
        # Step 4: Create timeline (if temporal data)
        timeline = await self.create_timeline(data) if self.has_temporal_data(data) else None
        
        # Step 5: Generate breakdown
        breakdown = await self.create_breakdown(data, query_type)
        
        # Step 6: Plan layout
        layout = await self.plan_layout(query_type, kpis, charts, timeline, breakdown)
        
        return VisualizationSpec(
            layout=layout,
            kpis=kpis,
            charts=charts,
            timeline=timeline,
            breakdown=breakdown,
            metadata=query_metadata
        )
```

---

## 📊 Example Visualization Plans

### Example 1: "Show prescription trends for last 6 months"

**Detected Type:** Trend Analysis

**Generated Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  KPI Row (3 cards)                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Total Rx     │  │ Avg per Month│  │ Growth Rate  │     │
│  │ 7,485        │  │ 1,248        │  │ ↑ 15.3%      │     │
│  │ ━━━━━━━      │  │ ━━━━━━━      │  │ vs prev 6mo  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  Main Chart (Line chart with area fill)                    │
│  📈 Prescription Volume Over Time                          │
│  [Interactive line chart showing monthly trends]            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Timeline View (Right sidebar - optional)                   │
│  📅 Last Month          📅 2 Months Ago                    │
│  • 1,350 Rx            • 1,280 Rx                          │
│  • Peak: Week 3         • Peak: Week 2                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Backend Response:**
```json
{
  "visualization_type": "trend_analysis",
  "layout": "dashboard",
  "components": {
    "kpis": [
      {
        "type": "kpi_card",
        "title": "Total Prescriptions",
        "value": 7485,
        "trend": { "direction": "up", "percentage": 15.3, "label": "vs previous 6 months" },
        "sparkline": [1100, 1150, 1200, 1280, 1350, 1405]
      },
      {
        "type": "kpi_card",
        "title": "Avg per Month",
        "value": 1248,
        "unit": "Rx/month"
      },
      {
        "type": "kpi_card",
        "title": "Growth Rate",
        "value": 15.3,
        "unit": "%",
        "status": "success"
      }
    ],
    "charts": [
      {
        "type": "line",
        "title": "Prescription Volume Over Time",
        "data": { /* Plotly data */ },
        "config": {
          "interactive": true,
          "show_grid": true,
          "enable_zoom": true
        }
      }
    ],
    "timeline": {
      "type": "period_summary",
      "periods": [
        { "label": "Last Month", "value": 1350, "highlights": ["Peak in Week 3"] },
        { "label": "2 Months Ago", "value": 1280, "highlights": ["Peak in Week 2"] }
      ]
    }
  }
}
```

### Example 2: "Compare Q1 2024 vs Q1 2023 sales"

**Detected Type:** Period Comparison

**Generated Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Comparison Cards (2 large cards)                           │
│  ┌────────────────────────┐  ┌────────────────────────┐   │
│  │ Q1 2024                │  │ Q1 2023                │   │
│  │ $1,750,000             │  │ $1,500,000             │   │
│  │ ━━━━━━━━━━━━━━━        │  │ ━━━━━━━━━━━            │   │
│  │ ↑ 16.7% vs Q1 2023     │  │ Baseline               │   │
│  └────────────────────────┘  └────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Delta Indicator (Center)                                   │
│          ↗ +$250,000 (+16.7%) Growth                       │
├─────────────────────────────────────────────────────────────┤
│  Comparison Charts (Side by side)                           │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ Q1 2024 Breakdown│      │ Q1 2023 Breakdown│           │
│  │ [Bar Chart]      │      │ [Bar Chart]      │           │
│  └──────────────────┘      └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Example 3: "Recent healthcare provider activities"

**Detected Type:** Activity Stream

**Generated Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Summary Cards (3 cards)                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Total    │  │ This Week│  │ Pending  │                 │
│  │ 120      │  │ 45       │  │ 8        │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
├───────────────────────────┬─────────────────────────────────┤
│ Activity Breakdown        │ Timeline View                   │
│ ┌─────────────────────┐  │ 📅 Last Week                    │
│ │ 20 Total            │  │ ✓ Dr. Smith - Follow-up         │
│ │ 0  Action Items     │  │   Jun 15, 2024 10:30 AM        │
│ │ 2  Emails           │  │                                 │
│ │ 1  Calls            │  │ ✓ Meeting with Dr. Johnson     │
│ │ 17 Meetings         │  │   Jun 14, 2024 2:00 PM         │
│ └─────────────────────┘  │                                 │
│                           │ 📅 2 Weeks Ago                  │
│                           │ ⚠ Dr. Williams - Pending       │
│                           │   Jun 8, 2024 9:00 AM          │
└───────────────────────────┴─────────────────────────────────┘
```

---

## 🛠️ Implementation Plan

### Phase 1: Backend Enhancement (Week 1-2)
**Files to Create/Modify:**

1. **`backend/agents/visualization_planner.py`** (NEW)
   - LLM-driven query intent detection
   - Visualization strategy planning
   - Layout recommendation engine

2. **`backend/agents/comprehensive_viz_generator.py`** (NEW)
   - KPI extraction from data
   - Timeline generation for temporal data
   - Breakdown creation for categorical data
   - Multi-component visualization specs

3. **`backend/tools/chart_builder.py`** (ENHANCE)
   - Add KPI card specifications
   - Add timeline data formatting
   - Add comparison metrics calculation
   - Add trend analysis utilities

### Phase 2: Frontend Component Library (Week 3-4)
**Files to Create:**

1. **`frontend/src/components/visualizations/KPICard.tsx`**
   - Rich KPI card with trends
   - Sparkline integration
   - Status indicators

2. **`frontend/src/components/visualizations/TimelineView.tsx`**
   - Activity timeline
   - Grouping by time periods
   - Status indicators

3. **`frontend/src/components/visualizations/MetricBreakdown.tsx`**
   - Categorical breakdown
   - Icon support
   - Grid/list layouts

4. **`frontend/src/components/visualizations/ComparisonCard.tsx`**
   - Period comparison
   - Delta indicators
   - Side-by-side charts

5. **`frontend/src/components/visualizations/PerformanceScorecard.tsx`**
   - Multi-metric dashboard
   - Goal indicators
   - Gauge charts

6. **`frontend/src/components/visualizations/AdaptiveLayout.tsx`**
   - Dynamic layout engine
   - Responsive grid system
   - Component orchestration

### Phase 3: Integration (Week 5)
**Files to Modify:**

1. **`backend/orchestrators/dynamic_agent_orchestrator.py`**
   - Add visualization planning step
   - Integrate comprehensive viz generator

2. **`frontend/src/components/EnterpriseAgenticUI.tsx`**
   - Integrate adaptive layout component
   - Handle comprehensive viz specs
   - Render multi-component layouts

### Phase 4: LLM Prompt Engineering (Week 6)
**Prompt Templates:**

```python
VISUALIZATION_PLANNING_PROMPT = """
You are a Visualization Planning Expert. Analyze the user query and data to plan comprehensive visualizations.

USER QUERY: "{query}"

DATA CHARACTERISTICS:
- Rows: {row_count}
- Columns: {columns}
- Temporal columns: {temporal_columns}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

AVAILABLE COMPONENTS:
1. KPI Cards - Show key metrics with trends
2. Charts - Bar, line, pie, scatter, etc.
3. Timeline - Chronological activity view
4. Breakdown - Categorical distribution
5. Comparison - Period-over-period
6. Scorecard - Multi-metric dashboard

TASK:
Generate a comprehensive visualization plan including:
1. Primary visualization type (dashboard, trend_analysis, comparison, activity_stream)
2. KPIs to extract (3-4 key metrics)
3. Chart types needed
4. Layout structure
5. Component priorities

OUTPUT FORMAT: JSON
"""
```

---

## 📐 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ User Query: "Show prescription trends last 6 months"        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Dynamic Agent Orchestrator                                  │
│  └─> Query Execution → Data Retrieved                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Visualization Planner (LLM-Driven)                          │
│  • Detect query type: "trend_analysis"                      │
│  • Analyze data: temporal=yes, numeric=yes                  │
│  • Plan layout: kpi_row + line_chart + timeline             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Comprehensive Viz Generator                                 │
│  • Extract KPIs: total, average, growth                     │
│  • Generate chart: line chart with area fill                │
│  • Create timeline: period summaries                        │
│  • Build layout spec: JSON structure                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Frontend: Adaptive Layout Engine                            │
│  • Parse viz spec                                           │
│  • Render KPI cards                                         │
│  • Render charts (Plotly)                                   │
│  • Render timeline                                          │
│  • Apply responsive grid                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Design System

### Color Palette (Healthcare/Pharma Theme)
```css
:root {
  /* Primary Colors */
  --primary-blue: #1f77b4;
  --primary-green: #2ca02c;
  --primary-orange: #ff7f0e;
  
  /* Status Colors */
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #ef4444;
  --info: #3b82f6;
  
  /* Neutral Colors */
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #f1f5f9;
  --text-primary: #1e293b;
  --text-secondary: #64748b;
  --border: #e2e8f0;
  
  /* Chart Colors */
  --chart-1: #1f77b4;
  --chart-2: #ff7f0e;
  --chart-3: #2ca02c;
  --chart-4: #d62728;
  --chart-5: #9467bd;
}
```

### Typography
```css
/* Headers */
.kpi-title { font-size: 14px; font-weight: 600; color: var(--text-secondary); }
.kpi-value { font-size: 32px; font-weight: 700; color: var(--text-primary); }
.trend-label { font-size: 12px; font-weight: 500; }

/* Charts */
.chart-title { font-size: 18px; font-weight: 600; }
.chart-subtitle { font-size: 14px; color: var(--text-secondary); }

/* Timeline */
.timeline-period { font-size: 12px; font-weight: 600; text-transform: uppercase; }
.timeline-item-title { font-size: 14px; font-weight: 500; }
.timeline-item-time { font-size: 12px; color: var(--text-secondary); }
```

---

## 📊 Sample Implementation Code

### Backend: Visualization Planner

```python
# backend/agents/visualization_planner.py

from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
import pandas as pd
import json

class VisualizationPlanner:
    """
    LLM-driven visualization planning for comprehensive displays
    """
    
    def __init__(self):
        self.client = AsyncOpenAI()
        self.query_patterns = {
            'trend': ['trend', 'over time', 'progression', 'growth', 'change'],
            'comparison': ['compare', 'vs', 'versus', 'difference', 'between'],
            'summary': ['summary', 'overview', 'total', 'breakdown'],
            'activity': ['recent', 'activities', 'events', 'timeline', 'history'],
            'performance': ['performance', 'metrics', 'kpi', 'scorecard']
        }
    
    async def plan_visualization(
        self,
        query: str,
        data: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plan comprehensive visualization based on query and data
        """
        
        # Step 1: Quick pattern matching
        query_type = self._detect_query_type_simple(query)
        
        # Step 2: Data profiling
        data_profile = self._profile_data(data)
        
        # Step 3: LLM-based detailed planning
        viz_plan = await self._llm_plan_visualization(
            query, query_type, data_profile, metadata
        )
        
        return viz_plan
    
    def _detect_query_type_simple(self, query: str) -> str:
        """Quick pattern matching for query type"""
        query_lower = query.lower()
        
        scores = {}
        for qtype, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            scores[qtype] = score
        
        # Return type with highest score, default to 'summary'
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'summary'
    
    def _profile_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Profile data characteristics"""
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'numeric_columns': list(df.select_dtypes(include=['number']).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'temporal_columns': [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()],
            'has_temporal': any('date' in col.lower() or 'time' in col.lower() for col in df.columns),
            'data_density': len(df) * len(df.columns)
        }
    
    async def _llm_plan_visualization(
        self,
        query: str,
        query_type: str,
        data_profile: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to plan detailed visualization"""
        
        prompt = f"""You are a Visualization Planning Expert for healthcare data analytics.

USER QUERY: "{query}"
DETECTED TYPE: {query_type}

DATA CHARACTERISTICS:
- Total rows: {data_profile['row_count']}
- Columns: {', '.join(data_profile['columns'][:10])}
- Numeric columns: {', '.join(data_profile['numeric_columns'][:5])}
- Temporal data: {data_profile['has_temporal']}

TASK: Plan a comprehensive visualization that includes:

1. PRIMARY LAYOUT TYPE:
   - dashboard: Multi-metric overview with KPIs and charts
   - trend_analysis: Time-based progression with trends
   - comparison: Side-by-side period/category comparisons
   - activity_stream: Timeline with activity breakdown
   - performance: Scorecard with goal indicators

2. KPI CARDS (3-4 key metrics to highlight):
   - Identify the most important metrics
   - Suggest trend indicators where applicable
   - Recommend status colors (success/warning/danger)

3. PRIMARY CHART:
   - Type: bar, line, pie, scatter, area, etc.
   - X-axis and Y-axis
   - Visual style recommendations

4. ADDITIONAL COMPONENTS (if applicable):
   - Timeline view for temporal data
   - Breakdown table for categorical data
   - Comparison cards for period analysis
   - Activity indicators

5. LAYOUT STRUCTURE:
   - Row configuration (how to organize components)
   - Priority ranking of components

OUTPUT FORMAT: Valid JSON only, no explanations.

Example output:
{{
  "layout_type": "trend_analysis",
  "kpis": [
    {{"title": "Total Prescriptions", "value_column": "prescription_count", "aggregation": "sum", "trend": true}},
    {{"title": "Average per Month", "value_column": "prescription_count", "aggregation": "mean", "trend": false}},
    {{"title": "Growth Rate", "calculation": "percentage_change", "trend": true, "status_thresholds": {{"success": 10, "warning": 5}}}}
  ],
  "primary_chart": {{
    "type": "line",
    "x_axis": "date",
    "y_axis": "prescription_count",
    "style": "area_fill",
    "title": "Prescription Trends Over Time"
  }},
  "additional_components": [
    {{"type": "timeline", "enabled": true, "group_by": "month"}},
    {{"type": "breakdown", "enabled": false}}
  ],
  "layout_structure": [
    {{"row": 1, "type": "kpi_row", "columns": 3}},
    {{"row": 2, "type": "main_chart", "columns": 1}},
    {{"row": 3, "type": "timeline", "columns": 1}}
  ]
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
            
            viz_plan = json.loads(content)
            
            # Add metadata
            viz_plan['query_type'] = query_type
            viz_plan['data_profile'] = data_profile
            
            return viz_plan
            
        except Exception as e:
            print(f"LLM planning failed: {e}")
            # Fallback to simple plan
            return self._create_fallback_plan(query_type, data_profile)
    
    def _create_fallback_plan(
        self,
        query_type: str,
        data_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create simple fallback plan"""
        return {
            "layout_type": query_type,
            "kpis": [],
            "primary_chart": {
                "type": "bar",
                "x_axis": data_profile['columns'][0] if data_profile['columns'] else "category",
                "y_axis": data_profile['numeric_columns'][0] if data_profile['numeric_columns'] else "value",
                "title": "Data Visualization"
            },
            "additional_components": [],
            "layout_structure": [
                {"row": 1, "type": "main_chart", "columns": 1}
            ]
        }
```

### Frontend: KPI Card Component

```typescript
// frontend/src/components/visualizations/KPICard.tsx

import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import './KPICard.css';

interface TrendData {
  direction: 'up' | 'down' | 'neutral';
  percentage: number;
  label: string;
}

interface KPICardProps {
  title: string;
  value: number | string;
  unit?: string;
  trend?: TrendData;
  sparkline?: number[];
  icon?: React.ReactNode;
  status?: 'success' | 'warning' | 'danger' | 'info';
  format?: 'number' | 'currency' | 'percentage';
}

export const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  unit,
  trend,
  sparkline,
  icon,
  status = 'info',
  format = 'number'
}) => {
  const formatValue = (val: number | string): string => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0
        }).format(val);
      case 'percentage':
        return `${val.toFixed(1)}%`;
      default:
        return val.toLocaleString();
    }
  };

  const getTrendIcon = () => {
    if (!trend) return null;
    
    switch (trend.direction) {
      case 'up':
        return <TrendingUp className={`trend-icon trend-${trend.percentage >= 0 ? 'positive' : 'negative'}`} />;
      case 'down':
        return <TrendingDown className={`trend-icon trend-${trend.percentage < 0 ? 'positive' : 'negative'}`} />;
      default:
        return <Minus className="trend-icon trend-neutral" />;
    }
  };

  const renderSparkline = () => {
    if (!sparkline || sparkline.length === 0) return null;
    
    const max = Math.max(...sparkline);
    const min = Math.min(...sparkline);
    const range = max - min;
    
    const points = sparkline.map((val, idx) => {
      const x = (idx / (sparkline.length - 1)) * 100;
      const y = 100 - ((val - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');
    
    return (
      <svg className="sparkline" viewBox="0 0 100 30" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
      </svg>
    );
  };

  return (
    <div className={`kpi-card kpi-card--${status}`}>
      <div className="kpi-card__header">
        {icon && <div className="kpi-card__icon">{icon}</div>}
        <h3 className="kpi-card__title">{title}</h3>
      </div>
      
      <div className="kpi-card__body">
        <div className="kpi-card__value-container">
          <span className="kpi-card__value">
            {formatValue(value)}
          </span>
          {unit && <span className="kpi-card__unit">{unit}</span>}
        </div>
        
        {trend && (
          <div className="kpi-card__trend">
            {getTrendIcon()}
            <span className={`trend-value trend-${trend.direction}`}>
              {trend.percentage > 0 ? '+' : ''}{trend.percentage.toFixed(1)}%
            </span>
            <span className="trend-label">{trend.label}</span>
          </div>
        )}
        
        {sparkline && (
          <div className="kpi-card__sparkline">
            {renderSparkline()}
          </div>
        )}
      </div>
    </div>
  );
};
```

```css
/* frontend/src/components/visualizations/KPICard.css */

.kpi-card {
  background: var(--bg-primary);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid var(--border);
  transition: all 0.3s ease;
}

.kpi-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.kpi-card__header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.kpi-card__icon {
  font-size: 20px;
  color: var(--text-secondary);
}

.kpi-card__title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 0;
}

.kpi-card__value-container {
  display: flex;
  align-items: baseline;
  gap: 6px;
  margin-bottom: 8px;
}

.kpi-card__value {
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1;
}

.kpi-card__unit {
  font-size: 16px;
  font-weight: 500;
  color: var(--text-secondary);
}

.kpi-card__trend {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 8px;
}

.trend-icon {
  width: 16px;
  height: 16px;
}

.trend-positive {
  color: var(--success);
}

.trend-negative {
  color: var(--danger);
}

.trend-neutral {
  color: var(--text-secondary);
}

.trend-value {
  font-size: 14px;
  font-weight: 600;
}

.trend-label {
  font-size: 12px;
  color: var(--text-secondary);
}

.kpi-card__sparkline {
  margin-top: 12px;
  height: 30px;
  color: var(--primary-blue);
  opacity: 0.7;
}

/* Status variants */
.kpi-card--success {
  border-left: 4px solid var(--success);
}

.kpi-card--warning {
  border-left: 4px solid var(--warning);
}

.kpi-card--danger {
  border-left: 4px solid var(--danger);
}

.kpi-card--info {
  border-left: 4px solid var(--info);
}
```

---

## 🎯 Expected Outcomes

### Before (Current State)
```
User Query: "Show prescription trends for last 6 months"

Result:
┌────────────────────────────────┐
│ Single Line Chart              │
│ (Prescription trends)          │
│                                │
│                                │
│                                │
└────────────────────────────────┘

Table Below:
Month | Count
Jan   | 1100
Feb   | 1150
...
```

### After (Enhanced System)
```
User Query: "Show prescription trends for last 6 months"

Result:
┌─────────────────────────────────────────────────────────────┐
│ KPI Cards                                                    │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│ │ Total Rx     │  │ Avg/Month    │  │ Growth       │      │
│ │ 7,485        │  │ 1,248        │  │ ↑ 15.3%      │      │
│ │ [sparkline]  │  │ vs target:   │  │ vs prev 6mo  │      │
│ │              │  │ 98.5%        │  │ ━━━━━━━      │      │
│ └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│ Interactive Line Chart with Area Fill                       │
│ 📈 Prescription Volume Trends                               │
│ [Rich Plotly chart with zoom, hover tooltips]              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Timeline Summary (Optional sidebar)                         │
│ 📅 Last Month: 1,350 Rx (Peak in Week 3)                   │
│ 📅 2 Months Ago: 1,280 Rx (Peak in Week 2)                 │
└─────────────────────────────────────────────────────────────┘

Enhanced Table:
Month | Count | Change | Trend
Jan   | 1100  | --     | ━━━━
Feb   | 1150  | +4.5%  | ━━━━━
...
```

---

## 📦 Deliverables

1. **Backend Enhancement Package**
   - `visualization_planner.py` - LLM-driven planning
   - `comprehensive_viz_generator.py` - Multi-component generation
   - Enhanced `chart_builder.py` with KPI/timeline support

2. **Frontend Component Library**
   - `KPICard.tsx` - Rich metric cards
   - `TimelineView.tsx` - Activity timeline
   - `MetricBreakdown.tsx` - Categorical breakdown
   - `ComparisonCard.tsx` - Period comparisons
   - `PerformanceScorecard.tsx` - Multi-metric dashboard
   - `AdaptiveLayout.tsx` - Layout orchestration

3. **Integration Code**
   - Orchestrator integration
   - API endpoint enhancements
   - Frontend rendering logic

4. **Documentation**
   - Component usage guide
   - Layout pattern library
   - Customization guide

---

## 🚀 Next Steps

1. **Review & Approve** this proposal
2. **Phase 1**: Implement backend visualization planner
3. **Phase 2**: Create frontend component library
4. **Phase 3**: Integration & testing
5. **Phase 4**: LLM prompt tuning
6. **Phase 5**: Production deployment

---

**Would you like me to start implementing any specific component from this proposal?** 

I recommend starting with:
1. **Backend Visualization Planner** - The foundation for intelligent displays
2. **KPI Card Component** - Most impactful visual improvement
3. **Adaptive Layout Engine** - Enables comprehensive layouts

Let me know which you'd like to tackle first!
