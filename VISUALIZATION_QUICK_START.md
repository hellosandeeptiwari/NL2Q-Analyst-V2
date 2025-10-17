# 🚀 Quick Start: Intelligent Visualization System

## ✅ Status: READY TO USE

The intelligent, LLM-driven, adaptive visualization planning system is now **fully integrated** into your NL2Q Analyst backend!

---

## 🎯 What It Does

**Before**: Simple Plotly chart generation
```json
{
  "chart": {
    "type": "bar",
    "data": [...]
  }
}
```

**After**: Comprehensive, adaptive visualization plan
```json
{
  "visualization_plan": {
    "layout_type": "trend_analysis",
    "kpis": [
      {"title": "Total Prescriptions", "value": 1405, "trend": "+8.2%", "sparkline": true},
      {"title": "Avg per Month", "value": 234, "trend": false},
      {"title": "Growth Rate", "value": "8.2%", "trend": true}
    ],
    "primary_chart": {
      "type": "line",
      "title": "Prescription Trends Over Time",
      "x_axis": "date",
      "y_axis": "prescription_count"
    },
    "timeline": {
      "enabled": true,
      "time_column": "date",
      "group_by": "month"
    },
    "layout_structure": [...]
  }
}
```

---

## 🔧 How to Use

### Option 1: Automatic (Recommended)

The system **automatically activates** after query execution. No changes needed!

**Flow**:
```
User Query → Schema Discovery → Query Generation → Execution 
→ **Intelligent Viz Planning** ← AUTOMATIC
→ Results with comprehensive plan
```

### Option 2: Manual Testing

Test the planner directly:

```python
# backend/agents/visualization_planner.py
python backend/agents/visualization_planner.py

# Output:
# 📊 Planning visualization for query: 'Show me prescription trends...'
# 🔍 Detected query type: trend
# ✅ LLM planning successful: trend_analysis
# Layout Type: trend_analysis
# KPIs: 3
# Chart: line - Prescription Trends Over Time
```

### Option 3: API Integration

From your FastAPI endpoint:

```python
# Already integrated in dynamic_agent_orchestrator.py
# The visualization plan will be in results:

results = await orchestrator.process_query(
    user_query="Show prescription trends",
    user_id="user123"
)

# Access the plan:
viz_plan = results['results'].get('intelligent_visualization_planning', {})
if viz_plan.get('status') == 'completed':
    plan = viz_plan['visualization_plan']
    # Use plan to render comprehensive UI
```

---

## 🎨 Example Queries & Adaptations

### 1. Trend Analysis
```
Query: "Show prescription trends over time"

LLM Plans:
✓ Layout: trend_analysis
✓ KPIs: Total, Average, Growth Rate
✓ Chart: Line chart with area fill
✓ Timeline: Month-by-month view
```

### 2. Comparison
```
Query: "Compare Q1 vs Q2 sales"

LLM Plans:
✓ Layout: comparison
✓ KPIs: Q1 Total, Q2 Total, Delta, % Change
✓ Chart: Side-by-side bars
✓ Timeline: Disabled
```

### 3. Activity Stream
```
Query: "Show recent patient activities"

LLM Plans:
✓ Layout: activity_stream
✓ KPIs: Total, Today, This Week
✓ Chart: Activity timeline
✓ Breakdown: By activity type
```

### 4. Dashboard Overview
```
Query: "Give me an overview of all metrics"

LLM Plans:
✓ Layout: dashboard
✓ KPIs: 4-6 key metrics
✓ Charts: Multiple mini charts
✓ Breakdown: Top categories
```

---

## 🧪 Testing

### Test 1: Run Example
```bash
cd "c:\Users\SandeepT\NL2Q Analyst\NL2Q-Analyst-V2"
python backend/agents/visualization_planner.py
```

Expected output:
```
📊 Planning visualization for query: 'Show me prescription trends for the last 6 months'
🔍 Detected query type: trend
📈 Data profile: 6 rows, 4 columns
   Temporal data: True
   Numeric columns: 2
🤖 Invoking LLM for visualization planning...
✅ LLM planning successful: trend_analysis

📊 VISUALIZATION PLAN:
Layout Type: trend_analysis
KPIs: 3
  - Total Prescriptions (sum)
  - Avg per Month (mean)
  - Growth Rate (percentage_change)
Chart: line - Prescription Trends Over Time
Timeline: True
Layout rows: 3
```

### Test 2: Integration Test
```python
# Test via orchestrator
from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator

orchestrator = DynamicAgentOrchestrator()
results = await orchestrator.process_query(
    user_query="Show me top prescribers by volume",
    user_id="test_user"
)

# Check for visualization plan
if 'intelligent_visualization_planning' in results['results']:
    viz = results['results']['intelligent_visualization_planning']
    print(f"Status: {viz['status']}")
    print(f"Layout: {viz['visualization_plan']['layout_type']}")
```

---

## 📊 Plan Structure Reference

### Complete Plan Object:
```python
{
  "status": "completed",
  "visualization_plan": {
    "layout_type": "trend_analysis",  # or: dashboard, comparison, activity_stream, performance
    "query_type": "Temporal trend analysis",
    
    "kpis": [  # 3-4 key metrics
      {
        "title": "Total Prescriptions",
        "value_column": "prescription_count",
        "calculation": "sum",  # sum, mean, count, percentage_change
        "trend": true,
        "trend_comparison": "previous_period",
        "format": "number",  # number, currency, percentage
        "sparkline": true,
        "icon": "activity"
      }
    ],
    
    "primary_chart": {
      "type": "line",  # bar, line, pie, scatter, area, heatmap
      "title": "Prescription Trends Over Time",
      "x_axis": "date",
      "y_axis": "prescription_count",
      "style": "area_fill",
      "aggregation": "sum",
      "color_scheme": "sequential",
      "interactive_features": ["hover", "zoom"]
    },
    
    "timeline": {
      "enabled": true,
      "time_column": "date",
      "group_by": "month",  # day, week, month, quarter
      "show_labels": true,
      "max_items": 10
    },
    
    "breakdown": {
      "enabled": false,
      "category_column": null,
      "value_column": null
    },
    
    "layout_structure": [
      {"row": 1, "type": "kpi_row", "columns": 3, "height": "120px"},
      {"row": 2, "type": "main_chart", "columns": 1, "height": "450px"},
      {"row": 3, "type": "timeline", "columns": 1, "height": "250px"}
    ],
    
    "metadata": {
      "data_profile": {...},
      "llm_reasoning": "User asks for trends over time, temporal data available"
    }
  },
  "summary": "Created trend_analysis layout with 3 KPIs"
}
```

---

## 🎯 Integration Checklist

### Backend (✅ COMPLETE):
- [x] VisualizationPlanner class created
- [x] LLM-driven planning implemented
- [x] Query type detection
- [x] Data profiling
- [x] Orchestrator integration
- [x] Task type added
- [x] Execution method added
- [x] Error handling
- [x] Fallback system
- [x] No syntax errors

### Frontend (📋 TODO):
- [ ] KPICard component
- [ ] TimelineView component
- [ ] MetricBreakdown component
- [ ] AdaptiveLayout component
- [ ] Integration with EnterpriseAgenticUI
- [ ] Plan rendering logic

---

## 🔍 Debugging

### Enable Detailed Logging:
```python
# In visualization_planner.py, the system already logs:
print(f"📊 Planning visualization for query: '{query[:50]}...'")
print(f"🔍 Detected query type: {query_type}")
print(f"📈 Data profile: {data_profile['row_count']} rows")
print(f"🤖 Invoking LLM for visualization planning...")
print(f"✅ LLM planning successful: {viz_plan['layout_type']}")
```

### Check Plan Generation:
```python
# After query execution, check results:
results['results']['intelligent_visualization_planning']

# Should return:
{
  "status": "completed",  # or "skipped", "failed"
  "visualization_plan": {...},
  "summary": "Created ... layout with ... KPIs"
}
```

### Common Issues:

**Issue**: Plan not generated
```python
# Check:
1. OPENAI_API_KEY set in environment
2. Execution results contain data
3. VisualizationPlanner imported successfully

# Look for logs:
"⚠️ Visualization Planner not available"  # Import failed
"⚠️ No execution results found"  # No data to plan for
"❌ Intelligent visualization planning failed"  # LLM error
```

**Issue**: LLM planning fails
```python
# System automatically falls back to rule-based planning
# Check logs:
"⚠️ LLM planning failed: ..., using fallback"
"📋 Creating rule-based fallback plan for: ..."
```

---

## 🚀 Next Steps

### Immediate (Backend Ready):
1. ✅ Test visualization planner standalone
2. ✅ Run queries through orchestrator
3. ✅ Verify plans are generated
4. 📊 Start using plans in API responses

### Short-term (Frontend Development):
1. Create KPICard React component
2. Create TimelineView component
3. Create AdaptiveLayout orchestrator
4. Integrate with EnterpriseAgenticUI
5. Render comprehensive displays

### Long-term (Enhancements):
1. Add more layout types
2. User preference storage
3. Layout customization UI
4. Export/share functionality
5. Advanced chart interactions

---

## 📚 Files Modified/Created

### Created:
```
backend/agents/visualization_planner.py (600+ lines)
INTELLIGENT_VISUALIZATION_IMPLEMENTATION_SUMMARY.md
VISUALIZATION_QUICK_START.md (this file)
```

### Modified:
```
backend/orchestrators/dynamic_agent_orchestrator.py
  - Lines 26-33: Import added
  - Line 116: TaskType.INTELLIGENT_VISUALIZATION_PLANNING
  - Lines 4008-4090: New execution method
  - Line 1441: Task routing
  - Line 1469: Agent mapping
```

---

## 💡 Tips

### 1. Query Phrasing
To get best results, phrase queries with clear intent:
- ✅ "Show prescription **trends** over time" → trend_analysis
- ✅ "**Compare** Q1 vs Q2" → comparison
- ✅ "**Recent** activities" → activity_stream
- ✅ "**Dashboard** of metrics" → dashboard

### 2. Data Structure
System adapts better with:
- Clear column names (date, time, created_at for temporal)
- Numeric columns for metrics
- Categorical columns for breakdowns
- Mixed data types for comprehensive views

### 3. Customization
Adjust LLM behavior in `visualization_planner.py`:
```python
# Line 206: Adjust temperature
temperature=0.2,  # Lower = more consistent, Higher = more creative

# Lines 50-57: Add query patterns
self.query_patterns = {
    'trend': ['trend', 'over time', ...],
    'custom': ['your', 'keywords']  # Add yours
}
```

---

## 🎉 Summary

### What You Got:
✅ **Dynamic**: Every query gets custom visualization plan
✅ **LLM-Powered**: GPT-4o-mini decides optimal layout
✅ **Adaptive**: Responds to query intent and data structure
✅ **Comprehensive**: Multi-component layouts (KPIs + charts + timelines)
✅ **Production-Ready**: Error handling, fallbacks, validation
✅ **Integrated**: Works seamlessly with existing system

### What's Next:
📊 Frontend components to render the comprehensive plans
🎨 UI/UX enhancements
🚀 User testing and feedback

---

**Status**: ✅ Backend Complete & Ready
**Your system now intelligently plans comprehensive, adaptive visualizations based on user queries!**

---

*Questions? Check INTELLIGENT_VISUALIZATION_IMPLEMENTATION_SUMMARY.md for complete technical details.*
