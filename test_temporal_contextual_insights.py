"""
Test script for Temporal and Contextual Insights Feature
Tests the visualization planner's ability to detect and generate:
1. Temporal context for time-based queries
2. Contextual insights for non-temporal queries
"""

import sys
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from backend.agents.visualization_planner import VisualizationPlanner
import pandas as pd

async def test_temporal_query():
    """Test temporal context detection for time-based queries"""
    print("\n" + "="*80)
    print("TEST 1: TEMPORAL QUERY - 'Show me sales for last quarter vs previous quarter'")
    print("="*80)
    
    planner = VisualizationPlanner()
    
    # Simulate query and data
    query = "Show me total sales for last quarter compared to previous quarter"
    
    sample_data = [
        {"month": "2024-Q1", "total_sales": 150000, "region": "North"},
        {"month": "2024-Q2", "total_sales": 180000, "region": "North"},
        {"month": "2024-Q3", "total_sales": 220000, "region": "North"},
        {"month": "2024-Q4", "total_sales": 250000, "region": "North"},
        {"month": "2024-Q1", "total_sales": 120000, "region": "South"},
        {"month": "2024-Q2", "total_sales": 140000, "region": "South"},
        {"month": "2024-Q3", "total_sales": 160000, "region": "South"},
        {"month": "2024-Q4", "total_sales": 190000, "region": "South"},
    ]
    
    try:
        df = pd.DataFrame(sample_data)
        plan = await planner.plan_visualization(query, df, {})
        
        if plan:
            print("\n✅ Visualization plan created successfully!")
            
            print(f"\n📊 Layout Type: {plan.layout_type}")
            print(f"🎯 Query Type: {plan.query_type}")
            
            if plan.temporal_context and plan.temporal_context.enabled:
                tc = plan.temporal_context
                print(f"\n⏰ TEMPORAL CONTEXT DETECTED:")
                print(f"   Context Type: {getattr(tc, 'context_type', 'temporal')}")
                print(f"   Query Timeframe: {getattr(tc, 'query_timeframe', 'N/A')}")
                print(f"   Time Granularity: {getattr(tc, 'time_granularity', 'N/A')}")
                print(f"   Number of Comparison Periods: {len(tc.comparison_periods)}")
                
                print(f"\n   Comparison Cards:")
                for i, card in enumerate(tc.comparison_periods, 1):
                    # Cards are strings, not objects
                    print(f"   {i}. {card}")
            else:
                print("\n⚠️  No temporal context detected")
            
            print(f"\n📈 KPIs: {len(plan.kpis)}")
            for kpi in plan.kpis:
                print(f"   - {kpi.title}")
                if kpi.time_period:
                    print(f"     Time Period: {kpi.time_period}")
                if kpi.comparison_text:
                    print(f"     Comparison: {kpi.comparison_text}")
            
            print(f"\n📉 Primary Chart: {plan.primary_chart.type} - {plan.primary_chart.title}")
            
        else:
            print(f"\n❌ Planning failed: No plan returned")
                
    except Exception as e:
        print(f"\n❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_contextual_top_query():
    """Test contextual insights for top/bottom queries"""
    print("\n" + "="*80)
    print("TEST 2: CONTEXTUAL QUERY - 'Show me top 10 prescribers by Nrx, Trx'")
    print("="*80)
    
    planner = VisualizationPlanner()
    
    query = "Show me top 10 prescribers by Nrx and Trx"
    
    sample_data = [
        {"prescriber_name": "Dr. Smith", "nrx": 450, "trx": 890, "specialty": "Cardiology", "region": "Northeast"},
        {"prescriber_name": "Dr. Johnson", "nrx": 420, "trx": 850, "specialty": "Cardiology", "region": "West"},
        {"prescriber_name": "Dr. Williams", "nrx": 380, "trx": 780, "specialty": "Internal Medicine", "region": "South"},
        {"prescriber_name": "Dr. Brown", "nrx": 360, "trx": 720, "specialty": "Family Medicine", "region": "Midwest"},
        {"prescriber_name": "Dr. Davis", "nrx": 340, "trx": 680, "specialty": "Cardiology", "region": "Northeast"},
        {"prescriber_name": "Dr. Miller", "nrx": 320, "trx": 650, "specialty": "Internal Medicine", "region": "West"},
        {"prescriber_name": "Dr. Wilson", "nrx": 300, "trx": 610, "specialty": "Family Medicine", "region": "South"},
        {"prescriber_name": "Dr. Moore", "nrx": 280, "trx": 570, "specialty": "Cardiology", "region": "Midwest"},
        {"prescriber_name": "Dr. Taylor", "nrx": 260, "trx": 530, "specialty": "Internal Medicine", "region": "Northeast"},
        {"prescriber_name": "Dr. Anderson", "nrx": 240, "trx": 490, "specialty": "Family Medicine", "region": "West"},
    ]
    
    try:
        df = pd.DataFrame(sample_data)
        plan = await planner.plan_visualization(query, df, {})
        
        if plan:
            print("\n✅ Visualization plan created successfully!")
            
            print(f"\n📊 Layout Type: {plan.layout_type}")
            print(f"🎯 Query Type: {plan.query_type}")
            
            if plan.temporal_context and plan.temporal_context.enabled:
                tc = plan.temporal_context
                print(f"\n📊 CONTEXTUAL INSIGHTS DETECTED:")
                print(f"   Context Type: {getattr(tc, 'context_type', 'contextual')}")
                print(f"   Query Timeframe: {getattr(tc, 'query_timeframe', 'N/A')}")
                print(f"   Insight Type: {getattr(tc, 'insight_type', 'N/A')}")
                print(f"   Number of Comparison Cards: {len(tc.comparison_periods)}")
                
                print(f"\n   Contextual Comparison Cards:")
                for i, card in enumerate(tc.comparison_periods, 1):
                    # Cards are strings, not objects
                    print(f"   {i}. {card}")
            else:
                print("\n⚠️  No contextual insights detected")
            
            print(f"\n📈 KPIs: {len(plan.kpis)}")
            for kpi in plan.kpis:
                print(f"   - {kpi.title}")
            
            print(f"\n📉 Primary Chart: {plan.primary_chart.type} - {plan.primary_chart.title}")
            
        else:
            print(f"\n❌ Planning failed: No plan returned")
                
    except Exception as e:
        print(f"\n❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_contextual_categorical_query():
    """Test contextual insights for categorical breakdown queries"""
    print("\n" + "="*80)
    print("TEST 3: CONTEXTUAL QUERY - 'Show me product sales breakdown'")
    print("="*80)
    
    planner = VisualizationPlanner()
    
    query = "Show me product sales breakdown by category"
    
    sample_data = [
        {"product": "Product A", "category": "Cardiovascular", "sales": 1500000, "region": "National"},
        {"product": "Product B", "category": "Diabetes", "sales": 1200000, "region": "National"},
        {"product": "Product C", "category": "Oncology", "sales": 1800000, "region": "National"},
        {"product": "Product D", "category": "Cardiovascular", "sales": 900000, "region": "National"},
        {"product": "Product E", "category": "Diabetes", "sales": 1100000, "region": "National"},
    ]
    
    try:
        df = pd.DataFrame(sample_data)
        plan = await planner.plan_visualization(query, df, {})
        
        if plan:
            print("\n✅ Visualization plan created successfully!")
            
            print(f"\n📊 Layout Type: {plan.layout_type}")
            print(f"🎯 Query Type: {plan.query_type}")
            
            if plan.temporal_context and plan.temporal_context.enabled:
                tc = plan.temporal_context
                print(f"\n📊 CONTEXTUAL INSIGHTS DETECTED:")
                print(f"   Context Type: {getattr(tc, 'context_type', 'contextual')}")
                print(f"   Query Timeframe: {getattr(tc, 'query_timeframe', 'N/A')}")
                print(f"   Insight Type: {getattr(tc, 'insight_type', 'N/A')}")
                print(f"   Number of Comparison Cards: {len(tc.comparison_periods)}")
                
                print(f"\n   Contextual Comparison Cards:")
                for i, card in enumerate(tc.comparison_periods, 1):
                    # Cards are strings, not objects
                    print(f"   {i}. {card}")
            else:
                print("\n⚠️  No contextual insights detected")
            
            print(f"\n📈 KPIs: {len(plan.kpis)}")
            print(f"📉 Primary Chart: {plan.primary_chart.type} - {plan.primary_chart.title}")
            
        else:
            print(f"\n❌ Planning failed: No plan returned")
                
    except Exception as e:
        print(f"\n❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_ytd_temporal_query():
    """Test YTD temporal pattern"""
    print("\n" + "="*80)
    print("TEST 4: TEMPORAL QUERY - 'Show me YTD sales vs last year'")
    print("="*80)
    
    planner = VisualizationPlanner()
    
    query = "Show me year-to-date sales compared to last year"
    
    sample_data = [
        {"month": "Jan", "sales_2024": 100000, "sales_2023": 85000},
        {"month": "Feb", "sales_2024": 120000, "sales_2023": 95000},
        {"month": "Mar", "sales_2024": 140000, "sales_2023": 110000},
        {"month": "Apr", "sales_2024": 160000, "sales_2023": 125000},
        {"month": "May", "sales_2024": 180000, "sales_2023": 140000},
    ]
    
    try:
        df = pd.DataFrame(sample_data)
        plan = await planner.plan_visualization(query, df, {})
        
        if plan:
            print("\n✅ Visualization plan created successfully!")
            
            if plan.temporal_context and plan.temporal_context.enabled:
                tc = plan.temporal_context
                print(f"\n⏰ TEMPORAL CONTEXT DETECTED:")
                print(f"   Query Timeframe: {getattr(tc, 'query_timeframe', 'N/A')}")
                print(f"   Time Granularity: {getattr(tc, 'time_granularity', 'N/A')}")
                print(f"\n   Comparison Periods: {len(tc.comparison_periods)}")
                for card in tc.comparison_periods:
                    # Cards are strings, not objects
                    print(f"   - {card}")
            else:
                print("\n⚠️  No temporal context detected")
                
        else:
            print(f"\n❌ Planning failed: No plan returned")
                
    except Exception as e:
        print(f"\n❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    print("\n" + "🧪 "*40)
    print("TESTING TEMPORAL AND CONTEXTUAL INSIGHTS FEATURE")
    print("🧪 "*40)
    
    # Test 1: Temporal query
    await test_temporal_query()
    await asyncio.sleep(1)
    
    # Test 2: Contextual top query
    await test_contextual_top_query()
    await asyncio.sleep(1)
    
    # Test 3: Contextual categorical query
    await test_contextual_categorical_query()
    await asyncio.sleep(1)
    
    # Test 4: YTD temporal query
    await test_ytd_temporal_query()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review the temporal context detection for time-based queries")
    print("2. Review the contextual insights for non-temporal queries")
    print("3. Check frontend integration at http://localhost:3000")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
