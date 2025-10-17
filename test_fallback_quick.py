"""Quick test to verify fallback preserves temporal context"""
import sys
import os
import asyncio
from dotenv import load_dotenv
import pandas as pd

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agents.visualization_planner import VisualizationPlanner

async def test_fallback():
    planner = VisualizationPlanner()
    
    # Contextual query
    query = "Show me top 10 prescribers by Nrx"
    data = pd.DataFrame({
        "prescriber_name": [f"Dr. {i}" for i in range(10)],
        "nrx": [450 - i*10 for i in range(10)],
        "specialty": ["Cardiology"] * 10
    })
    
    plan = await planner.plan_visualization(query, data, {})
    
    print("\n" + "="*60)
    print("FALLBACK TEMPORAL CONTEXT CHECK")
    print("="*60)
    print(f"Plan has temporal_context: {plan.temporal_context is not None}")
    
    if plan.temporal_context:
        tc = plan.temporal_context
        print(f"✅ Temporal context exists!")
        print(f"   Enabled: {tc.enabled}")
        print(f"   Context Type: {tc.context_type}")
        print(f"   Query Timeframe: {tc.query_timeframe}")
        print(f"   Insight Type: {tc.insight_type}")
        print(f"   Number of comparison periods: {len(tc.comparison_periods) if tc.comparison_periods else 0}")
        if tc.comparison_periods:
            for card in tc.comparison_periods:
                print(f"      - {card.time_period}")
    else:
        print("❌ No temporal context in plan!")
        print(f"   Metadata: {plan.metadata.get('temporal_context', 'Not in metadata')}")

if __name__ == "__main__":
    asyncio.run(test_fallback())
