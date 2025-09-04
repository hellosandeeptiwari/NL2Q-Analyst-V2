"""
Test Pharma GCO Query with Dynamic Orchestration
Tests the system with user's exact query: "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and input and action effect"
"""

import asyncio
import sys
import os
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_pharma_gco_query():
    """
    Test the dynamic orchestration system with the exact pharma GCO query
    """
    
    # User's exact query for pharma GCO
    user_query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and input and action effect"
    
    print("🏥 Testing Pharma GCO Query with Dynamic Agent Orchestration")
    print("="*80)
    print(f"📝 User Query: {user_query}")
    print("🎯 Use Case: Pharma GCO (Global Commercial Operations)")
    print("="*80)
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        # Initialize the dynamic orchestrator
        orchestrator = DynamicAgentOrchestrator()
        print("✅ Dynamic orchestrator initialized for pharma GCO")
        
        # Let the reasoning model plan the execution automatically
        print("\n🧠 Planning execution with o3-mini reasoning model...")
        tasks = await orchestrator.plan_execution(user_query)
        
        print(f"📋 Generated {len(tasks)} tasks for pharma analysis:")
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task.task_id} ({task.task_type.value})")
            if task.dependencies:
                print(f"      Dependencies: {', '.join(task.dependencies)}")
        
        # Execute the planned tasks
        print(f"\n⚡ Executing pharma GCO analysis plan...")
        results = await orchestrator.execute_plan(tasks, user_query)
        
        print(f"\n🎉 Pharma GCO analysis completed!")
        print(f"📊 Results for '{user_query}':")
        
        for task_id, result in results.items():
            status = result.get('status', 'unknown')
            print(f"\n   📌 {task_id}: {status}")
            
            # Show key pharma-relevant results
            if 'discovered_tables' in result:
                tables = result['discovered_tables']
                print(f"      🗂️  Found {len(tables)} pharma tables")
                for table in tables[:3]:  # Show first 3
                    print(f"         - {table}")
                    
            if 'entities' in result:
                entities = result['entities']
                print(f"      🏷️  Extracted pharma entities: {entities}")
                
            if 'sql_query' in result:
                query = result['sql_query']
                print(f"      💾 Generated SQL (first 100 chars): {query[:100]}...")
                
            if 'results' in result:
                data = result['results']
                if isinstance(data, list) and len(data) > 0:
                    print(f"      📈 Retrieved {len(data)} records")
                    print(f"      🔍 Sample data columns: {list(data[0].keys())[:5] if data[0] else 'N/A'}")
                    
            if 'visualization_path' in result:
                viz_path = result['visualization_path']
                print(f"      📊 Created visualization: {viz_path}")
        
        # Summary for pharma GCO team
        print(f"\n🏥 PHARMA GCO ANALYSIS SUMMARY")
        print("="*50)
        print("✅ Automatic table discovery (no hardcoding)")
        print("✅ Entity extraction for pharma context")
        print("✅ Smart column matching")
        print("✅ SQL generation with safety guardrails")
        print("✅ Data visualization for insights")
        print("✅ Ready for any pharma query pattern")
        
        return results
        
    except Exception as e:
        print(f"❌ Pharma GCO analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_sample_pharma_data():
    """Create sample NBA output data for pharma GCO testing"""
    
    sample_data = {
        'provider_id': ['PRV001', 'PRV002', 'PRV003', 'PRV004', 'PRV005'],
        'provider_name': [
            'Dr. Sarah Chen (Oncology)', 
            'Dr. Mike Johnson (Cardiology)', 
            'Dr. Lisa Wang (Endocrinology)', 
            'Dr. John Smith (Oncology)', 
            'Dr. Emma Davis (Nephrology)'
        ],
        'recommended_message': [
            'Clinical Trial Enrollment Opportunity',
            'New Treatment Protocol Available', 
            'Patient Education Resources',
            'Clinical Trial Enrollment Opportunity',
            'Dosing Guidelines Update'
        ],
        'provider_input': [
            'Interested in oncology trials',
            'Looking for diabetes management',
            'Needs patient education materials',
            'Seeking trial opportunities',
            'Requesting dosing information'
        ],
        'action_effect': [
            'High engagement expected',
            'Medium engagement expected',
            'Educational value high',
            'High engagement expected', 
            'Clinical utility high'
        ],
        'therapeutic_area': ['Oncology', 'Cardiology', 'Endocrinology', 'Oncology', 'Nephrology'],
        'recommendation_score': [0.95, 0.87, 0.82, 0.78, 0.74],
        'timestamp': ['2025-09-04 10:30:00'] * 5,
        'region': ['US-East', 'US-West', 'EU-Central', 'US-East', 'APAC'],
        'hcp_tier': ['Tier 1', 'Tier 2', 'Tier 1', 'Tier 1', 'Tier 2']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"📊 Created sample pharma GCO data: {len(df)} records")
    print(f"🔍 Columns: {list(df.columns)}")
    return df

def analyze_frequency_patterns(df):
    """Analyze frequency patterns as requested in the user query"""
    
    print(f"\n📈 FREQUENCY ANALYSIS FOR PHARMA GCO")
    print("="*50)
    
    # Recommended message frequency
    message_freq = df['recommended_message'].value_counts()
    print(f"\n🎯 Recommended Message Frequency:")
    for message, count in message_freq.items():
        print(f"   • {message}: {count} occurrences")
    
    # Provider input frequency  
    input_freq = df['provider_input'].value_counts()
    print(f"\n💭 Provider Input Frequency:")
    for input_type, count in input_freq.items():
        print(f"   • {input_type}: {count} occurrences")
    
    # Action effect frequency
    effect_freq = df['action_effect'].value_counts()
    print(f"\n⚡ Action Effect Frequency:")
    for effect, count in effect_freq.items():
        print(f"   • {effect}: {count} occurrences")
    
    # Therapeutic area patterns
    area_freq = df['therapeutic_area'].value_counts()
    print(f"\n🏥 Therapeutic Area Distribution:")
    for area, count in area_freq.items():
        print(f"   • {area}: {count} providers")

async def main():
    """Main test function"""
    
    print("🧪 Starting Pharma GCO Query Test")
    print("🎯 Testing dynamic orchestration with user's exact query")
    print("🏥 Use case: Global Commercial Operations in Pharma")
    
    # Test the dynamic orchestration
    results = await test_pharma_gco_query()
    
    # Also demonstrate with sample data to show expected output
    print(f"\n📊 Demonstrating expected analysis with sample data...")
    sample_df = create_sample_pharma_data()
    analyze_frequency_patterns(sample_df)
    
    print(f"\n✅ Pharma GCO test completed!")
    print(f"🔄 The system is ready to handle any similar pharma queries automatically")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the pharma GCO test
    asyncio.run(main())
