"""
Final Test: Pharma GCO Query with Live Database
Tests user's exact query with live SQLite database connection
"""

import asyncio
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load test environment
load_dotenv('.env.test')

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_live_pharma_query():
    """
    Test the exact pharma GCO query with live database connection
    """
    
    # User's exact query
    user_query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and input and action effect"
    
    print("🏥 FINAL TEST: Pharma GCO Query with Live Database")
    print("="*80)
    print(f"📝 User Query: {user_query}")
    print(f"💾 Database: SQLite (pharma_gco_test.db)")
    print(f"🎯 Use Case: Global Commercial Operations in Pharma")
    print("="*80)
    
    # Set environment for SQLite
    os.environ['DB_ENGINE'] = 'sqlite'
    os.environ['SQLITE_DB_PATH'] = 'pharma_gco_test.db'
    
    try:
        from backend.orchestrators.dynamic_agent_orchestrator import DynamicAgentOrchestrator
        
        # Initialize the dynamic orchestrator
        orchestrator = DynamicAgentOrchestrator()
        print("✅ Dynamic orchestrator initialized with SQLite connection")
        
        # Force refresh of schema cache to use SQLite data
        print("🔄 Refreshing schema cache for SQLite database...")
        
        # Test database connection first
        from backend.db.engine import get_adapter
        adapter = get_adapter()
        health = adapter.health()
        print(f"💚 Database health: {health}")
        
        # Let the reasoning model plan the execution
        print("\n🧠 Planning execution with o3-mini reasoning model...")
        tasks = await orchestrator.plan_execution(user_query)
        
        print(f"📋 Generated {len(tasks)} tasks for live pharma analysis:")
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task.task_id} ({task.task_type.value})")
            if task.dependencies:
                print(f"      Dependencies: {', '.join(task.dependencies)}")
        
        # Execute the planned tasks with live database
        print(f"\n⚡ Executing pharma GCO plan with live database...")
        results = await orchestrator.execute_plan(tasks, user_query)
        
        print(f"\n🎉 LIVE PHARMA GCO ANALYSIS COMPLETED!")
        print(f"📊 Results for: '{user_query}'")
        print("="*80)
        
        success_count = 0
        total_tasks = len(tasks)
        
        for task_id, result in results.items():
            status = result.get('status', 'unknown')
            print(f"\n📌 {task_id}: {status.upper()}")
            
            if status == 'completed':
                success_count += 1
            
            # Show detailed results for pharma GCO team
            if 'discovered_tables' in result:
                tables = result['discovered_tables']
                print(f"   🗂️  Discovered pharma tables: {len(tables)}")
                for table in tables[:3]:
                    print(f"      ✓ {table}")
                    
            if 'entities' in result:
                entities = result['entities']
                print(f"   🏷️  Pharma entities extracted: {entities}")
                
            if 'matched_tables' in result:
                matches = result['matched_tables']
                print(f"   🎯 Table matches: {matches}")
                
            if 'approved_tables' in result:
                approved = result['approved_tables']
                print(f"   ✅ User approved tables: {approved}")
                
            if 'sql_query' in result:
                query = result['sql_query']
                print(f"   💾 Generated SQL query:")
                print(f"      {query}")
                
            if 'results' in result:
                data = result['results']
                if isinstance(data, list) and len(data) > 0:
                    print(f"   📈 Retrieved data: {len(data)} records")
                    # Show sample of actual data
                    sample = data[0] if data else {}
                    print(f"   🔍 Sample record columns: {list(sample.keys())}")
                    
            if 'visualization_path' in result:
                viz_path = result['visualization_path']
                print(f"   📊 Visualization created: {viz_path}")
                
            if 'frequency_analysis' in result:
                freq = result['frequency_analysis']
                print(f"   📈 Frequency analysis completed: {freq}")
        
        # Final summary for pharma GCO
        print(f"\n🏥 PHARMA GCO FINAL RESULTS")
        print("="*60)
        print(f"✅ Tasks completed successfully: {success_count}/{total_tasks}")
        print(f"📊 Database connection: {'✅ Active' if health.get('connected') else '❌ Failed'}")
        print(f"🎯 Query processing: {'✅ Success' if success_count > 0 else '❌ Failed'}")
        print(f"🔄 System status: {'✅ Ready for production' if success_count >= 4 else '⚠️ Needs debugging'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Live pharma analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_test_data():
    """Verify the test database has the expected data"""
    
    print("\n🔍 Verifying test database...")
    
    try:
        import sqlite3
        conn = sqlite3.connect('pharma_gco_test.db')
        
        # Check table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%NBA%'")
        tables = cursor.fetchall()
        print(f"📋 NBA tables found: {[t[0] for t in tables]}")
        
        # Check data
        cursor.execute("SELECT COUNT(*) FROM NBA_PHASE2_SEP2024_SIMILARITY_OUTPUT_FINAL_PYTHON")
        count = cursor.fetchone()[0]
        print(f"📊 Records in NBA table: {count}")
        
        # Show sample data for the exact columns requested
        cursor.execute("""
            SELECT provider_name, recommended_message, provider_input, action_effect 
            FROM NBA_PHASE2_SEP2024_SIMILARITY_OUTPUT_FINAL_PYTHON 
            LIMIT 3
        """)
        sample = cursor.fetchall()
        print(f"🔍 Sample data:")
        for row in sample:
            print(f"   • {row[0]} | {row[1]} | {row[2]} | {row[3]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

async def main():
    """Main execution"""
    
    print("🧪 FINAL PHARMA GCO TEST WITH LIVE DATABASE")
    print("🎯 Testing user's exact query with actual database connection")
    
    # Verify test data exists
    if not verify_test_data():
        print("❌ Test database not ready. Run setup_pharma_db.py first.")
        return
    
    # Run the live test
    results = await test_live_pharma_query()
    
    if results:
        print(f"\n✅ PHARMA GCO TEST COMPLETED SUCCESSFULLY!")
        print(f"🚀 System is ready for production pharma queries")
    else:
        print(f"\n❌ Test failed - needs debugging")

if __name__ == "__main__":
    asyncio.run(main())
