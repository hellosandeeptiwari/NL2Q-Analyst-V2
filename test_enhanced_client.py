#!/usr/bin/env python3
"""
Test client for the enhanced NBA data analysis server
Demonstrates the complete workflow from schema to insights
"""
import requests
import json
import time
from typing import Dict, Any

class NBAAnalysisClient:
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url
        
    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_table_suggestions(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Get table suggestions using semantic search"""
        try:
            response = requests.get(
                f"{self.base_url}/suggest-tables",
                params={"query": query, "top_k": top_k},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_code(self, query: str, tables: list = None) -> Dict[str, Any]:
        """Generate analysis code"""
        try:
            payload = {"query": query}
            if tables:
                payload["tables"] = tables
                
            response = requests.post(
                f"{self.base_url}/generate-code",
                json=payload,
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Complete analysis workflow"""
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                json={"query": query},
                timeout=60
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_dataframes(self) -> Dict[str, Any]:
        """List stored dataframes"""
        try:
            response = requests.get(f"{self.base_url}/dataframes", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def test_complete_workflow():
    """Test the complete analysis workflow"""
    client = NBAAnalysisClient()
    
    print("🧪 Testing Enhanced NBA Data Analysis Workflow\n")
    
    # 1. Health check
    print("1. 🏥 Health Check")
    health = client.health_check()
    if "error" in health:
        print(f"❌ Health check failed: {health['error']}")
        return
    
    print(f"✅ Server healthy")
    print(f"   Database: {health.get('database', {}).get('connected', 'Unknown')}")
    print(f"   System: {health.get('system', {}).get('initialized', 'Unknown')}")
    
    # 2. Table suggestions
    print("\n2. 🔍 Testing Table Suggestions")
    test_queries = [
        "NBA basketball player statistics",
        "Player scoring data with points and rebounds",
        "Basketball game results and team performance"
    ]
    
    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        suggestions = client.get_table_suggestions(query, top_k=3)
        
        if "error" in suggestions:
            print(f"❌ Error: {suggestions['error']}")
        else:
            print(f"✅ Found {len(suggestions.get('suggestions', []))} suggestions:")
            for suggestion in suggestions.get('suggestions', [])[:3]:
                print(f"   • {suggestion['table_name']} (similarity: {suggestion['similarity_score']:.3f})")
                print(f"     {suggestion['description'][:100]}...")
    
    # 3. Code generation
    print("\n3. 🧠 Testing Code Generation")
    code_query = "Show me the top 10 NBA players by points per game with their team names"
    print(f"📋 Query: '{code_query}'")
    
    code_result = client.generate_code(code_query)
    if "error" in code_result:
        print(f"❌ Code generation failed: {code_result['error']}")
    else:
        print("✅ Code generated successfully")
        if "generated_code" in code_result and "sql_query" in code_result["generated_code"]:
            sql = code_result["generated_code"]["sql_query"]
            print(f"📊 SQL Query:\n{sql[:200]}..." if len(sql) > 200 else f"📊 SQL Query:\n{sql}")
        
        if "generated_code" in code_result and "code" in code_result["generated_code"]:
            python_code = code_result["generated_code"]["code"]
            print(f"🐍 Python Code:\n{python_code[:300]}..." if len(python_code) > 300 else f"🐍 Python Code:\n{python_code}")
    
    # 4. Complete analysis (if previous steps worked)
    print("\n4. 🔬 Testing Complete Analysis Workflow")
    analysis_query = "Analyze NBA player performance: show distribution of points scored"
    print(f"📋 Analysis Query: '{analysis_query}'")
    
    start_time = time.time()
    analysis_result = client.analyze(analysis_query)
    analysis_time = time.time() - start_time
    
    if "error" in analysis_result:
        print(f"❌ Analysis failed: {analysis_result['error']}")
    else:
        print(f"✅ Analysis completed in {analysis_time:.2f}s")
        print(f"📊 Steps completed: {len(analysis_result.get('steps', []))}")
        
        for i, step in enumerate(analysis_result.get('steps', []), 1):
            print(f"   {i}. {step}")
        
        # Show results
        if "data_loaded" in analysis_result:
            data_info = analysis_result["data_loaded"]
            print(f"📈 Data loaded: {data_info.get('shape', 'Unknown shape')}")
        
        if "analysis_result" in analysis_result:
            analysis_info = analysis_result["analysis_result"]
            if "new_dataframes" in analysis_info:
                print(f"🗂️ DataFrames created: {len(analysis_info['new_dataframes'])}")
    
    # 5. Check stored dataframes
    print("\n5. 🗂️ Checking Stored DataFrames")
    dataframes = client.get_dataframes()
    if "error" in dataframes:
        print(f"❌ Error getting dataframes: {dataframes['error']}")
    else:
        total = dataframes.get('total_dataframes', 0)
        print(f"✅ {total} dataframes in memory")
        
        for name, info in dataframes.get('dataframes', {}).items():
            print(f"   • {name}: {info['shape']} ({info['memory_mb']} MB)")

def main():
    """Run the test workflow"""
    print("🚀 Enhanced NBA Data Analysis - Test Client")
    print("=" * 50)
    
    try:
        test_complete_workflow()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🏁 Test completed")

if __name__ == "__main__":
    main()
