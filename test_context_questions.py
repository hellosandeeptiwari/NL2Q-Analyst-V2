#!/usr/bin/env python3
"""
Test script to verify context-aware follow-up question functionality
"""

import requests
import json

# Test the intent detection endpoint with context
def test_context_aware_intent():
    """Test that the system can handle context-aware questions"""
    
    base_url = "http://localhost:8000"
    
    # Simulate context from a completed analysis
    mock_context = {
        "hasCharts": True,
        "hasTable": True,
        "chartTypes": ["plotly", "matplotlib"],
        "keyInsights": [
            "Sales increased 25% in Q4",
            "Top performer: Product ABC",
            "Seasonal trend identified"
        ],
        "lastAnalysis": "Analysis of pharmaceutical sales data showing strong Q4 performance with Product ABC leading revenue growth. Clear seasonal patterns observed across all product categories."
    }
    
    # Test scenarios
    test_cases = [
        {
            "name": "Context Question - Chart Explanation",
            "query": "What does this chart show?",
            "context": mock_context,
            "expected_planning": False,
            "expected_context": True
        },
        {
            "name": "Context Question - Data Details", 
            "query": "What caused the spike in sales?",
            "context": mock_context,
            "expected_planning": False,
            "expected_context": True
        },
        {
            "name": "New Analysis Request",
            "query": "Show me data for last year",
            "context": mock_context,
            "expected_planning": True,
            "expected_context": False
        },
        {
            "name": "Casual Conversation",
            "query": "Thank you for the analysis",
            "context": mock_context,
            "expected_planning": False,
            "expected_context": False
        },
        {
            "name": "Follow-up Insight",
            "query": "Tell me more about Product ABC performance",
            "context": mock_context,
            "expected_planning": False,
            "expected_context": True
        }
    ]
    
    print("🧪 Testing Context-Aware Intent Detection")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\n📝 Test: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        
        try:
            response = requests.post(
                f"{base_url}/api/agent/detect-intent",
                json={
                    "query": test_case["query"],
                    "context": test_case["context"]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {json.dumps(result, indent=2)}")
                
                # Validate expectations
                planning_correct = result.get("needsPlanning") == test_case["expected_planning"]
                context_correct = result.get("isContextQuestion") == test_case["expected_context"]
                
                if planning_correct and context_correct:
                    print("✅ Test PASSED - Intent correctly detected")
                else:
                    print("❌ Test FAILED - Intent detection mismatch")
                    print(f"   Expected planning: {test_case['expected_planning']}, got: {result.get('needsPlanning')}")
                    print(f"   Expected context: {test_case['expected_context']}, got: {result.get('isContextQuestion')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection failed - Is the server running on localhost:8000?")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Context-aware testing complete!")

if __name__ == "__main__":
    test_context_aware_intent()
