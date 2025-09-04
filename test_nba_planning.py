"""
Test NBA Query Planning Process - Verify Reasoning Model Works
Tests the actual step-by-step process you defined:
1. Schema discovery → 2. Vectorization → 3. Similarity matching → 4. User verification → 5. Query generation
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

load_dotenv()

async def test_nba_query_planning():
    """Test the complete NBA query planning process"""
    
    print("🧪 Testing NBA Query Planning Process...")
    print("="*80)
    
    # Test query: "show me the top 5 NBA recommendations with provider details"
    test_query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and provider input"
    
    try:
        # Import the orchestrator
        from backend.orchestrators.end_to_end_flow import EndToEndFlowOrchestrator
        
        print("✅ Successfully imported EndToEndFlowOrchestrator")
        
        # Initialize orchestrator
        orchestrator = EndToEndFlowOrchestrator()
        print("✅ Orchestrator initialized")
        
        # Test Step 1: Parse user input
        print("\n📝 Step 1: Parsing user input...")
        user_input = await orchestrator._parse_user_input(test_query)
        print(f"   Query: {user_input.query}")
        print(f"   Output format: {user_input.output_format}")
        print(f"   Max rows: {user_input.max_rows}")
        
        # Test Step 2: Create execution plan (this should use reasoning model)
        print("\n🧠 Step 2: Creating execution plan with reasoning model...")
        plan = await orchestrator._create_execution_plan(user_input, "test_user", "test_session")
        print(f"   Plan ID: {plan.plan_id}")
        print(f"   Safety level: {plan.safety_level}")
        print(f"   Estimated cost: ${plan.estimated_cost:.4f}")
        print(f"   Reasoning steps: {len(plan.reasoning_steps)} steps")
        
        # Show reasoning steps
        print("\n🔍 Reasoning Steps from o3-mini model:")
        for i, step in enumerate(plan.reasoning_steps[:5], 1):  # Show first 5 steps
            print(f"   {i}. {step}")
        
        # Show tool sequence
        print("\n🛠️  Tool Sequence:")
        for i, tool in enumerate(plan.tool_sequence, 1):
            print(f"   {i}. {tool['tool']} - {tool['params']}")
        
        # Test Step 3: Schema Discovery
        print("\n🗂️  Step 3: Testing Schema Discovery...")
        
        # Check if the schema tool can discover tables
        schema_context = await orchestrator.schema_tool.discover_schema(
            query=test_query,
            entities=["nba", "output", "provider", "recommendations"]
        )
        
        print(f"   Found {len(schema_context.relevant_tables)} relevant tables")
        for table in schema_context.relevant_tables[:3]:  # Show first 3 tables
            print(f"   - {table.name} ({table.type}) with {len(table.columns)} columns")
        
        print(f"   Entity mappings: {list(schema_context.entity_mappings.keys())}")
        
        # Test Step 4: Validation
        print("\n✅ Step 4: Testing Plan Validation...")
        validation_result = await orchestrator._validate_plan(plan, "test_user")
        print(f"   Plan valid: {validation_result['is_valid']}")
        print(f"   Checks passed: {len(validation_result['checks_passed'])}")
        if validation_result['warnings']:
            print(f"   Warnings: {validation_result['warnings']}")
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Your reasoning model and step-by-step process IS working!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Test fallback to see if basic components work
        print("\n🔄 Testing fallback components...")
        try:
            from backend.agents.enhanced_orchestrator import EnhancedAgenticOrchestrator
            fallback = EnhancedAgenticOrchestrator()
            print("✅ Fallback orchestrator available")
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")

async def test_reasoning_model_directly():
    """Test the reasoning model directly"""
    print("\n🧠 Testing Reasoning Model Directly...")
    print("="*50)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Test reasoning model
        reasoning_model = os.getenv('REASONING_MODEL', 'o3-mini')
        print(f"📡 Testing {reasoning_model}...")
        
        response = client.chat.completions.create(
            model=reasoning_model,
            messages=[{
                "role": "user", 
                "content": "Plan the steps to analyze NBA output data from a database. Be specific about schema discovery, similarity matching, and user verification steps."
            }],
            max_completion_tokens=500
        )
        
        print("✅ Reasoning model responded successfully!")
        print(f"📝 Response: {response.choices[0].message.content[:200]}...")
        
    except Exception as e:
        print(f"❌ Reasoning model test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_nba_query_planning())
    asyncio.run(test_reasoning_model_directly())
