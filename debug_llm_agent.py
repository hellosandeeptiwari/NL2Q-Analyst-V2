#!/usr/bin/env python3
"""
Debug script to test LLM Agent functionality
"""
import openai
import os
import json

def test_llm_agent():
    """Test the LLM agent functionality"""
    print("🧪 Testing LLM Agent...")
    
    # Load API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OpenAI API key found in environment")
        return
    
    print("✅ OpenAI API key loaded")
    openai.api_key = api_key
    
    # Test simple completion
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "What is 2+2? Respond with just the number."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ Simple test successful: {result}")
        
        # Test query analysis
        test_query = "Show me NBA player data from the Lakers team"
        tables = ["NBA_PLAYERS", "NBA_TEAMS", "NBA_STATS", "OTHER_TABLE"]
        
        prompt = f"""
You are a data analyst. Analyze this query and respond with valid JSON only:

Query: "{test_query}"
Available tables: {tables}

Respond with JSON containing:
- intent: brief description of what user wants
- recommended_tables: list of best table names
- query_type: type of analysis needed

Response format:
{{"intent": "...", "recommended_tables": [...], "query_type": "..."}}
"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"🔍 Query analysis response: {result}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(result)
            print("✅ JSON parsing successful!")
            print(f"   Intent: {parsed.get('intent', 'N/A')}")
            print(f"   Tables: {parsed.get('recommended_tables', [])}")
            print(f"   Type: {parsed.get('query_type', 'N/A')}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"   Raw response: {result}")
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        print(f"   Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_llm_agent()
