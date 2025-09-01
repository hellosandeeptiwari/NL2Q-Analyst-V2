#!/usr/bin/env python3
"""
Simple OpenAI API test
"""
import os
from dotenv import load_dotenv
load_dotenv()

def test_openai_api():
    """Test OpenAI API with the correct version"""
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"🔑 OpenAI API Key: {'✅ Found' if api_key else '❌ Not found'}")
    
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return False
    
    try:
        import openai
        openai.api_key = api_key
        
        print(f"🔑 API Key starts with: {api_key[:10]}...")
        
        # Test embedding
        response = openai.Embedding.create(
            input="NBA basketball data table",
            model="text-embedding-3-small"
        )
        
        embedding = response['data'][0]['embedding']
        print(f"✅ OpenAI API test successful!")
        print(f"📊 Embedding dimension: {len(embedding)}")
        print(f"📊 First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

def test_table_matching():
    """Test simple table matching logic"""
    print("\n🔍 Testing table matching logic...")
    
    # Sample table names
    sample_tables = [
        "Final_NBA_Output_python_20250519",
        "NBA_Player_Stats_2024",
        "Customer_Orders_2024",
        "Product_Inventory",
        "Final_NBA_Features_ML"
    ]
    
    query = "NBA basketball data"
    print(f"Query: '{query}'")
    
    # Simple keyword matching
    matches = []
    for table in sample_tables:
        score = 0
        query_words = query.lower().split()
        table_lower = table.lower()
        
        for word in query_words:
            if word in table_lower:
                score += 1
        
        if score > 0:
            matches.append((table, score))
    
    # Sort by score
    matches.sort(key=lambda x: x[1], reverse=True)
    
    print("📋 Matches found:")
    for table, score in matches:
        print(f"  • {table} (score: {score})")
    
    return len(matches) > 0

if __name__ == "__main__":
    print("🧪 OpenAI Integration Test\n")
    
    # Test OpenAI API
    openai_works = test_openai_api()
    
    # Test table matching
    matching_works = test_table_matching()
    
    print(f"\n📊 Test Results:")
    print(f"  OpenAI API: {'✅' if openai_works else '❌'}")
    print(f"  Table Matching: {'✅' if matching_works else '❌'}")
    
    if openai_works:
        print("\n🎉 OpenAI integration is working! Ready to build vector embeddings.")
    else:
        print("\n⚠️ OpenAI integration needs attention. Check API key and internet connection.")
