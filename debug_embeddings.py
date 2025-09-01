#!/usr/bin/env python3
"""
Debug OpenAI embedding API issue
"""
import os
import openai
from dotenv import load_dotenv

load_dotenv()

def test_embedding_api():
    """Test the embedding API with different formats"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    openai.api_key = api_key
    print(f"🔑 API Key: {api_key[:10]}...")
    print(f"📦 OpenAI Version: {openai.__version__}")
    
    # Test 1: Single input
    print("\n🧪 Test 1: Single text input...")
    try:
        response = openai.Embedding.create(
            input="NBA basketball data table",
            model="text-embedding-3-small"
        )
        print("✅ Single input works!")
        print(f"   Dimension: {len(response['data'][0]['embedding'])}")
    except Exception as e:
        print(f"❌ Single input failed: {e}")
    
    # Test 2: List input (current format)
    print("\n🧪 Test 2: List input...")
    try:
        response = openai.Embedding.create(
            input=["NBA basketball data table", "Player statistics"],
            model="text-embedding-3-small"
        )
        print("✅ List input works!")
        print(f"   Got {len(response['data'])} embeddings")
    except Exception as e:
        print(f"❌ List input failed: {e}")
    
    # Test 3: Alternative model
    print("\n🧪 Test 3: Alternative model...")
    try:
        response = openai.Embedding.create(
            input="NBA basketball data table",
            model="text-embedding-ada-002"
        )
        print("✅ Alternative model works!")
    except Exception as e:
        print(f"❌ Alternative model failed: {e}")

if __name__ == "__main__":
    test_embedding_api()
