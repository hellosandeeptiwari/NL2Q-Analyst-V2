#!/usr/bin/env python3
"""
Test environment variable loading for Pinecone
"""
import os
from dotenv import load_dotenv

print("Before loading .env:")
print(f"PINECONE_API_KEY from env: {os.getenv('PINECONE_API_KEY')}")

load_dotenv()

print("\nAfter loading .env:")
pinecone_key = os.getenv('PINECONE_API_KEY')
print(f"PINECONE_API_KEY loaded: {bool(pinecone_key)}")
if pinecone_key:
    print(f"Key length: {len(pinecone_key)}")
    print(f"Key starts with: {pinecone_key[:10]}...")
    print(f"Key ends with: ...{pinecone_key[-5:]}")
else:
    print("API key is None or empty")

# Check if file exists and can be read
try:
    with open('.env', 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"\n.env file size: {len(content)} characters")
        if 'PINECONE_API_KEY' in content:
            print("✅ PINECONE_API_KEY found in .env file")
        else:
            print("❌ PINECONE_API_KEY NOT found in .env file")
except Exception as e:
    print(f"Error reading .env file: {e}")

# Try manual parsing
try:
    with open('.env', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'PINECONE_API_KEY' in line:
                print(f"\nFound PINECONE_API_KEY on line {i+1}:")
                print(f"  Raw line: {repr(line)}")
                print(f"  Stripped: {line.strip()}")
                if '=' in line:
                    key, value = line.split('=', 1)
                    print(f"  Key: '{key.strip()}'")
                    print(f"  Value: '{value.strip()}'")
except Exception as e:
    print(f"Error parsing .env file: {e}")
