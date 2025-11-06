"""
Quick test for quarter patterns
"""
import re

quarter_test_queries = [
    "Q2 2022",
    "Q2 for 2022",
    "Q2 of 2022",
    "Q2 in 2022",
    "in Q2 2022",
    "during Q2 2022",
    "second quarter 2022",
    "second quarter of 2022",
    "second quarter for 2022",
    "second quarter in 2022",
    "the second quarter 2022",
    "in second quarter 2022",
    "during the second quarter of 2022",
    "2nd quarter 2022",
    "2nd quarter of 2022",
    "2Q 2022",
    "Q2 FY2022",
    "2nd Qtr 2022",
    "2022-Q2",
    "2022 Q2",
]

# Test pattern
quarter_patterns = [
    r'\b(Q[1-4])\s+(?:of|for|in|during)?\s*(\d{4})\b',
    r'\b(\d{4})[-/\s](Q[1-4])\b',
    r'\b(?:in|during|for)\s+(Q[1-4])\s+(?:of|for|in)?\s*(\d{4})\b',
    r'\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of|for|in)?\s*(\d{4})\b',
    r'\b(?:in|during|for)\s+(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of|for|in)?\s*(\d{4})\b',
    r'\b([1-4])Q\s+(\d{4})\b',
    r'\b(Q[1-4])\s+FY\s*(\d{4})\b',
    r'\b([1-4])(?:st|nd|rd|th)\s+(?:qtr|quarter)\s+(?:of|for|in)?\s*(\d{4})\b',
]

additional_quarter_pattern = r'\b(?:the|a|an)?\s*(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of|for|in)?\s*(\d{4})\b'

print("=" * 80)
print("QUARTER PATTERN TESTING")
print("=" * 80)

for query in quarter_test_queries:
    print(f"\n📝 Testing: '{query}'")
    matched = False
    
    for pattern in quarter_patterns:
        matches = list(re.finditer(pattern, query, re.IGNORECASE))
        if matches:
            print(f"   ✅ Matched by pattern: {pattern[:50]}...")
            for match in matches:
                print(f"      Groups: {match.groups()}")
            matched = True
            break
    
    if not matched:
        matches = list(re.finditer(additional_quarter_pattern, query, re.IGNORECASE))
        if matches:
            print(f"   ✅ Matched by additional pattern")
            for match in matches:
                print(f"      Groups: {match.groups()}")
            matched = True
    
    if not matched:
        print(f"   ❌ NOT MATCHED!")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
