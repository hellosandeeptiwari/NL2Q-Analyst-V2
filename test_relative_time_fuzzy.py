"""
Test fuzzy matching for relative time periods
"""
from difflib import SequenceMatcher

# Simulate what's in the database
db_values = ['Q3 2025', 'Q2 2025', 'Q1 2025', 'C4 WK', 'C13 WK', 'C WK', 'C QTD']

# What user might say
user_phrases = [
    'current week',
    'Current Week',
    'this week',
    'current quarter',
    'last 4 weeks',
    'last 13 weeks',
    'quarter to date',
]

print("=" * 80)
print("FUZZY MATCHING TEST: Relative Time Periods")
print("=" * 80)

for user_value in user_phrases:
    print(f"\n📝 User phrase: '{user_value}'")
    
    # Find best match
    best_match = None
    best_score = 0
    
    for db_value in db_values:
        # Normalize both for comparison
        normalized_user = user_value.lower().strip()
        normalized_db = db_value.lower().strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, normalized_user, normalized_db).ratio()
        
        if similarity > best_score:
            best_score = similarity
            best_match = db_value
    
    print(f"   Best match: '{best_match}' (score: {best_score:.2f})")
    
    if best_score >= 0.85:
        print(f"   ✅ PASS: Score >= 0.85 threshold")
    elif best_score >= 0.60:
        print(f"   ⚠️  MEDIUM: Score {best_score:.2f} (might work with lower threshold)")
    else:
        print(f"   ❌ FAIL: Score too low, fuzzy match won't work")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("For relative time periods, fuzzy matching may not work well because:")
print("  'current week' vs 'C WK' = very different strings")
print("  ")
print("Better approach: Let LLM see the actual database values in the prompt")
print("and let it choose the correct TimePeriod value based on context.")
