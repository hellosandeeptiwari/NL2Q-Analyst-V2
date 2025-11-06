"""
Test comprehensive filter pattern extraction
"""
from backend.tools.filter_value_resolver import FilterValueResolver

def test_patterns():
    resolver = FilterValueResolver(None)
    
    test_queries = [
        # Quarter patterns
        "What was the total TRx for Tirosint Capsules in Q2 for 2022?",
        "Show me Q2 of 2022 data",
        "Get Q2 2022 results",
        "Data during Q2 in 2022",
        "Second quarter of 2022",
        "2nd quarter 2022",
        
        # Month patterns
        "Show January 2022 data",
        "In January of 2022",
        "During Jan 2022",
        
        # Week patterns
        "Week 12 of 2022",
        "W12 2022 data",
        
        # Year patterns
        "In 2022",
        "FY2022 results",
        "YTD 2022",
        
        # Range patterns
        "Between Q1 2022 and Q3 2022",
        "From January to March",
        "Q1-Q3 2022",
        
        # Product patterns
        "For Tirosint Capsules",
        "About Tirosint product",
        "Related to Tirosint brand",
        
        # Location patterns
        "In California",
        "Territory T123",
        "Region North",
        
        # Prescriber patterns
        "Prescriber ID 12345",
        "Account ABC123",
        "NPI 1234567890",
        
        # Comparison patterns
        "Greater than 1000",
        "At least 500",
        "Less than 100",
        
        # Pharma terms
        "Target prescribers",
        "PDRP enabled",
        "New growth decliner",
    ]
    
    print("=" * 80)
    print("TESTING COMPREHENSIVE FILTER PATTERNS")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        filters = resolver._extract_filter_hints(query)
        print(f"   Extracted {len(filters)} filters:")
        for f in filters:
            hint = f.get('hint', 'N/A')
            value = f.get('value')
            operator = f.get('operator', '=')
            print(f"      • [{hint}] {operator} '{value}'")

if __name__ == "__main__":
    test_patterns()
