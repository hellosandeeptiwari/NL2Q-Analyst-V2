"""
Test improved schema formatting to prevent column mapping errors
"""

import os
import sys
sys.path.append('C:\\Users\\SandeepT\\NL2Q Analyst\\NL2Q-Analyst-V2')

from backend.nl2sql.enhanced_generator import format_catalog_for_prompt

# Test with the same schema structure
test_schema = {
    "Reporting_BI_PrescriberProfile": [
        "RegionId", "RegionName", "TerritoryId", "TerritoryName", 
        "PrescriberId", "PrescriberName", "TRX", "ProductGroupName"
    ],
    "Reporting_BI_PrescriberOverview": [
        "RegionId", "RegionName", "TerritoryId", "TerritoryName",
        "PrescriberId", "PrescriberName", "PrimaryProduct", "SecondaryProduct",
        "TRX(C4 Wk)", "TRX(C13 Wk)"
    ],
    "Reporting_BI_NGD": [
        "TerritoryId", "TerritoryName", "PrescriberId", "Product", "NGDType"
    ]
}

print("🧪 Testing Improved Schema Formatting")
print("=" * 50)
catalog = format_catalog_for_prompt(test_schema)
print(catalog)
print("\n" + "=" * 50)
print("✅ Schema formatting complete - this should be much clearer for OpenAI!")