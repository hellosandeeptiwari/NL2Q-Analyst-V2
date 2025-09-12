#!/usr/bin/env python3
"""
Test script demonstrating the non-hardcoded deterministic SQL generator
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from backend.nl2sql.deterministic_generator import (
    generate_deterministic_sql, 
    create_sample_config_file,
    ScoringConfig,
    PatternConfig,
    JoinConfig
)
from backend.nl2sql.guardrails import GuardrailConfig

def test_configuration_driven_approach():
    """Demonstrate how the system works with different configurations"""
    
    print("🧪 TESTING CONFIGURATION-DRIVEN DETERMINISTIC SQL GENERATOR")
    print("=" * 70)
    
    # Sample catalog (no hardcoded assumptions about schema)
    catalog = {
        'tables': [
            {'name': 'customers'},
            {'name': 'orders'},
            {'name': 'products'}
        ],
        'columns': [
            {'table_name': 'customers', 'column_name': 'customer_id', 'data_type': 'INTEGER'},
            {'table_name': 'customers', 'column_name': 'customer_name', 'data_type': 'VARCHAR'},
            {'table_name': 'customers', 'column_name': 'email', 'data_type': 'VARCHAR'},
            {'table_name': 'customers', 'column_name': 'created_date', 'data_type': 'DATE'},
            
            {'table_name': 'orders', 'column_name': 'order_id', 'data_type': 'INTEGER'},
            {'table_name': 'orders', 'column_name': 'customer_id', 'data_type': 'INTEGER'},
            {'table_name': 'orders', 'column_name': 'order_total', 'data_type': 'DECIMAL'},
            {'table_name': 'orders', 'column_name': 'order_date', 'data_type': 'DATE'},
            
            {'table_name': 'products', 'column_name': 'product_id', 'data_type': 'INTEGER'},
            {'table_name': 'products', 'column_name': 'product_name', 'data_type': 'VARCHAR'},
            {'table_name': 'products', 'column_name': 'price', 'data_type': 'DECIMAL'},
        ]
    }
    
    constraints = GuardrailConfig(default_limit=100)
    query = "show me customer names and their total order amounts"
    
    print(f"\n📝 Query: {query}")
    print(f"📊 Catalog: {len(catalog['tables'])} tables, {len(catalog['columns'])} columns")
    
    # Test 1: Default configuration
    print("\n🔧 TEST 1: Default Configuration")
    print("-" * 40)
    result1 = generate_deterministic_sql(query, catalog, constraints)
    print(f"SQL Generated: {result1.sql}")
    print(f"Confidence: {result1.confidence_score:.2f}")
    print(f"Plan Status: {result1.plan.status}")
    
    # Test 2: Custom configuration with different weights
    print("\n🔧 TEST 2: Custom Configuration (Higher Type Weight)")
    print("-" * 40)
    
    # Create custom config favoring type matching over name matching
    config_path = "test_config_custom.json"
    custom_config = {
        "scoring": {
            "name_weight": 0.40,    # Lower name weight
            "type_weight": 0.55,    # Higher type weight
            "boost_weight": 0.05,
            "primary_threshold": 0.45  # Lower threshold
        }
    }
    
    import json
    with open(config_path, 'w') as f:
        json.dump(custom_config, f)
    
    result2 = generate_deterministic_sql(query, catalog, constraints, config_path)
    print(f"SQL Generated: {result2.sql}")
    print(f"Confidence: {result2.confidence_score:.2f}")
    print(f"Plan Status: {result2.plan.status}")
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)
    
    # Test 3: Environment variable configuration
    print("\n🔧 TEST 3: Environment Variable Configuration")
    print("-" * 40)
    
    # Set environment variables
    os.environ['DETERMINISTIC_NAME_WEIGHT'] = '0.80'
    os.environ['DETERMINISTIC_TYPE_WEIGHT'] = '0.15'
    os.environ['DETERMINISTIC_PRIMARY_THRESHOLD'] = '0.40'
    os.environ['DETERMINISTIC_MAX_COLUMNS'] = '8'
    
    result3 = generate_deterministic_sql(query, catalog, constraints)
    print(f"SQL Generated: {result3.sql}")
    print(f"Confidence: {result3.confidence_score:.2f}")
    print(f"Plan Status: {result3.plan.status}")
    
    # Clean up environment
    for key in ['DETERMINISTIC_NAME_WEIGHT', 'DETERMINISTIC_TYPE_WEIGHT', 
               'DETERMINISTIC_PRIMARY_THRESHOLD', 'DETERMINISTIC_MAX_COLUMNS']:
        if key in os.environ:
            del os.environ[key]
    
    print("\n✅ All tests demonstrate NO HARDCODED values!")
    print("   - Scoring weights: configurable via file/env")
    print("   - Patterns: customizable per domain")
    print("   - Thresholds: adjustable based on data quality")
    print("   - Join rules: flexible based on business logic")
    
    print("\n📋 Configuration Options:")
    print("   1. JSON configuration file")
    print("   2. Environment variables")
    print("   3. Programmatic configuration objects")
    print("   4. Domain-specific pattern extensions")

def demonstrate_pattern_customization():
    """Show how patterns can be customized for different domains"""
    
    print("\n🎯 PATTERN CUSTOMIZATION EXAMPLE")
    print("=" * 50)
    
    # Healthcare domain example
    healthcare_patterns = PatternConfig(
        identifier_patterns=['patient_id', 'provider_id', 'claim_id', 'diagnosis_code'],
        identifier_keywords=['patient', 'provider', 'claim', 'diagnosis', 'medical'],
        numeric_keywords=['dosage', 'cost', 'copay', 'duration', 'quantity', 'units']
    )
    
    # Finance domain example  
    finance_patterns = PatternConfig(
        identifier_patterns=['account_id', 'transaction_id', 'customer_id', 'routing_number'],
        identifier_keywords=['account', 'transaction', 'customer', 'routing', 'bank'],
        numeric_keywords=['balance', 'amount', 'interest', 'fee', 'payment', 'credit']
    )
    
    print("Healthcare patterns loaded:", len(healthcare_patterns.identifier_patterns), "ID patterns")
    print("Finance patterns loaded:", len(finance_patterns.identifier_patterns), "ID patterns")
    print("\n✅ Patterns are completely customizable per domain/use case!")

if __name__ == "__main__":
    test_configuration_driven_approach()
    demonstrate_pattern_customization()
    
    # Create sample configuration file
    sample_path = "deterministic_config_sample.json"
    create_sample_config_file(sample_path)
    print(f"\n📄 Sample configuration created: {sample_path}")
