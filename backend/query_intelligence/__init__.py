"""
Query Intelligence Module

This module provides intelligent query planning and schema analysis capabilities
for enhanced natural language to SQL conversion.

Components (Cleaned up - duplicates removed):
- IntelligentQueryPlanner: Advanced query planning with semantic understanding (73KB, 1605 lines)
- SchemaSemanticAnalyzer: Deep schema analysis for business context (33KB, 802 lines)
"""

# Export the comprehensive implementations (duplicates removed)
try:
    from .intelligent_query_planner import IntelligentQueryPlanner
    from .schema_analyzer import SchemaSemanticAnalyzer
    
    __all__ = [
        'IntelligentQueryPlanner',
        'SchemaSemanticAnalyzer'
    ]
    
    print("Query Intelligence module loaded with comprehensive implementations (duplicates cleaned up)")
    
except ImportError as e:
    print(f"⚠️ Query intelligence components failed to import: {e}")
    __all__ = []