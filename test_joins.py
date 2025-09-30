#!/usr/bin/env python3
"""
Test join relationship discovery
"""

from backend.query_intelligence.intelligent_query_planner import IntelligentQueryPlanner
from backend.db.engine import get_adapter
import os

os.environ['DB_ENGINE'] = 'azure_sql'
db_adapter = get_adapter()

context = {
    'matched_tables': [
        {'table_name': 'Reporting_BI_PrescriberProfile', 'columns': []},
        {'table_name': 'Reporting_BI_PrescriberOverview', 'columns': []}
    ],
    'db_adapter': db_adapter
}

planner = IntelligentQueryPlanner()
metadata = planner._extract_table_metadata(context, ['Reporting_BI_PrescriberProfile', 'Reporting_BI_PrescriberOverview'])

print('=== JOIN RELATIONSHIP TEST ===')
print('Tables processed:', list(metadata.keys()))

for table_name, table_data in metadata.items():
    relationships = table_data.get('relationships', [])
    print(f'\n{table_name}:')
    print(f'  - Relationships found: {len(relationships)}')
    
    for i, rel in enumerate(relationships):
        print(f'  - Join {i+1}: {rel["column1"]} = {rel["column2"]} (confidence: {rel["confidence"]})')
        print(f'    Type: {rel["relationship_type"]}, Source: {rel["source"]}')

# Also test the schema context building
print('\n=== SCHEMA CONTEXT TEST ===')
query_semantics = {'entities': ['prescriber'], 'relationships': ['grouping']}
schema_context = planner._build_schema_context(metadata, query_semantics)

print('Schema context keys:', list(schema_context.keys()))
print('Join paths found:', len(schema_context.get('join_paths', [])))

for i, join_path in enumerate(schema_context.get('join_paths', [])):
    print(f'Join path {i+1}: {join_path["table1"]} → {join_path["table2"]}')
    print(f'  Columns: {join_path["join_columns"]}, Confidence: {join_path["confidence"]}')