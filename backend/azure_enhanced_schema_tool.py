"""
Enhanced Schema Discovery with Azure AI Search Integration
Replaces vector matching with Azure-powered similarity search
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from backend.tools.schema_tool import SchemaTool, SchemaContext, TableInfo
from backend.azure_schema_vector_store import AzureSchemaVectorStore

class AzureEnhancedSchemaTool(SchemaTool):
    """
    Enhanced schema discovery using Azure AI Search for better table matching
    """
    
    def __init__(self):
        super().__init__()
        self.azure_vector_store = AzureSchemaVectorStore()
        print("✅ Azure-Enhanced Schema Discovery initialized")
    
    async def discover_schema(self, query: str, context: Dict[str, Any] = None) -> SchemaContext:
        """
        Enhanced schema discovery with Azure AI Search
        Returns top 4 relevant tables as suggestions
        """
        print(f"🔍 Azure-Enhanced Schema Discovery for: '{query}'")
        
        try:
            # Use Azure AI Search to find relevant tables
            table_matches = await self.azure_vector_store.search_relevant_tables(
                query, 
                top_k=4  # Always get top 4 for user selection
            )
            
            print(f"📊 Found {len(table_matches)} relevant tables from Azure Search")
            
            # Convert Azure search results to TableInfo objects
            relevant_tables = []
            
            for match in table_matches:
                table_name = match['table_name']
                
                # Get detailed table information
                table_details = await self.azure_vector_store.get_table_details(table_name)
                
                # Extract column information from indexed data
                columns = []
                if 'columns' in table_details.get('chunks', {}):
                    column_chunk = table_details['chunks']['columns']
                    column_names = column_chunk.get('metadata', {}).get('column_names', [])
                    
                    for col_name in column_names:
                        columns.append({
                            "name": col_name,
                            "data_type": "unknown",  # Will be filled by database query if needed
                            "nullable": True,
                            "description": None
                        })
                
                # Create TableInfo object
                table_info = TableInfo(
                    name=table_name,
                    schema="ENHANCED_NBA",
                    type="table",
                    columns=columns,
                    row_count=None,
                    description=f"Table containing {table_name.replace('_', ' ').lower()} data"
                )
                
                relevant_tables.append(table_info)
            
            # Create entity mappings based on query analysis
            entities = self._extract_entities(query)
            entity_mappings = self._create_entity_mappings(entities, relevant_tables)
            
            # Create business glossary
            business_glossary = self._create_business_glossary(relevant_tables)
            
            schema_context = SchemaContext(
                relevant_tables=relevant_tables,
                entity_mappings=entity_mappings,
                join_paths=[],  # Will be populated if needed
                metrics_available=[],
                date_columns=[],
                filter_suggestions=[],
                business_glossary=business_glossary
            )
            
            print(f"✅ Schema discovery complete: {len(relevant_tables)} tables, {len(entity_mappings)} entities")
            
            return schema_context
            
        except Exception as e:
            print(f"❌ Azure schema discovery failed: {e}")
            # Fallback to original schema discovery
            print("🔄 Falling back to original schema discovery...")
            return await super().discover_schema(query, context)
    
    def _create_entity_mappings(self, entities: List[str], tables: List[TableInfo]) -> Dict[str, List[str]]:
        """Create entity to column mappings"""
        
        entity_mappings = {}
        
        for entity in entities:
            entity_lower = entity.lower()
            matching_columns = []
            
            for table in tables:
                for column in table.columns:
                    col_name_lower = column["name"].lower()
                    
                    # Check if entity matches column name or contains entity
                    if (entity_lower in col_name_lower or 
                        col_name_lower in entity_lower or
                        any(word in col_name_lower for word in entity_lower.split('_'))):
                        matching_columns.append(f"{table.name}.{column['name']}")
            
            if matching_columns:
                entity_mappings[entity] = matching_columns
        
        return entity_mappings
    
    def _create_business_glossary(self, tables: List[TableInfo]) -> Dict[str, str]:
        """Create business glossary from table information"""
        
        glossary = {}
        
        for table in tables:
            table_name = table.name
            
            # Add table name variations
            glossary[table_name] = f"Database table: {table_name}"
            
            # Add searchable variations
            name_parts = table_name.split('_')
            for part in name_parts:
                if len(part) > 2:  # Skip short parts
                    glossary[part.lower()] = f"Related to table: {table_name}"
            
            # Add business context
            if 'NBA' in table_name:
                glossary['nba'] = "Next Best Action - healthcare recommendation system"
                glossary['basketball'] = "This refers to NBA (Next Best Action), not basketball"
            
            if 'FINAL' in table_name:
                glossary['final'] = "Final processed output data"
                glossary['output'] = "System output or results"
            
            if 'PYTHON' in table_name:
                glossary['python'] = "Data processed or formatted for Python analysis"
        
        return glossary

    async def get_table_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """
        Get table suggestions with metadata for user selection
        """
        print(f"💡 Getting table suggestions for: '{query}'")
        
        try:
            # Get top 4 table matches
            table_matches = await self.azure_vector_store.search_relevant_tables(query, top_k=4)
            
            suggestions = []
            
            for i, match in enumerate(table_matches):
                table_name = match['table_name']
                
                # Get additional table details if needed
                table_details = await self.azure_vector_store.get_table_details(table_name)
                
                suggestion = {
                    "rank": i + 1,
                    "table_name": table_name,
                    "relevance_score": match['best_chunk_score'],
                    "description": f"Table containing {table_name.replace('_', ' ').lower()} data",
                    "sample_content": match['sample_content'],
                    "chunk_types": list(match['chunk_types']),
                    "estimated_relevance": "High" if match['best_chunk_score'] > 0.8 else "Medium" if match['best_chunk_score'] > 0.6 else "Low"
                }
                
                suggestions.append(suggestion)
            
            print(f"✅ Generated {len(suggestions)} table suggestions")
            return suggestions
            
        except Exception as e:
            print(f"❌ Failed to get table suggestions: {e}")
            return []

# Integration with Dynamic Orchestrator
async def get_azure_enhanced_schema_discovery():
    """Factory function to get Azure-enhanced schema discovery"""
    return AzureEnhancedSchemaTool()

if __name__ == "__main__":
    async def test_azure_schema_discovery():
        """Test Azure-enhanced schema discovery"""
        
        print("🧪 Testing Azure-Enhanced Schema Discovery")
        print("=" * 50)
        
        try:
            # Create enhanced schema tool
            schema_tool = AzureEnhancedSchemaTool()
            
            # Test query
            query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and provider input"
            
            print(f"🔍 Test Query: {query}")
            
            # Get schema discovery
            schema_context = await schema_tool.discover_schema(query)
            
            print(f"\n📊 DISCOVERY RESULTS:")
            print(f"   Tables found: {len(schema_context.relevant_tables)}")
            print(f"   Entity mappings: {len(schema_context.entity_mappings)}")
            
            print(f"\n📋 DISCOVERED TABLES:")
            for i, table in enumerate(schema_context.relevant_tables):
                print(f"   {i+1}. {table.name}")
                print(f"      Schema: {table.schema}")
                print(f"      Columns: {len(table.columns)}")
                print(f"      Description: {table.description}")
            
            # Get table suggestions
            print(f"\n💡 TABLE SUGGESTIONS:")
            suggestions = await schema_tool.get_table_suggestions(query)
            
            for suggestion in suggestions:
                print(f"\n   {suggestion['rank']}. {suggestion['table_name']}")
                print(f"      Relevance: {suggestion['estimated_relevance']} ({suggestion['relevance_score']:.3f})")
                print(f"      Description: {suggestion['description']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run test
    # asyncio.run(test_azure_schema_discovery())
