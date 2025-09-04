"""
Azure AI Search Integration for Schema Vector Storage
Handles chunking, embedding, and similarity search of database schemas
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
import openai
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SchemaChunk:
    """Represents a chunk of schema information"""
    chunk_id: str
    table_name: str
    table_schema: str
    chunk_type: str  # 'table_info', 'columns', 'relationships', 'sample_data'
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class AzureSchemaVectorStore:
    """
    Azure AI Search integration for schema vector storage and retrieval
    """
    
    def __init__(self):
        # Azure AI Search configuration
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_KEY") 
        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "schema-index")
        
        # OpenAI configuration for embeddings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        
        # Validate configuration
        self._validate_config()
        
        # Initialize clients
        self.credential = AzureKeyCredential(self.search_key)
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.search_index_name,
            credential=self.credential
        )
        self.index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=self.credential
        )
        
        # Initialize OpenAI
        openai.api_key = self.openai_api_key
        
        print(f"✅ Azure AI Search initialized:")
        print(f"   Endpoint: {self.search_endpoint}")
        print(f"   Index: {self.search_index_name}")
        print(f"   Embedding Model: {self.embedding_model}")
    
    def _validate_config(self):
        """Validate required environment variables"""
        required_vars = [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_KEY", 
            "OPENAI_API_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    async def create_search_index(self):
        """Create the Azure AI Search index for schema storage"""
        
        print("🔧 Creating Azure AI Search index...")
        
        # Define search fields
        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="table_name", type=SearchFieldDataType.String),
            SearchableField(name="table_schema", type=SearchFieldDataType.String),
            SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="metadata", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=3072,  # text-embedding-3-large dimensions
                vector_search_profile_name="content-vector-profile"
            )
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="content-hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="content-vector-profile",
                    algorithm_configuration_name="content-hnsw"
                )
            ]
        )
        
        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="schema-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="table_name"),
                content_fields=[
                    SemanticField(field_name="content"),
                    SemanticField(field_name="metadata")
                ],
                keywords_fields=[SemanticField(field_name="chunk_type")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create index
        index = SearchIndex(
            name=self.search_index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        try:
            result = self.index_client.create_or_update_index(index)
            print(f"✅ Search index '{self.search_index_name}' created successfully")
            return result
        except Exception as e:
            print(f"❌ Failed to create search index: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text"""
        try:
            response = await openai.Embedding.acreate(
                model=self.embedding_model,
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            raise
    
    def chunk_schema_information(self, table_info: Dict[str, Any]) -> List[SchemaChunk]:
        """
        Chunk schema information into searchable pieces
        """
        chunks = []
        table_name = table_info.get("name", "unknown")
        table_schema = table_info.get("schema", "public")
        
        # Chunk 1: Table overview
        table_overview = f"""
        Table: {table_name}
        Schema: {table_schema}
        Type: {table_info.get('table_type', 'table')}
        Row Count: {table_info.get('row_count', 'unknown')}
        Description: {table_info.get('description', 'No description available')}
        Business Purpose: Database table containing structured data
        """
        
        chunks.append(SchemaChunk(
            chunk_id=f"{table_name}_overview",
            table_name=table_name,
            table_schema=table_schema,
            chunk_type="table_info",
            content=table_overview.strip(),
            metadata={
                "table_type": table_info.get('table_type', 'table'),
                "row_count": table_info.get('row_count'),
                "has_description": bool(table_info.get('description'))
            }
        ))
        
        # Chunk 2: Column information (grouped by related columns)
        columns = table_info.get("columns", {})
        if columns:
            # Group columns by type/purpose
            column_groups = {
                "id_keys": [],
                "dates_times": [],
                "names_descriptions": [],
                "numeric_metrics": [],
                "categorical": [],
                "other": []
            }
            
            for col_name, col_info in columns.items():
                col_type = col_info.get("data_type", "").lower()
                col_name_lower = col_name.lower()
                
                if any(keyword in col_name_lower for keyword in ["id", "key", "pk", "fk"]):
                    column_groups["id_keys"].append((col_name, col_info))
                elif any(keyword in col_type for keyword in ["date", "time", "timestamp"]):
                    column_groups["dates_times"].append((col_name, col_info))
                elif any(keyword in col_name_lower for keyword in ["name", "desc", "title", "label"]):
                    column_groups["names_descriptions"].append((col_name, col_info))
                elif any(keyword in col_type for keyword in ["int", "float", "decimal", "numeric"]):
                    column_groups["numeric_metrics"].append((col_name, col_info))
                elif any(keyword in col_type for keyword in ["varchar", "char", "text", "string"]):
                    column_groups["categorical"].append((col_name, col_info))
                else:
                    column_groups["other"].append((col_name, col_info))
            
            # Create chunks for each group
            for group_name, group_columns in column_groups.items():
                if group_columns:
                    column_content = f"Table: {table_name}\nColumn Group: {group_name}\n\n"
                    
                    for col_name, col_info in group_columns:
                        column_content += f"""
                        Column: {col_name}
                        Type: {col_info.get('data_type', 'unknown')}
                        Nullable: {col_info.get('nullable', True)}
                        Description: {col_info.get('description', 'No description')}
                        Sample Values: {col_info.get('sample_values', [])}
                        """
                    
                    chunks.append(SchemaChunk(
                        chunk_id=f"{table_name}_columns_{group_name}",
                        table_name=table_name,
                        table_schema=table_schema,
                        chunk_type="columns",
                        content=column_content.strip(),
                        metadata={
                            "column_group": group_name,
                            "column_count": len(group_columns),
                            "column_names": [col[0] for col in group_columns]
                        }
                    ))
        
        # Chunk 3: Business context and searchable terms
        business_context = f"""
        Table: {table_name}
        Business Context: {table_name.replace('_', ' ').replace('PHASE2', 'Phase 2').replace('SEP2024', 'September 2024')}
        Searchable Terms: {' '.join(table_name.split('_')).lower()}
        Domain: {'Healthcare' if any(term in table_name.upper() for term in ['NBA', 'HCP', 'PROVIDER']) else 'General'}
        Common Queries: data analysis, reporting, visualization, top records, filtering
        """
        
        chunks.append(SchemaChunk(
            chunk_id=f"{table_name}_business_context",
            table_name=table_name,
            table_schema=table_schema,
            chunk_type="business_context",
            content=business_context.strip(),
            metadata={
                "searchable_terms": table_name.split('_'),
                "domain": "healthcare" if "NBA" in table_name.upper() else "general"
            }
        ))
        
        return chunks
    
    async def index_database_schema(self, db_adapter):
        """
        Index all database schema information into Azure AI Search
        """
        print("🚀 Starting database schema indexing...")
        
        try:
            # Get all tables from database
            result = db_adapter.run("SHOW TABLES IN SCHEMA ENHANCED_NBA", dry_run=False)
            
            if result.error:
                raise Exception(f"Failed to get tables: {result.error}")
            
            all_tables = [row[1] if len(row) > 1 else str(row[0]) for row in result.rows]
            print(f"📊 Found {len(all_tables)} tables to index")
            
            # Process tables in batches
            batch_size = 10
            all_chunks = []
            
            for i in range(0, len(all_tables), batch_size):
                batch_tables = all_tables[i:i + batch_size]
                print(f"📝 Processing batch {i//batch_size + 1}: {len(batch_tables)} tables")
                
                for table_name in batch_tables:
                    try:
                        # Get table information
                        table_info = await self._get_table_info(db_adapter, table_name)
                        
                        # Chunk the schema information
                        table_chunks = self.chunk_schema_information(table_info)
                        
                        # Generate embeddings for each chunk
                        for chunk in table_chunks:
                            chunk.embedding = await self.generate_embedding(chunk.content)
                        
                        all_chunks.extend(table_chunks)
                        print(f"   ✅ Processed {table_name}: {len(table_chunks)} chunks")
                        
                    except Exception as e:
                        print(f"   ⚠️ Failed to process {table_name}: {e}")
                        continue
                
                # Index batch to Azure Search
                if all_chunks:
                    await self._upload_chunks_to_search(all_chunks[-len(batch_tables)*3:])  # Upload recent chunks
            
            print(f"🎉 Schema indexing complete: {len(all_chunks)} total chunks indexed")
            return len(all_chunks)
            
        except Exception as e:
            print(f"❌ Schema indexing failed: {e}")
            raise
    
    async def _get_table_info(self, db_adapter, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a table"""
        
        # Get column information
        columns_result = db_adapter.run(f"DESCRIBE TABLE {table_name}", dry_run=False)
        
        columns = {}
        if not columns_result.error:
            for row in columns_result.rows:
                col_name = row[0]
                col_type = row[1]
                nullable = row[2] == 'Y'
                
                columns[col_name] = {
                    "data_type": col_type,
                    "nullable": nullable,
                    "description": None,
                    "sample_values": []
                }
        
        # Get row count
        count_result = db_adapter.run(f"SELECT COUNT(*) FROM {table_name} LIMIT 1", dry_run=False)
        row_count = None
        if not count_result.error and count_result.rows:
            row_count = count_result.rows[0][0]
        
        return {
            "name": table_name,
            "schema": "ENHANCED_NBA",
            "table_type": "table",
            "columns": columns,
            "row_count": row_count,
            "description": f"Table containing {table_name.replace('_', ' ').lower()} data"
        }
    
    async def _upload_chunks_to_search(self, chunks: List[SchemaChunk]):
        """Upload schema chunks to Azure AI Search"""
        
        documents = []
        for chunk in chunks:
            doc = {
                "chunk_id": chunk.chunk_id,
                "table_name": chunk.table_name,
                "table_schema": chunk.table_schema,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "metadata": json.dumps(chunk.metadata),
                "content_vector": chunk.embedding
            }
            documents.append(doc)
        
        try:
            result = self.search_client.upload_documents(documents)
            successful = sum(1 for r in result if r.succeeded)
            print(f"   📤 Uploaded {successful}/{len(documents)} chunks to search index")
            return successful
        except Exception as e:
            print(f"   ❌ Failed to upload chunks: {e}")
            raise
    
    async def search_relevant_tables(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for relevant tables using vector similarity and hybrid search
        """
        print(f"🔍 Searching for tables relevant to: '{query}'")
        
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k * 2,  # Get more candidates for reranking
                fields="content_vector"
            )
            
            # Perform hybrid search (vector + text)
            search_results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k * 3,  # Get more results for diversity
                include_total_count=True
            )
            
            # Process and rank results
            table_scores = {}
            seen_tables = set()
            
            for result in search_results:
                table_name = result['table_name']
                score = result['@search.score']
                
                # Aggregate scores per table
                if table_name not in table_scores:
                    table_scores[table_name] = {
                        'table_name': table_name,
                        'total_score': 0,
                        'chunk_count': 0,
                        'best_chunk_score': 0,
                        'chunk_types': set(),
                        'sample_content': result['content'][:200] + "..."
                    }
                
                table_scores[table_name]['total_score'] += score
                table_scores[table_name]['chunk_count'] += 1
                table_scores[table_name]['best_chunk_score'] = max(
                    table_scores[table_name]['best_chunk_score'], 
                    score
                )
                table_scores[table_name]['chunk_types'].add(result['chunk_type'])
            
            # Sort tables by relevance score
            ranked_tables = sorted(
                table_scores.values(),
                key=lambda x: (x['best_chunk_score'], x['total_score']),
                reverse=True
            )
            
            # Return top K tables
            top_tables = ranked_tables[:top_k]
            
            print(f"✅ Found {len(top_tables)} relevant tables:")
            for i, table in enumerate(top_tables):
                print(f"   {i+1}. {table['table_name']} (score: {table['best_chunk_score']:.3f})")
            
            return top_tables
            
        except Exception as e:
            print(f"❌ Table search failed: {e}")
            return []
    
    async def get_table_details(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table from the search index"""
        
        try:
            # Search for all chunks related to this table
            search_results = self.search_client.search(
                search_text="",
                filter=f"table_name eq '{table_name}'",
                top=20
            )
            
            table_details = {
                'table_name': table_name,
                'chunks': {},
                'column_info': {},
                'metadata': {}
            }
            
            for result in search_results:
                chunk_type = result['chunk_type']
                table_details['chunks'][chunk_type] = {
                    'content': result['content'],
                    'metadata': json.loads(result['metadata'])
                }
            
            return table_details
            
        except Exception as e:
            print(f"❌ Failed to get table details: {e}")
            return {}

# Usage example and test functions
if __name__ == "__main__":
    async def test_azure_search():
        """Test Azure AI Search integration"""
        
        print("🧪 Testing Azure AI Search Schema Integration")
        print("=" * 60)
        
        try:
            # Initialize vector store
            vector_store = AzureSchemaVectorStore()
            
            # Create search index
            await vector_store.create_search_index()
            
            # Test with sample query
            query = "read table final nba output python and fetch top 5 rows"
            results = await vector_store.search_relevant_tables(query, top_k=4)
            
            print(f"\n🎯 Query: '{query}'")
            print(f"📊 Found {len(results)} relevant tables")
            
            return results
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    # asyncio.run(test_azure_search())
