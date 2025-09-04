"""
Pinecone Integration for Schema Vector Storage
Handles chunking, embedding, and similarity search of database schemas
"""
import os
import json
from pinecone import Pinecone, ServerlessSpec
import openai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SchemaChunk:
    chunk_id: str
    table_name: str
    table_schema: str
    chunk_type: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class PineconeSchemaVectorStore:
    def clear_index(self):
        """Delete all vectors from the Pinecone index"""
        self.index.delete(deleteAll=True)
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "nl2q-schema-index")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("EMBED_MODEL", "text-embedding-3-large")
        
        # Initialize Pinecone with v5 API
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # text-embedding-3-large dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"✅ Created new Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"⚠️ Index creation info: {e}")
        
        self.index = self.pc.Index(self.index_name)
        
        # Initialize OpenAI client with v1.0+ API
        from openai import AsyncOpenAI
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        
        print(f"✅ Pinecone initialized: {self.index_name}")

    async def generate_embedding(self, text: str) -> List[float]:
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def chunk_schema_information(self, table_info: Dict[str, Any]) -> List[SchemaChunk]:
        """Create optimized schema chunks for better vector search"""
        chunks = []
        table_name = table_info.get("name", "unknown")
        table_schema = table_info.get("schema", "public")
        
        # 1. Comprehensive table overview chunk - includes all context
        table_description = table_info.get("description", f"Table containing {table_name.replace('_', ' ').lower()} data")
        columns = table_info.get("columns", {})
        column_names = list(columns.keys()) if columns else []
        column_types = [f"{col}: {info.get('data_type', 'unknown')}" for col, info in columns.items()] if columns else []
        
        # Create rich table context
        table_chunk_content = f"""Table: {table_name}
Schema: {table_schema}
Description: {table_description}
Total Columns: {len(column_names)}
Column Names: {', '.join(column_names[:20])}
Column Types: {', '.join(column_types[:15])}
Business Domain: {self._extract_business_domain(table_name)}
Data Category: {self._categorize_table(table_name, column_names)}
Table Purpose: {self._infer_table_purpose(table_name, column_names)}
Row Count: {table_info.get('row_count', 'unknown')}"""
        
        chunks.append(SchemaChunk(
            chunk_id=f"{table_name}_overview",
            table_name=table_name,
            table_schema=table_schema,
            chunk_type="table_overview",
            content=table_chunk_content,
            metadata={
                "total_columns": len(column_names),
                "business_domain": self._extract_business_domain(table_name),
                "data_category": self._categorize_table(table_name, column_names),
                "row_count": table_info.get('row_count')
            }
        ))
        
        # 2. Column group chunks - group related columns together
        if columns:
            column_groups = self._group_columns_by_purpose(columns)
            for group_name, group_columns in column_groups.items():
                if not group_columns:
                    continue
                    
                group_content = f"""Table: {table_name} - {group_name} Columns
{', '.join([f"{col} ({info.get('data_type', 'unknown')})" for col, info in group_columns.items()])}
Column Group: {group_name}
Table Context: {table_description}
Business Purpose: {self._infer_table_purpose(table_name, column_names)}"""
                
                chunks.append(SchemaChunk(
                    chunk_id=f"{table_name}_columns_{group_name.lower().replace(' ', '_')}",
                    table_name=table_name,
                    table_schema=table_schema,
                    chunk_type="column_group",
                    content=group_content,
                    metadata={
                        "column_group": group_name,
                        "column_count": len(group_columns),
                        "columns": list(group_columns.keys())
                    }
                ))
        
        # 3. Business context chunk - semantic meaning and use cases
        business_chunk_content = f"""Business Context for {table_name}
Business Domain: {self._extract_business_domain(table_name)}
Likely Use Cases: {self._generate_use_cases(table_name, column_names)}
Data Category: {self._categorize_table(table_name, column_names)}
Related Concepts: {self._extract_business_concepts(table_name, column_names)}
Table Purpose: {self._infer_table_purpose(table_name, column_names)}
Data Analytics Focus: {self._get_analytics_focus(table_name)}"""
        
        chunks.append(SchemaChunk(
            chunk_id=f"{table_name}_business_context",
            table_name=table_name,
            table_schema=table_schema,
            chunk_type="business_context",
            content=business_chunk_content,
            metadata={
                "business_domain": self._extract_business_domain(table_name),
                "use_cases": self._generate_use_cases(table_name, column_names),
                "concepts": self._extract_business_concepts(table_name, column_names)
            }
        ))
        
        return chunks
        
    def _extract_business_domain(self, table_name: str) -> str:
        """Extract business domain from table name"""
        name_lower = table_name.lower()
        if any(word in name_lower for word in ['nba', 'basketball', 'sports', 'game', 'player', 'team']):
            return "Sports Analytics"
        elif any(word in name_lower for word in ['sales', 'revenue', 'commercial', 'market']):
            return "Sales & Marketing"
        elif any(word in name_lower for word in ['prediction', 'forecast', 'model', 'ml', 'ai']):
            return "Predictive Analytics"
        elif any(word in name_lower for word in ['feature', 'attribute', 'metric']):
            return "Feature Engineering"
        elif any(word in name_lower for word in ['flow', 'process', 'pipeline']):
            return "Data Pipeline"
        else:
            return "General Business"
            
    def _categorize_table(self, table_name: str, columns: List[str]) -> str:
        """Categorize table based on name and columns"""
        name_lower = table_name.lower()
        col_names = [col.lower() for col in columns]
        
        if any(word in name_lower for word in ['prediction', 'forecast', 'model']):
            return "Predictive Model"
        elif any(word in name_lower for word in ['sales', 'revenue']):
            return "Sales Data"
        elif any(word in name_lower for word in ['feature', 'attribute']):
            return "Feature Table"
        elif any(col for col in col_names if 'date' in col or 'time' in col):
            return "Time Series"
        elif any(col for col in col_names if 'id' in col):
            return "Transactional"
        else:
            return "Analytical"
            
    def _group_columns_by_purpose(self, columns: Dict) -> Dict[str, Dict]:
        """Group columns by their likely purpose"""
        groups = {
            "Identifiers": {},
            "Metrics": {},
            "Dates": {},
            "Categories": {},
            "Descriptions": {}
        }
        
        for col_name, col_info in columns.items():
            col_lower = col_name.lower()
            data_type = col_info.get('data_type', '').lower()
            
            if any(word in col_lower for word in ['id', 'key', 'code']):
                groups["Identifiers"][col_name] = col_info
            elif any(word in col_lower for word in ['date', 'time', 'timestamp']):
                groups["Dates"][col_name] = col_info
            elif 'number' in data_type or 'int' in data_type or 'float' in data_type or 'decimal' in data_type:
                groups["Metrics"][col_name] = col_info
            elif any(word in col_lower for word in ['name', 'description', 'text', 'comment']):
                groups["Descriptions"][col_name] = col_info
            else:
                groups["Categories"][col_name] = col_info
                
        return groups
        
    def _infer_table_purpose(self, table_name: str, columns: List[str]) -> str:
        """Infer the main purpose of the table"""
        name_lower = table_name.lower()
        
        if 'prediction' in name_lower:
            return "Store machine learning predictions and forecasts"
        elif 'sales' in name_lower:
            return "Track sales performance and revenue metrics"
        elif 'feature' in name_lower:
            return "Store engineered features for analytics"
        elif 'global' in name_lower:
            return "Provide comprehensive aggregated data"
        elif 'flow' in name_lower:
            return "Track data processing flows and pipelines"
        else:
            return "Support business analytics and reporting"
            
    def _generate_use_cases(self, table_name: str, columns: List[str]) -> str:
        """Generate likely use cases for the table"""
        name_lower = table_name.lower()
        use_cases = []
        
        if 'nba' in name_lower:
            use_cases.append("NBA sports analytics")
        if 'sales' in name_lower:
            use_cases.append("Sales performance analysis")
        if 'prediction' in name_lower:
            use_cases.append("Predictive modeling")
        if 'feature' in name_lower:
            use_cases.append("Machine learning feature engineering")
        if any(col for col in columns if 'date' in col.lower()):
            use_cases.append("Time series analysis")
            
        return ", ".join(use_cases) if use_cases else "General business analytics"
        
    def _extract_business_concepts(self, table_name: str, columns: List[str]) -> str:
        """Extract business concepts from table and columns"""
        concepts = set()
        
        # From table name
        name_parts = table_name.lower().replace('_', ' ').split()
        concepts.update(name_parts)
        
        # From column names
        for col in columns[:10]:  # First 10 columns
            col_parts = col.lower().replace('_', ' ').split()
            concepts.update(col_parts)
            
        # Filter and return meaningful concepts
        meaningful_concepts = [c for c in concepts if len(c) > 2 and c not in ['the', 'and', 'for', 'with']]
        return ", ".join(sorted(meaningful_concepts)[:15])  # Top 15 concepts
        
    def _get_analytics_focus(self, table_name: str) -> str:
        """Determine the analytics focus of the table"""
        name_lower = table_name.lower()
        
        if 'nba' in name_lower:
            return "Sports performance and business intelligence"
        elif any(word in name_lower for word in ['sales', 'revenue']):
            return "Revenue optimization and market analysis"
        elif 'prediction' in name_lower:
            return "Forecasting and predictive modeling"
        else:
            return "Business intelligence and data analysis"

    async def index_database_schema(self, db_adapter):
        print("🚀 Indexing database schema in Pinecone...")
        result = db_adapter.run("SHOW TABLES IN SCHEMA ENHANCED_NBA", dry_run=False)
        if result.error:
            raise Exception(f"Failed to get tables: {result.error}")
        all_tables = [row[1] if len(row) > 1 else str(row[0]) for row in result.rows]
        print(f"📊 Found {len(all_tables)} tables to index")
        for table_name in all_tables:
            try:
                table_info = await self._get_table_info(db_adapter, table_name)
                table_chunks = self.chunk_schema_information(table_info)
                for chunk in table_chunks:
                    chunk.embedding = await self.generate_embedding(chunk.content)
                    # Upsert to Pinecone
                    self.index.upsert([(chunk.chunk_id, chunk.embedding, {"table_name": chunk.table_name, "chunk_type": chunk.chunk_type, "metadata": json.dumps(chunk.metadata)})])
                print(f"   ✅ Indexed {table_name}: {len(table_chunks)} chunks")
            except Exception as e:
                print(f"   ⚠️ Failed to process {table_name}: {e}")
        print(f"🎉 Pinecone schema indexing complete!")

    async def _get_table_info(self, db_adapter, table_name: str) -> Dict[str, Any]:
        columns_result = db_adapter.run(f"DESCRIBE TABLE {table_name}", dry_run=False)
        columns = {}
        if not columns_result.error:
            for row in columns_result.rows:
                col_name = row[0]
                col_type = row[1]
                nullable = row[2] == 'Y'
                columns[col_name] = {"data_type": col_type, "nullable": nullable, "description": None}
        count_result = db_adapter.run(f"SELECT COUNT(*) FROM {table_name} LIMIT 1", dry_run=False)
        row_count = None
        if not count_result.error and count_result.rows:
            row_count = count_result.rows[0][0]
        return {"name": table_name, "schema": "ENHANCED_NBA", "table_type": "table", "columns": columns, "row_count": row_count, "description": f"Table containing {table_name.replace('_', ' ').lower()} data"}

    async def search_relevant_tables(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        print(f"🔍 Searching Pinecone for relevant tables: '{query}'")
        query_embedding = await self.generate_embedding(query)
        results = self.index.query(vector=query_embedding, top_k=top_k*3, include_metadata=True)
        table_scores = {}
        for match in results.matches:
            table_name = match.metadata.get("table_name")
            score = match.score
            if table_name not in table_scores:
                table_scores[table_name] = {"table_name": table_name, "total_score": 0, "best_score": 0, "chunk_types": set(), "sample_content": ""}
            table_scores[table_name]["total_score"] += score
            table_scores[table_name]["best_score"] = max(table_scores[table_name]["best_score"], score)
            table_scores[table_name]["chunk_types"].add(match.metadata.get("chunk_type"))
            table_scores[table_name]["sample_content"] = match.metadata.get("metadata", "")
        ranked_tables = sorted(table_scores.values(), key=lambda x: (x["best_score"], x["total_score"]), reverse=True)
        return ranked_tables[:top_k]

    async def get_table_details(self, table_name: str) -> Dict[str, Any]:
        # Use dummy vector for filter-only query (Pinecone requires either vector or ID)
        dummy_vector = [0.0] * 3072  # text-embedding-3-large dimension
        results = self.index.query(
            vector=dummy_vector,
            filter={"table_name": {"$eq": table_name}}, 
            top_k=20, 
            include_metadata=True
        )
        table_details = {"table_name": table_name, "chunks": {}, "metadata": {}}
        for match in results.matches:
            chunk_type = match.metadata.get("chunk_type")
            table_details["chunks"][chunk_type] = {"metadata": match.metadata}
        return table_details
