"""
Pinecone Integration for Schema Vector Storage
Handles chunking, embedding, and similarity search of database schemas
"""
import os
import json
import asyncio
from pinecone import Pinecone, ServerlessSpec
import openai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
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
    # Performance tuning constants
    TABLE_BATCH_SIZE = int(os.getenv("INDEX_TABLE_BATCH_SIZE", "5"))  # Tables processed concurrently
    EMBEDDING_BATCH_SIZE = int(os.getenv("INDEX_EMBEDDING_BATCH_SIZE", "10"))  # Embeddings per API call
    UPSERT_BATCH_SIZE = int(os.getenv("INDEX_UPSERT_BATCH_SIZE", "100"))  # Vectors per Pinecone upsert
    SKIP_ROW_COUNTS = os.getenv("INDEX_SKIP_ROW_COUNTS", "true").lower() == "true"  # Skip slow COUNT queries
    
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
        # Validate token count before sending to API
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens > 7000:  # Conservative limit
            print(f"⚠️ Content too large ({estimated_tokens} tokens), this should have been pre-split!")
            # This should not happen with proper chunking, but as emergency fallback
            # Split the content and take the first valid chunk
            chunks = self._split_content_by_tokens(text, 4000)
            if chunks:
                text = chunks[0]
                print(f"   Using first chunk with {self._estimate_tokens(text)} tokens")
        
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 3 characters for safety)"""
        return len(text) // 3
    
    def _split_content_by_tokens(self, content: str, max_tokens: int = 4000) -> List[str]:
        """Split content into chunks that fit within token limits with NO DATA LOSS"""
        if self._estimate_tokens(content) <= max_tokens:
            return [content]
        
        # First try splitting by lines
        lines = content.split('\n')
        chunks = []
        current_chunk = ""
        
        for line in lines:
            # If adding this line would exceed limit, start new chunk
            if current_chunk and self._estimate_tokens(current_chunk + "\n" + line) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    # Single line is too long, split it by words
                    line_chunks = self._split_long_line(line, max_tokens)
                    chunks.extend(line_chunks[:-1])
                    current_chunk = line_chunks[-1]
            else:
                current_chunk += "\n" + line if current_chunk else line
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Second pass: check all chunks and split further if needed
        final_chunks = []
        for chunk in chunks:
            if self._estimate_tokens(chunk) > max_tokens:
                # Split by sentences, then words if necessary
                sentences = chunk.replace('. ', '.\n').split('\n')
                sentence_chunk = ""
                
                for sentence in sentences:
                    if sentence_chunk and self._estimate_tokens(sentence_chunk + '\n' + sentence) > max_tokens:
                        if sentence_chunk.strip():
                            final_chunks.append(sentence_chunk.strip())
                        sentence_chunk = sentence
                    else:
                        sentence_chunk += '\n' + sentence if sentence_chunk else sentence
                
                # If sentence chunk is still too big, split by words
                if sentence_chunk and self._estimate_tokens(sentence_chunk) > max_tokens:
                    words = sentence_chunk.split()
                    word_chunk = ""
                    for word in words:
                        if word_chunk and self._estimate_tokens(word_chunk + ' ' + word) > max_tokens:
                            if word_chunk.strip():
                                final_chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            word_chunk += ' ' + word if word_chunk else word
                    if word_chunk.strip():
                        final_chunks.append(word_chunk.strip())
                elif sentence_chunk.strip():
                    final_chunks.append(sentence_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks if final_chunks else [content[:10000]]  # Last resort fallback
    
    def _split_long_line(self, line: str, max_tokens: int) -> List[str]:
        """Split a very long line into smaller pieces"""
        max_chars = max_tokens * 3  # More conservative conversion
        chunks = []
        
        while len(line) > max_chars:
            # Try to split at word boundaries
            split_pos = line.rfind(' ', 0, max_chars)
            if split_pos == -1:
                split_pos = max_chars
            
            chunks.append(line[:split_pos])
            line = line[split_pos:].lstrip()
        
        if line:
            chunks.append(line)
        
        return chunks

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
        
        # Create rich table context - NO DATA LOSS, just split if needed
        row_count = table_info.get('row_count')
        row_count_text = f"Row Count: {row_count:,} rows" if isinstance(row_count, int) else f"Row Count: {row_count}"
        
        table_chunk_content = f"""Table: {table_name}
Schema: {table_schema}
Description: {table_description}
Total Columns: {len(column_names)}
{row_count_text}
Column Names: {', '.join(column_names)}
Column Types: {', '.join(column_types)}
Business Domain: {self._extract_business_domain(table_name)}
Data Category: {self._categorize_table(table_name, column_names)}
Table Purpose: {self._infer_table_purpose(table_name, column_names)}
Table Size: {"Large table" if isinstance(row_count, int) and row_count > 1000000 else "Medium table" if isinstance(row_count, int) and row_count > 10000 else "Small table" if isinstance(row_count, int) else "Unknown size"}"""
        
        # Split table overview if it's too large
        overview_parts = self._split_content_by_tokens(table_chunk_content)
        for i, part in enumerate(overview_parts):
            chunk_id = f"{table_name}_overview" if len(overview_parts) == 1 else f"{table_name}_overview_{i+1}"
            chunks.append(SchemaChunk(
                chunk_id=chunk_id,
                table_name=table_name,
                table_schema=table_schema,
                chunk_type="table_overview",
                content=part,
                metadata={
                    "total_columns": len(column_names),
                    "business_domain": self._extract_business_domain(table_name),
                    "data_category": self._categorize_table(table_name, column_names),
                    "row_count": table_info.get('row_count'),
                    "table_size_category": "large" if isinstance(row_count, int) and row_count > 1000000 else "medium" if isinstance(row_count, int) and row_count > 10000 else "small" if isinstance(row_count, int) else "unknown",
                    "part": i + 1 if len(overview_parts) > 1 else None
                }
            ))
        
        # 2. Column group chunks - group related columns together
        if columns:
            column_groups = self._group_columns_by_purpose(columns)
            for group_name, group_columns in column_groups.items():
                if not group_columns:
                    continue
                    
                # Include ALL column details - NO DATA LOSS
                group_content = f"""Table: {table_name} - {group_name} Columns
{', '.join([f"{col} ({info.get('data_type', 'unknown')})" for col, info in group_columns.items()])}
Column Group: {group_name}
Table Context: {table_description}
Business Purpose: {self._infer_table_purpose(table_name, column_names)}"""
                
                # Split group content if too large
                group_parts = self._split_content_by_tokens(group_content)
                for i, part in enumerate(group_parts):
                    chunk_id = f"{table_name}_columns_{group_name.lower().replace(' ', '_')}"
                    if len(group_parts) > 1:
                        chunk_id += f"_{i+1}"
                    
                    chunks.append(SchemaChunk(
                        chunk_id=chunk_id,
                        table_name=table_name,
                        table_schema=table_schema,
                        chunk_type="column_group",
                        content=part,
                        metadata={
                            "column_group": group_name,
                            "column_count": len(group_columns),
                            "columns": list(group_columns.keys()),
                            "part": i + 1 if len(group_parts) > 1 else None
                        }
                    ))
        
        # 3. Business context chunk - ALL semantic information preserved
        business_chunk_content = f"""Business Context for {table_name}
Business Domain: {self._extract_business_domain(table_name)}
Likely Use Cases: {self._generate_use_cases(table_name, column_names)}
Data Category: {self._categorize_table(table_name, column_names)}
Related Concepts: {self._extract_business_concepts(table_name, column_names)}
Table Purpose: {self._infer_table_purpose(table_name, column_names)}
Data Analytics Focus: {self._get_analytics_focus(table_name)}"""
        
        # Split business context if too large
        business_parts = self._split_content_by_tokens(business_chunk_content)
        for i, part in enumerate(business_parts):
            chunk_id = f"{table_name}_business_context" if len(business_parts) == 1 else f"{table_name}_business_context_{i+1}"
            chunks.append(SchemaChunk(
                chunk_id=chunk_id,
                table_name=table_name,
                table_schema=table_schema,
                chunk_type="business_context",
                content=part,
                metadata={
                    "business_domain": self._extract_business_domain(table_name),
                    "use_cases": self._generate_use_cases(table_name, column_names),
                    "concepts": self._extract_business_concepts(table_name, column_names),
                    "part": i + 1 if len(business_parts) > 1 else None
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

    async def index_database_schema(self, db_adapter, progress_callback=None):
        import time
        start_time = time.time()
        
        print("🚀 Indexing database schema in Pinecone with optimized batch processing...")
        result = db_adapter.run("SHOW TABLES IN SCHEMA ENHANCED_NBA", dry_run=False)
        if result.error:
            raise Exception(f"Failed to get tables: {result.error}")
        all_tables = [row[1] if len(row) > 1 else str(row[0]) for row in result.rows]
        print(f"📊 Found {len(all_tables)} tables to index")
        
        # Notify progress callback of total tables
        if progress_callback:
            progress_callback("start", total=len(all_tables))
        
        # Check existing indexed tables efficiently
        check_start = time.time()
        existing_tables = await self._get_indexed_tables_fast()
        check_time = time.time() - check_start
        print(f"📋 Found {len(existing_tables)} already indexed tables (check took {check_time:.1f}s)")
        
        # Filter out already indexed tables
        tables_to_index = [table for table in all_tables if table not in existing_tables]
        print(f"🎯 Need to index {len(tables_to_index)} new tables")
        
        if not tables_to_index:
            print("✅ All tables already indexed!")
            if progress_callback:
                progress_callback("complete")
            return
        
        # Performance configuration info
        print(f"⚙️ Performance config: {self.TABLE_BATCH_SIZE} tables/batch, {self.EMBEDDING_BATCH_SIZE} embeddings/batch, {self.UPSERT_BATCH_SIZE} vectors/upsert")
        
        # Process tables in batches for optimal performance
        batch_size = self.TABLE_BATCH_SIZE  # Configurable batch size
        indexed_count = 0
        total_chunks = 0
        processed_tables = len(existing_tables)  # Start with already indexed count
        
        for i in range(0, len(tables_to_index), batch_size):
            batch_start = time.time()
            batch = tables_to_index[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(tables_to_index) + batch_size - 1)//batch_size
            
            print(f"🔄 Processing batch {batch_num}/{total_batches}: {batch}")
            
            # Process each table in the batch individually for progress tracking
            all_chunks = []
            for table_name in batch:
                try:
                    if progress_callback:
                        progress_callback("table_start", current_table=table_name)
                    
                    result = await self._process_table_optimized(db_adapter, table_name)
                    if isinstance(result, Exception):
                        print(f"   ⚠️ Failed to process {table_name}: {result}")
                        if progress_callback:
                            progress_callback("error", current_table=table_name, error=str(result))
                    else:
                        chunks = result
                        all_chunks.extend(chunks)
                        processed_tables += 1
                        print(f"   ✅ Prepared {table_name}: {len(chunks)} chunks")
                        
                        if progress_callback:
                            progress_callback("table_complete", current_table=table_name, processed=processed_tables, total=len(all_tables))
                except Exception as e:
                    print(f"   ⚠️ Failed to process {table_name}: {e}")
                    if progress_callback:
                        progress_callback("error", current_table=table_name, error=str(e))
            
            # Batch generate embeddings for all chunks
            if all_chunks:
                if progress_callback:
                    progress_callback("table_start", current_table=f"Generating embeddings for batch {batch_num}")
                
                embed_start = time.time()
                await self._batch_generate_embeddings(all_chunks)
                embed_time = time.time() - embed_start
                
                if progress_callback:
                    progress_callback("table_start", current_table=f"Uploading vectors for batch {batch_num}")
                
                # Batch upsert all chunks at once
                upsert_start = time.time()
                await self._batch_upsert_chunks(all_chunks)
                upsert_time = time.time() - upsert_start
                
                total_chunks += len(all_chunks)
                batch_time = time.time() - batch_start
                
                print(f"   📤 Batch {batch_num} complete: {len(all_chunks)} vectors (embed: {embed_time:.1f}s, upsert: {upsert_time:.1f}s, total: {batch_time:.1f}s)")
        
        total_time = time.time() - start_time
        avg_time_per_table = total_time / len(tables_to_index) if tables_to_index else 0
        
        if progress_callback:
            progress_callback("complete")
        
        print(f"🎉 Pinecone schema indexing complete!")
        print(f"📈 Performance summary:")
        print(f"   • Indexed: {len(tables_to_index)} tables, Skipped: {len(existing_tables)}")
        print(f"   • Total vectors: {total_chunks}")
        print(f"   • Total time: {total_time:.1f}s")
        print(f"   • Average: {avg_time_per_table:.1f}s per table")

    async def _process_table_optimized(self, db_adapter, table_name: str):
        """Process a single table and return chunks (without embeddings yet)"""
        table_info = await self._get_table_info(db_adapter, table_name)
        table_chunks = self.chunk_schema_information(table_info)
        return table_chunks

    async def _batch_generate_embeddings(self, chunks):
        """Generate embeddings for chunks in batches"""
        batch_size = self.EMBEDDING_BATCH_SIZE  # Configurable batch size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            contents = [chunk.content for chunk in batch]
            
            # Generate embeddings in batch
            embeddings = await self._generate_embeddings_batch(contents)
            
            # Assign embeddings back to chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

    async def _generate_embeddings_batch(self, contents):
        """Generate embeddings for multiple contents at once"""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            
            response = await client.embeddings.create(
                model="text-embedding-3-large",
                input=contents,
                dimensions=3072
            )
            
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"⚠️ Batch embedding failed, falling back to individual: {e}")
            # Fallback to individual embeddings
            embeddings = []
            for content in contents:
                embedding = await self.generate_embedding(content)
                embeddings.append(embedding)
            return embeddings

    async def _batch_upsert_chunks(self, chunks):
        """Upsert multiple chunks to Pinecone in batches"""
        batch_size = self.UPSERT_BATCH_SIZE  # Configurable batch size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare vectors for upsert
            vectors = []
            for chunk in batch:
                vectors.append((
                    chunk.chunk_id,
                    chunk.embedding,
                    {
                        "table_name": chunk.table_name,
                        "chunk_type": chunk.chunk_type,
                        "metadata": json.dumps(chunk.metadata)
                    }
                ))
            
            # Batch upsert
            self.index.upsert(vectors)

    async def _get_indexed_tables_fast(self) -> set:
        """Get list of tables that are already indexed in Pinecone efficiently"""
        try:
            # Get index stats first to check if there are any vectors
            stats = self.index.describe_index_stats()
            if stats.total_vector_count == 0:
                return set()
            
            # Use a more efficient approach - query with namespaces or use list_ids if available
            # For now, use a smaller query to get table names
            dummy_vector = [0.0] * 3072
            
            # Query in smaller batches to get all table names
            indexed_tables = set()
            top_k = 1000  # Smaller batch size
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            for match in results.matches:
                table_name = match.metadata.get("table_name")
                if table_name:
                    indexed_tables.add(table_name)
            
            return indexed_tables
        except Exception as e:
            print(f"⚠️ Error checking indexed tables: {e}")
            return set()  # Return empty set if error, will re-index all

    async def _get_table_info(self, db_adapter, table_name: str) -> Dict[str, Any]:
        """Get table information with optimized queries"""
        try:
            # Check if we should include row counts from environment settings
            skip_row_counts = self.SKIP_ROW_COUNTS
            
            # Always get column information with proper Snowflake quoting
            qualified_table_name = f'"COMMERCIAL_AI"."ENHANCED_NBA"."{table_name}"'
            columns_task = asyncio.create_task(
                asyncio.to_thread(db_adapter.run, f"DESCRIBE TABLE {qualified_table_name}", False)
            )
            
            # Conditionally get row count based on settings
            count_task = None
            if not skip_row_counts:
                count_task = asyncio.create_task(
                    asyncio.to_thread(db_adapter.run, f"SELECT COUNT(*) FROM {qualified_table_name}", False)
                )
            
            columns_result = await columns_task
            count_result = None
            if count_task:
                count_result = await count_task
            
            columns = {}
            if not columns_result.error:
                for row in columns_result.rows:
                    col_name = row[0]
                    col_type = row[1]
                    nullable = row[2] == 'Y' if len(row) > 2 else False
                    columns[col_name] = {
                        "data_type": col_type, 
                        "nullable": nullable, 
                        "description": None
                    }
            
            # Get row count if enabled and query succeeded
            row_count = None
            if count_result and not count_result.error and count_result.rows:
                row_count = count_result.rows[0][0]
            elif not skip_row_counts:
                # If row count was requested but failed, set to "Unknown"
                row_count = "Unknown"
            
            return {
                "name": table_name,
                "schema": "ENHANCED_NBA",
                "table_type": "table",
                "columns": columns,
                "row_count": row_count,
                "description": f"Table containing {table_name.replace('_', ' ').lower()} data"
            }
        except Exception as e:
            print(f"⚠️ Error getting table info for {table_name}: {e}")
            # Return minimal table info on error
            return {
                "name": table_name,
                "schema": "ENHANCED_NBA",
                "table_type": "table",
                "columns": {},
                "row_count": None,
                "description": f"Table {table_name}"
            }

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
        import json
        
        # Use dummy vector for filter-only query (Pinecone requires either vector or ID)
        dummy_vector = [0.0] * 3072  # text-embedding-3-large dimension
        results = self.index.query(
            vector=dummy_vector,
            filter={"table_name": {"$eq": table_name}}, 
            top_k=20, 
            include_metadata=True
        )
        
        table_details = {"table_name": table_name, "chunks": {}, "metadata": {}, "columns": []}
        all_columns = []
        
        for match in results.matches:
            chunk_type = match.metadata.get("chunk_type")
            metadata = match.metadata
            
            # Process column_group chunks to extract actual columns
            if chunk_type == "column_group":
                # Check if columns are in direct metadata
                columns = metadata.get("columns", [])
                
                # If not found, check nested metadata JSON
                if not columns and "metadata" in metadata:
                    try:
                        nested_metadata = json.loads(metadata["metadata"])
                        columns = nested_metadata.get("columns", [])
                        column_group = nested_metadata.get("column_group", "unknown")
                        print(f"📋 Extracted {len(columns)} columns from group '{column_group}': {columns}")
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"❌ Failed to parse nested metadata: {e}")
                        columns = []
                
                if columns:
                    all_columns.extend(columns)
            
            table_details["chunks"][chunk_type] = {"metadata": metadata}
        
        # Store all found columns
        table_details["columns"] = all_columns
        print(f"✅ Total columns found for {table_name}: {len(all_columns)} - {all_columns}")
        
        return table_details
