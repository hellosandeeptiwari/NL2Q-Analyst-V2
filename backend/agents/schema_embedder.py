"""
Schema Embedder for enhanced semantic understanding with LLM Intelligence
Converts database schema to embeddings and generates insights
Optimized for fast batch processing and parallel execution
Now integrates LLM-driven schema intelligence during indexing
"""
import json
import os
import openai
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pickle
from datetime import datetime
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_schema_intelligence import VectorSchemaStorage

@dataclass
class TableSchema:
    """Enhanced table schema with metadata and LLM intelligence"""
    table_name: str
    columns: List[Dict[str, str]]  # [{"name": "col", "type": "VARCHAR", "description": "..."}]
    row_count: Optional[int] = None
    primary_keys: List[str] = None
    foreign_keys: List[Dict[str, str]] = None
    description: str = ""
    embedding: Optional[np.ndarray] = None
    # New LLM intelligence fields
    llm_analysis: Optional[Dict[str, Any]] = None
    business_purpose: str = ""
    domain_classification: str = ""
    query_guidance: Optional[Dict[str, Any]] = None

class SchemaEmbedder:
    def __init__(self, api_key: str = None, batch_size: int = 100, max_workers: int = 3):
        self.api_key = os.getenv('OPENAI_API_KEY') or api_key
        if self.api_key:
            openai.api_key = self.api_key
            print("✅ OpenAI API key loaded for schema embedding")
        else:
            print("⚠️ No OpenAI API key - schema embedding unavailable")
        
        self.model = "text-embedding-3-small"
        self.batch_size = batch_size  # OpenAI allows up to 100 inputs per request
        self.max_workers = max_workers  # Parallel threads
        self.schemas: Dict[str, TableSchema] = {}
        self.cache_file = "backend/storage/schema_embeddings_enhanced.pkl"
        self.schema_cache_file = "backend/storage/schema_metadata.json"  # Raw schema cache
        self.intelligence_cache_file = "backend/storage/schema_intelligence.json"  # LLM intelligence cache
        
        # Initialize LLM intelligence system
        self.llm_intelligence = VectorSchemaStorage()
        
        # Performance tracking
        self.stats = {
            "total_tables": 0,
            "cached_tables": 0,
            "embedded_tables": 0,
            "skipped_tables": 0,
            "embedding_time": 0,
            "llm_analysis_time": 0
        }
        
    def extract_schema_from_db(self, adapter, max_workers: int = 5, use_cache: bool = True, use_bulk: bool = True) -> Dict[str, Dict]:
        """Extract comprehensive schema information with ultra-fast bulk processing"""
        print("📋 Extracting enhanced schema from database...")
        
        # Try to load cached schemas first
        if use_cache:
            cached_schemas = self._load_schema_cache()
            if cached_schemas:
                print(f"📁 Loaded {len(cached_schemas)} schemas from cache")
                return cached_schemas
        
        start_time = time.time()
        
        # Use ultra-fast bulk extraction if available
        if use_bulk:
            schemas = self._extract_schemas_bulk(adapter)
            if schemas:
                total_time = time.time() - start_time
                print(f"⚡ Ultra-fast bulk extraction: {len(schemas)} schemas in {total_time:.2f}s")
                print(f"   📊 Rate: {len(schemas)/total_time:.1f} tables/second")
                
                # Cache the results
                if use_cache:
                    self._save_schema_cache(schemas)
                
                return schemas
        
        # Fallback to parallel extraction
        return self._extract_schemas_parallel(adapter, max_workers, use_cache)
    
    def _extract_schemas_bulk(self, adapter) -> Dict[str, Dict]:
        """Ultra-fast bulk schema extraction using metadata queries"""
        try:
            print("⚡ Using ultra-fast bulk metadata extraction...")
            
            # Single query to get ALL column information at once
            bulk_query = """
            SELECT 
                table_name,
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns 
            WHERE table_schema = CURRENT_SCHEMA()
            ORDER BY table_name, ordinal_position
            """
            
            result = adapter.run(bulk_query)
            
            if result.error:
                print(f"⚠️ Bulk query failed, falling back to individual queries: {result.error}")
                return {}
            
            # Group columns by table
            table_columns = {}
            for row in result.rows:
                table_name = row[0]
                column_info = {
                    "name": row[1],
                    "type": row[2],
                    "nullable": row[3]
                }
                
                if table_name not in table_columns:
                    table_columns[table_name] = []
                table_columns[table_name].append(column_info)
            
            # Create schema objects
            schemas = {}
            for table_name, columns in table_columns.items():
                description = self._generate_table_description_fast(table_name, columns)
                
                schemas[table_name] = {
                    "table_name": table_name,
                    "columns": columns,
                    "row_count": None,  # Skip for speed
                    "description": description,
                    "column_count": len(columns)
                }
            
            return schemas
            
        except Exception as e:
            print(f"⚠️ Bulk extraction failed: {e}")
            return {}
    
    def _extract_schemas_parallel(self, adapter, max_workers: int, use_cache: bool) -> Dict[str, Dict]:
        """Parallel schema extraction (fallback method)"""
        schemas = {}
        
        try:
            # Step 1: Get all tables efficiently
            start_time = time.time()
            tables_result = adapter.run("SHOW TABLES")
            if tables_result.error:
                print(f"❌ Error getting tables: {tables_result.error}")
                return schemas
                
            table_names = [row[1] for row in tables_result.rows]
            tables_time = time.time() - start_time
            print(f"📊 Found {len(table_names)} tables in {tables_time:.2f}s")
            
            # Step 2: Parallel schema extraction
            print(f"🚀 Extracting schemas using {max_workers} parallel workers...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all schema extraction jobs
                future_to_table = {
                    executor.submit(self._get_table_schema_fast, adapter, table_name): table_name 
                    for table_name in table_names
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_table):
                    table_name = future_to_table[future]
                    try:
                        schema_info = future.result()
                        if schema_info:
                            schemas[table_name] = schema_info
                        
                        completed += 1
                        if completed % 20 == 0:  # Progress update every 20 tables
                            print(f"   ✅ Processed {completed}/{len(table_names)} tables")
                            
                    except Exception as e:
                        print(f"⚠️ Failed to extract schema for {table_name}: {e}")
            
            total_time = time.time() - start_time
            print(f"✅ Parallel extraction completed in {total_time:.2f}s")
            print(f"   📊 Rate: {len(schemas)/total_time:.1f} tables/second")
            
            # Cache the results
            if use_cache:
                self._save_schema_cache(schemas)
            
            return schemas
            
        except Exception as e:
            print(f"❌ Parallel extraction failed: {e}")
            return schemas
    
    def _load_schema_cache(self) -> Optional[Dict[str, Dict]]:
        """Load cached schema metadata"""
        try:
            if os.path.exists(self.schema_cache_file):
                with open(self.schema_cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is recent (less than 24 hours old)
                cache_time = datetime.fromisoformat(cache_data.get('created_at', '2000-01-01'))
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                if age_hours < 24:  # Cache valid for 24 hours
                    return cache_data.get('schemas', {})
                else:
                    print(f"⚠️ Schema cache is {age_hours:.1f} hours old, refreshing...")
        except Exception as e:
            print(f"⚠️ Failed to load schema cache: {e}")
        return None
    
    def _save_schema_cache(self, schemas: Dict[str, Dict]):
        """Save schema metadata to cache"""
        try:
            os.makedirs(os.path.dirname(self.schema_cache_file), exist_ok=True)
            cache_data = {
                'schemas': schemas,
                'created_at': datetime.now().isoformat(),
                'total_tables': len(schemas)
            }
            with open(self.schema_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"💾 Cached {len(schemas)} schema definitions")
        except Exception as e:
            print(f"⚠️ Failed to cache schemas: {e}")
    
    def _get_table_schema_fast(self, adapter, table_name: str) -> Optional[Dict]:
        """Optimized version for fast schema extraction"""
        try:
            # Use a single, efficient query to get column information
            desc_result = adapter.run(f'DESCRIBE TABLE "{table_name}"')
            if desc_result.error:
                # Fallback for different database types
                desc_result = adapter.run(f'SHOW COLUMNS FROM "{table_name}"')
                
            if desc_result.error:
                return None
                
            columns = []
            for row in desc_result.rows:
                column_info = {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] if len(row) > 2 else "YES"
                }
                columns.append(column_info)
            
            # Skip row count for speed (can be estimated or fetched separately if needed)
            row_count = None
            
            # Generate quick description
            description = self._generate_table_description_fast(table_name, columns)
            
            return {
                "table_name": table_name,
                "columns": columns,
                "row_count": row_count,
                "description": description,
                "column_count": len(columns)
            }
            
        except Exception as e:
            # Silent failure for speed - log only critical errors
            return None
    
    def _generate_table_description_fast(self, table_name: str, columns: List[Dict]) -> str:
        """Fast description generation with minimal processing"""
        name_lower = table_name.lower()
        
        # Quick pattern matching
        if 'analytics' in name_lower or 'azure' in name_lower:
            desc = f"Azure Analytics data table with {len(columns)} columns"
        elif 'customer' in name_lower:
            desc = f"Customer data table with {len(columns)} columns"
        elif 'order' in name_lower:
            desc = f"Order transaction table with {len(columns)} columns"
        elif 'product' in name_lower:
            desc = f"Product data table with {len(columns)} columns"
        else:
            desc = f"Business data table '{table_name}' with {len(columns)} columns"
        
        # Quick column type summary
        text_cols = sum(1 for col in columns if 'varchar' in col['type'].lower() or 'text' in col['type'].lower())
        num_cols = sum(1 for col in columns if any(t in col['type'].lower() for t in ['int', 'number', 'float', 'decimal']))
        
        if text_cols > 0 or num_cols > 0:
            desc += f" ({text_cols} text, {num_cols} numeric)"
            
        return desc
        """Get detailed schema for a specific table"""
        try:
            # Get column information
            desc_result = adapter.run(f'DESCRIBE TABLE "{table_name}"')
            if desc_result.error:
                desc_result = adapter.run(f'SHOW COLUMNS FROM "{table_name}"')
                
            if desc_result.error:
                print(f"⚠️ Could not describe table {table_name}")
                return None
                
            columns = []
            for row in desc_result.rows:
                # Handle different DESCRIBE formats
                if len(row) >= 2:
                    col_info = {
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] if len(row) > 2 else "YES",
                        "default": row[3] if len(row) > 3 else None
                    }
                    columns.append(col_info)
            
            # Try to get row count
            row_count = None
            try:
                count_result = adapter.run(f'SELECT COUNT(*) FROM "{table_name}"')
                if not count_result.error and count_result.rows:
                    row_count = count_result.rows[0][0]
            except:
                pass
                
            # Generate natural description
            description = self._generate_table_description(table_name, columns)
            
            return {
                "table_name": table_name,
                "columns": columns,
                "row_count": row_count,
                "description": description,
                "column_count": len(columns)
            }
            
        except Exception as e:
            print(f"⚠️ Error getting schema for {table_name}: {e}")
            return None
    
    def _generate_table_description(self, table_name: str, columns: List[Dict]) -> str:
        """Generate natural language description of table"""
        # Analyze table name
        name_parts = table_name.lower().replace('_', ' ').replace('-', ' ').split()
        
        desc = f"Table '{table_name}' "
        
        # Infer purpose from name
        if any(word in name_parts for word in ['analytics', 'azure', 'data', 'analysis']):
            desc += "containing Azure Analytics data "
        elif any(word in name_parts for word in ['customer', 'client', 'user']):
            desc += "containing customer/user information "
        elif any(word in name_parts for word in ['order', 'transaction', 'purchase']):
            desc += "containing transaction/order data "
        elif any(word in name_parts for word in ['product', 'item', 'inventory']):
            desc += "containing product/inventory data "
        else:
            desc += "containing business data "
            
        # Add column insights
        col_types = {}
        for col in columns:
            col_type = col['type'].upper()
            if 'VARCHAR' in col_type or 'TEXT' in col_type:
                col_types['text'] = col_types.get('text', 0) + 1
            elif 'INT' in col_type or 'NUMBER' in col_type or 'FLOAT' in col_type:
                col_types['numeric'] = col_types.get('numeric', 0) + 1
            elif 'DATE' in col_type or 'TIME' in col_type:
                col_types['datetime'] = col_types.get('datetime', 0) + 1
                
        desc += f"with {len(columns)} columns: "
        if col_types.get('text', 0) > 0:
            desc += f"{col_types['text']} text fields, "
        if col_types.get('numeric', 0) > 0:
            desc += f"{col_types['numeric']} numeric fields, "
        if col_types.get('datetime', 0) > 0:
            desc += f"{col_types['datetime']} date/time fields, "
            
        desc = desc.rstrip(', ')
        
        # Add common column patterns
        col_names = [col['name'].lower() for col in columns]
        if any('id' in name for name in col_names):
            desc += ". Contains identifier columns"
        if any('name' in name for name in col_names):
            desc += ". Contains name/title fields"
        if any(word in ' '.join(col_names) for word in ['score', 'point', 'stat']):
            desc += ". Contains statistical/scoring data"
            
    def _quick_filter_schemas(self, schemas: Dict[str, Dict]) -> Dict[str, Dict]:
        """Quick filtering to remove problematic schemas before embedding"""
        filtered = {}
        
        for table_name, schema_info in schemas.items():
            # Skip if already cached
            if table_name in self.schemas:
                self.stats["cached_tables"] += 1
                continue
                
            # Quick size check
            column_count = len(schema_info.get('columns', []))
            
            # Skip extremely wide tables (>500 columns)
            if column_count > 500:
                print(f"⚠️ Skipping {table_name}: Too many columns ({column_count})")
                self.stats["skipped_tables"] += 1
                continue
                
            # Estimate text size quickly
            avg_col_length = 50  # Average column name + type length
            estimated_size = column_count * avg_col_length + 500  # Base text
            
            # Skip if estimated to be too large
            if estimated_size > 25000:  # Conservative limit
                print(f"⚠️ Skipping {table_name}: Estimated too large ({estimated_size} chars)")
                self.stats["skipped_tables"] += 1
                continue
                
            filtered[table_name] = schema_info
            
        print(f"📊 Filtered: {len(filtered)} tables to embed, {self.stats['cached_tables']} cached, {self.stats['skipped_tables']} skipped")
        return filtered
    
    def _create_enhanced_embeddings_with_llm(self, schemas: Dict[str, Dict], 
                                           schema_intelligence: Dict[str, Any]) -> Dict[str, TableSchema]:
        """Create enhanced embeddings using LLM intelligence insights"""
        embedded_schemas = dict(self.schemas)  # Start with cached
        
        print("📊 Creating enhanced embeddings with LLM insights...")
        
        for table_name, schema_info in schemas.items():
            try:
                # Get LLM analysis for this table
                table_analysis = schema_intelligence.get("table_analyses", {}).get(table_name, {})
                
                # Create enhanced embedding text with LLM insights
                enhanced_text = self._create_enhanced_embedding_text(schema_info, table_analysis)
                
                # Generate embedding
                from openai import OpenAI
                client = OpenAI()
                response = client.embeddings.create(
                    input=enhanced_text,
                    model=self.model
                )
                embedding = np.array(response.data[0].embedding)
                
                # Create enhanced TableSchema with LLM intelligence
                table_schema = TableSchema(
                    table_name=table_name,
                    columns=schema_info['columns'],
                    row_count=schema_info.get('row_count'),
                    description=schema_info['description'],
                    embedding=embedding,
                    # Enhanced with LLM intelligence
                    llm_analysis=table_analysis,
                    business_purpose=table_analysis.get('business_purpose', ''),
                    domain_classification=table_analysis.get('domain', ''),
                    query_guidance=table_analysis.get('query_guidance', {})
                )
                
                embedded_schemas[table_name] = table_schema
                self.stats["embedded_tables"] += 1
                
                print(f"   ✅ Enhanced embedding: {table_name} ({table_analysis.get('domain', 'unknown')} domain)")
                
            except Exception as e:
                print(f"⚠️ Failed enhanced embedding for {table_name}: {e}")
                # Fallback to basic embedding
                embedded_schemas[table_name] = self._create_basic_table_schema(table_name, schema_info)
        
        return embedded_schemas
    
    def _create_enhanced_embedding_text(self, schema_info: Dict, table_analysis: Dict) -> str:
        """Create embedding text enriched with LLM intelligence"""
        text_parts = []
        
        # Basic table info
        text_parts.append(f"Table: {schema_info['table_name']}")
        
        # LLM-enhanced business context
        if table_analysis.get('business_purpose'):
            text_parts.append(f"Business Purpose: {table_analysis['business_purpose']}")
        else:
            text_parts.append(f"Description: {schema_info['description']}")
        
        # Domain classification from LLM
        if table_analysis.get('domain'):
            text_parts.append(f"Domain: {table_analysis['domain']}")
        
        # Enhanced column information with LLM insights
        columns = schema_info['columns']
        column_insights = table_analysis.get('column_insights', [])
        
        text_parts.append(f"Columns ({len(columns)}) with business context:")
        
        # Create lookup for column insights
        insight_lookup = {insight['column_name']: insight for insight in column_insights}
        
        for col in columns[:30]:  # Limit for token management
            col_text = f"- {col['name']} ({col['type']})"
            
            # Add LLM insights if available
            if col['name'] in insight_lookup:
                insight = insight_lookup[col['name']]
                semantic_role = insight.get('semantic_role', '')
                business_meaning = insight.get('business_meaning', '')
                
                if semantic_role:
                    col_text += f" [Role: {semantic_role}]"
                if business_meaning:
                    col_text += f" - {business_meaning}"
                
                # Add operation guidance
                operations = insight.get('data_operations', [])
                if 'SUM' in operations or 'AVG' in operations:
                    col_text += " [NUMERIC - can aggregate]"
                elif 'GROUP_BY' in operations:
                    col_text += " [CATEGORICAL - for grouping]"
            
            text_parts.append(col_text)
        
        if len(columns) > 30:
            text_parts.append(f"... and {len(columns) - 30} more columns")
        
        # Add query guidance from LLM
        query_guidance = table_analysis.get('query_guidance', {})
        if query_guidance.get('primary_amount_fields'):
            text_parts.append(f"Primary amount fields: {', '.join(query_guidance['primary_amount_fields'])}")
        if query_guidance.get('key_identifiers'):
            text_parts.append(f"Key identifiers: {', '.join(query_guidance['key_identifiers'])}")
        if query_guidance.get('forbidden_operations'):
            text_parts.append(f"Forbidden operations: {', '.join(query_guidance['forbidden_operations'])}")
        
        # Add relationship context
        relationships = table_analysis.get('relationship_potential', [])
        if relationships:
            rel_text = "Relationships: "
            rel_descriptions = []
            for rel in relationships[:5]:  # Limit relationships
                rel_descriptions.append(f"{rel['column']} -> {rel.get('likely_related_tables', ['unknown'])[0]}")
            rel_text += ", ".join(rel_descriptions)
            text_parts.append(rel_text)
        
        return '\n'.join(text_parts)
    
    def _create_basic_table_schema(self, table_name: str, schema_info: Dict) -> TableSchema:
        """Fallback method to create basic table schema without LLM insights"""
        return TableSchema(
            table_name=table_name,
            columns=schema_info['columns'],
            row_count=schema_info.get('row_count'),
            description=schema_info['description'],
            embedding=None  # Will be created later
        )
    
    def _save_intelligence_cache(self, schema_intelligence: Dict[str, Any]):
        """Save LLM intelligence to cache for quick retrieval"""
        try:
            os.makedirs(os.path.dirname(self.intelligence_cache_file), exist_ok=True)
            with open(self.intelligence_cache_file, 'w') as f:
                json.dump(schema_intelligence, f, indent=2)
            print(f"💾 Cached LLM intelligence for {len(schema_intelligence.get('table_analyses', {}))} tables")
        except Exception as e:
            print(f"⚠️ Failed to cache LLM intelligence: {e}")
    
    def load_intelligence_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached LLM intelligence"""
        try:
            if os.path.exists(self.intelligence_cache_file):
                with open(self.intelligence_cache_file, 'r') as f:
                    intelligence = json.load(f)
                
                # Check if cache is recent (less than 24 hours old)
                cache_time = datetime.fromisoformat(intelligence.get('generated_timestamp', '2000-01-01'))
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                if age_hours < 24:  # Cache valid for 24 hours
                    return intelligence
                else:
                    print(f"⚠️ Intelligence cache is {age_hours:.1f} hours old, will refresh...")
        except Exception as e:
            print(f"⚠️ Failed to load intelligence cache: {e}")
        return None
    
    def create_embeddings(self, schemas: Dict[str, Dict]) -> Dict[str, TableSchema]:
        """Create embeddings for all schemas with optimized batch processing"""
        if not self.api_key:
            print("⚠️ No OpenAI API key - skipping embedding generation")
            return {}
        
        start_time = time.time()
        self.stats["total_tables"] = len(schemas)
        
        print(f"� Optimized embedding for {len(schemas)} tables...")
        
        # Step 1: Quick filtering
        filtered_schemas = self._quick_filter_schemas(schemas)
        
        if not filtered_schemas:
            print("✅ All tables cached or filtered out")
            return self.schemas
        
        # Step 2: Prepare batches
        schema_items = list(filtered_schemas.items())
        batches = [schema_items[i:i + self.batch_size] 
                  for i in range(0, len(schema_items), self.batch_size)]
        
        print(f"📦 Processing {len(batches)} batches with up to {self.batch_size} tables each")
        
        # Step 3: Parallel batch processing
        embedded_schemas = dict(self.schemas)  # Start with cached
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self._create_embeddings_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    for table_name, table_schema in batch_results:
                        embedded_schemas[table_name] = table_schema
                        self.stats["embedded_tables"] += 1
                        
                    print(f"✅ Completed batch {batch_idx + 1}/{len(batches)}")
                    
                except Exception as e:
                    print(f"❌ Batch {batch_idx + 1} failed: {e}")
        
        # Update schemas and cache
        self.schemas = embedded_schemas
        self._save_cache(embedded_schemas)
        
        # Performance summary
        total_time = time.time() - start_time
        self.stats["embedding_time"] = total_time
        
        print(f"🎉 Embedding complete in {total_time:.2f}s:")
        print(f"   ✅ Embedded: {self.stats['embedded_tables']} tables")
        print(f"   📁 Cached: {self.stats['cached_tables']} tables")
        print(f"   ⚠️ Skipped: {self.stats['skipped_tables']} tables")
        print(f"   📊 Rate: {self.stats['embedded_tables']/total_time:.1f} tables/sec")
        
        return embedded_schemas
        
        return embedded_schemas
    
    def _create_embedding_text(self, schema_info: Dict) -> str:
        """Create rich text for embedding generation with token limit handling"""
        text_parts = []
        
        # Table information (always include)
        text_parts.append(f"Database table: {schema_info['table_name']}")
        text_parts.append(f"Description: {schema_info['description']}")
        
        if schema_info.get('row_count'):
            text_parts.append(f"Contains {schema_info['row_count']} rows")
            
        # Column information with chunking for large schemas
        columns = schema_info['columns']
        text_parts.append(f"Columns ({len(columns)}):")
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        base_text = '\n'.join(text_parts)
        base_tokens = len(base_text) // 4
        
        # Reserve space for table info and some columns (target max: 7000 tokens to be safe)
        max_tokens = 7000
        available_tokens = max_tokens - base_tokens - 200  # buffer
        
        # Add columns within token limit
        columns_text = []
        current_tokens = 0
        
        for i, col in enumerate(columns):
            col_desc = f"- {col['name']} ({col['type']})"
            if col.get('nullable') == 'NO':
                col_desc += " NOT NULL"
                
            col_tokens = len(col_desc) // 4
            
            if current_tokens + col_tokens > available_tokens and i > 10:  # Always include at least 10 columns
                # Add summary for remaining columns
                remaining = len(columns) - i
                columns_text.append(f"... and {remaining} more columns")
                break
            else:
                columns_text.append(col_desc)
                current_tokens += col_tokens
        
        text_parts.extend(columns_text)
        
        # Add semantic context (column names summary)
        if len(columns) <= 50:  # For reasonable sized schemas
            col_names = [col['name'] for col in columns]
            text_parts.append(f"Column names: {', '.join(col_names)}")
        else:
            # For very wide tables, just include first 30 column names
            col_names = [col['name'] for col in columns[:30]]
            text_parts.append(f"First 30 column names: {', '.join(col_names)}")
            text_parts.append(f"Total columns: {len(columns)}")
        
        final_text = '\n'.join(text_parts)
        
        # Final safety check - if still too long, truncate
        if len(final_text) > 30000:  # ~7500 tokens
            print(f"⚠️ Schema for {schema_info['table_name']} still too long, truncating...")
            final_text = final_text[:30000] + "... [truncated]"
            
        return final_text
    
    def _create_embeddings_batch(self, batch_data: List[tuple]) -> List[tuple]:
        """Create embeddings for a batch of schemas efficiently"""
        if not batch_data:
            return []
            
        try:
            # Prepare batch inputs
            batch_texts = []
            batch_info = []
            
            for table_name, schema_info in batch_data:
                embed_text = self._create_embedding_text_fast(schema_info)
                batch_texts.append(embed_text)
                batch_info.append((table_name, schema_info))
            
            # Single API call for entire batch
            start_time = time.time()
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                input=batch_texts,
                model=self.model
            )
            api_time = time.time() - start_time
            
            # Process results
            results = []
            for i, (table_name, schema_info) in enumerate(batch_info):
                try:
                    embedding = np.array(response.data[i].embedding)
                    
                    table_schema = TableSchema(
                        table_name=table_name,
                        columns=schema_info['columns'],
                        row_count=schema_info.get('row_count'),
                        description=schema_info['description'],
                        embedding=embedding
                    )
                    
                    results.append((table_name, table_schema))
                    
                except Exception as e:
                    print(f"❌ Failed to process {table_name}: {e}")
            
            print(f"✅ Batch embedded {len(results)}/{len(batch_data)} tables in {api_time:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ Batch embedding failed: {e}")
            return []
    
    def _create_embedding_text_fast(self, schema_info: Dict) -> str:
        """Optimized version for fast text creation"""
        # Pre-calculate limits
        max_columns = 30  # Limit column details
        
        text_parts = [
            f"Table: {schema_info['table_name']}",
            f"Description: {schema_info['description']}"
        ]
        
        if schema_info.get('row_count'):
            text_parts.append(f"Rows: {schema_info['row_count']}")
        
        columns = schema_info['columns'][:max_columns]  # Limit columns
        text_parts.append(f"Columns ({len(schema_info['columns'])}):")
        
        # Add limited column info
        for col in columns:
            text_parts.append(f"- {col['name']} ({col['type']})")
            
        if len(schema_info['columns']) > max_columns:
            remaining = len(schema_info['columns']) - max_columns
            text_parts.append(f"... and {remaining} more columns")
        
        # Add column names summary (limited)
        col_names = [col['name'] for col in schema_info['columns'][:20]]
        text_parts.append(f"Key columns: {', '.join(col_names)}")
        
        return '\n'.join(text_parts)
    
    def find_relevant_tables(self, query: str, top_k: int = 5) -> List[tuple]:
        """Find tables most relevant to user query using embeddings"""
        if not self.api_key or not self.schemas:
            return []
            
        try:
            # Get query embedding
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                input=query,
                model=self.model
            )
            query_embedding = np.array(response.data[0].embedding)
            
            # Calculate similarities
            similarities = []
            for table_name, schema in self.schemas.items():
                if schema.embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, schema.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(schema.embedding)
                    )
                    similarities.append((table_name, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            print(f"❌ Error finding relevant tables: {e}")
            return []
    
    def get_schema_json(self, table_names: List[str] = None) -> Dict[str, Any]:
        """Get schema information in JSON format for code generation"""
        if table_names is None:
            table_names = list(self.schemas.keys())
            
        schema_json = {
            "tables": {},
            "metadata": {
                "total_tables": len(table_names),
                "generated_at": datetime.now().isoformat()
            }
        }
        
        for table_name in table_names:
            if table_name in self.schemas:
                schema = self.schemas[table_name]
                schema_json["tables"][table_name] = {
                    "columns": schema.columns,
                    "row_count": schema.row_count,
                    "description": schema.description,
                    "column_count": len(schema.columns)
                }
        
        return schema_json
    
    def _save_cache(self, schemas: Dict[str, TableSchema]):
        """Save schemas to cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(schemas, f)
            print(f"💾 Cached {len(schemas)} schemas")
        except Exception as e:
            print(f"⚠️ Failed to cache schemas: {e}")
    
    def load_cache(self) -> bool:
        """Load schemas from cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.schemas = pickle.load(f)
                print(f"📁 Loaded {len(self.schemas)} schemas from cache")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
        return False
    
    def get_query_intelligence(self, table_names: List[str]) -> Dict[str, Any]:
        """Get LLM intelligence for specific tables to enhance SQL generation"""
        intelligence = {
            "table_insights": {},
            "cross_table_guidance": {},
            "query_patterns": {}
        }
        
        for table_name in table_names:
            if table_name in self.schemas:
                schema = self.schemas[table_name]
                if schema.llm_analysis:
                    intelligence["table_insights"][table_name] = {
                        "business_purpose": schema.business_purpose,
                        "domain": schema.domain_classification,
                        "query_guidance": schema.query_guidance,
                        "column_insights": schema.llm_analysis.get('column_insights', [])
                    }
        
        # Load cross-table intelligence from cache
        cached_intelligence = self.load_intelligence_cache()
        if cached_intelligence:
            intelligence["cross_table_guidance"] = cached_intelligence.get("cross_table_intelligence", {})
            intelligence["query_patterns"] = cached_intelligence.get("cross_table_intelligence", {}).get("query_patterns", {})
        
        return intelligence
    
    def get_amount_columns_for_table(self, table_name: str) -> List[str]:
        """Get columns that can be used for mathematical operations (SUM, AVG)"""
        if table_name not in self.schemas:
            return []
        
        schema = self.schemas[table_name]
        if not schema.llm_analysis:
            return []
        
        amount_columns = []
        for col_insight in schema.llm_analysis.get('column_insights', []):
            operations = col_insight.get('data_operations', [])
            if 'SUM' in operations or 'AVG' in operations:
                amount_columns.append(col_insight['column_name'])
        
        return amount_columns
    
    def get_forbidden_operations_for_table(self, table_name: str) -> List[str]:
        """Get operations that should not be performed on this table"""
        if table_name not in self.schemas:
            return []
        
        schema = self.schemas[table_name]
        if not schema.query_guidance:
            return []
        
        return schema.query_guidance.get('forbidden_operations', [])
    
    def get_relationship_guidance(self, table_names: List[str]) -> List[Dict[str, Any]]:
        """Get relationship guidance for joining tables"""
        cached_intelligence = self.load_intelligence_cache()
        if not cached_intelligence:
            return []
        
        relationships = cached_intelligence.get("cross_table_intelligence", {}).get("relationships", [])
        
        # Filter relationships relevant to the requested tables
        relevant_relationships = []
        for rel in relationships:
            if rel['from_table'] in table_names or rel['to_table'] in table_names:
                relevant_relationships.append(rel)
        
        return relevant_relationships

    def get_status(self) -> Dict[str, Any]:
        """Get embedder status including LLM intelligence"""
        basic_status = {
            "api_available": self.api_key is not None,
            "tables_embedded": len(self.schemas),
            "cache_available": os.path.exists(self.cache_file),
            "model": self.model
        }
        
        # Add LLM intelligence status
        intelligence_available = os.path.exists(self.intelligence_cache_file)
        tables_with_intelligence = sum(1 for schema in self.schemas.values() if schema.llm_analysis)
        
        basic_status.update({
            "llm_intelligence_available": intelligence_available,
            "tables_with_intelligence": tables_with_intelligence,
            "intelligence_cache": self.intelligence_cache_file
        })
        
        return basic_status
