"""
LLM-Driven Schema Intelligence for Vector Database Storage
This system uses LLM to analyze schema and stores findings in vector DB during indexing
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from openai import AsyncOpenAI

@dataclass
class SchemaAnalysisResult:
    table_name: str
    business_purpose: str
    domain_classification: str
    column_insights: List[Dict[str, Any]]
    relationship_potential: List[Dict[str, Any]]
    query_guidance: Dict[str, Any]
    llm_confidence: float

class LLMSchemaIntelligence:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ No OpenAI API key found - LLM analysis will be disabled")
            self.client = None
            self.model = None
            return
            
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    async def analyze_table_with_llm(self, table_name: str, columns_info: List[Dict], 
                                   sample_data: Dict[str, List] = None) -> SchemaAnalysisResult:
        """Let LLM analyze table structure and business meaning"""
        
        if not self.client:
            print(f"⚠️ No OpenAI client - using fallback analysis for {table_name}")
            return self._create_fallback_analysis(table_name, columns_info)
        
        # Prepare column information for LLM
        columns_text = self._format_columns_for_llm(columns_info)
        sample_text = self._format_sample_data_for_llm(sample_data) if sample_data else ""
        
        analysis_prompt = f"""You are a database schema analyst. Analyze this table and provide business intelligence insights.

DATABASE CONTEXT:
- Database: Healthcare Pricing Analytics
- Table: {table_name}

COLUMN STRUCTURE:
{columns_text}

{sample_text}

ANALYSIS INSTRUCTIONS:
1. Determine the business purpose of this table
2. Classify each column's role (identifier, amount, description, metric, etc.)
3. Identify which columns can be used for mathematical operations (SUM, AVG)
4. Identify which columns are likely foreign keys to other tables
5. Determine the table's domain (provider, financial, reference, etc.)
6. Provide query guidance for this table

Respond with a JSON object containing:
{{
    "business_purpose": "detailed business purpose of this table",
    "domain_classification": "provider|financial|service|reference|metric|volume",
    "column_insights": [
        {{
            "column_name": "COLUMN_NAME",
            "semantic_role": "identifier|amount|description|metric|code|reference",
            "data_operations": ["SUM", "AVG", "COUNT"] or ["COUNT", "DISTINCT"] or ["GROUP_BY"],
            "is_joinable": true/false,
            "business_meaning": "what this column represents in business terms",
            "likely_foreign_key_to": "table_name" or null,
            "aggregation_priority": 1-5 (for amount fields, 1=highest priority)
        }}
    ],
    "relationship_potential": [
        {{
            "column": "COLUMN_NAME",
            "likely_related_tables": ["table1", "table2"],
            "relationship_type": "primary_key|foreign_key|lookup",
            "confidence": 0.0-1.0
        }}
    ],
    "query_guidance": {{
        "primary_amount_fields": ["column1", "column2"],
        "key_identifiers": ["id_column1", "id_column2"],
        "forbidden_operations": ["operation description"],
        "recommended_joins": ["join pattern description"],
        "typical_use_cases": ["use case 1", "use case 2"]
    }},
    "confidence": 0.0-1.0
}}

Provide only the JSON response."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert database schema analyst. Respond only with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON if wrapped in markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            analysis_data = json.loads(content)
            
            return SchemaAnalysisResult(
                table_name=table_name,
                business_purpose=analysis_data.get("business_purpose", ""),
                domain_classification=analysis_data.get("domain_classification", "unknown"),
                column_insights=analysis_data.get("column_insights", []),
                relationship_potential=analysis_data.get("relationship_potential", []),
                query_guidance=analysis_data.get("query_guidance", {}),
                llm_confidence=analysis_data.get("confidence", 0.5)
            )
            
        except Exception as e:
            print(f"❌ LLM analysis failed for {table_name}: {e}")
            return self._create_fallback_analysis(table_name, columns_info)
    
    async def analyze_cross_table_relationships(self, all_tables_analysis: List[SchemaAnalysisResult]) -> Dict[str, Any]:
        """Let LLM analyze relationships across all tables"""
        
        if not self.client:
            print("⚠️ No OpenAI client - using fallback relationship analysis")
            return {"relationships": [], "query_patterns": {}, "business_rules": []}
        
        # Prepare table summaries for LLM
        tables_summary = []
        for analysis in all_tables_analysis:
            tables_summary.append({
                "table_name": analysis.table_name,
                "domain": analysis.domain_classification,
                "purpose": analysis.business_purpose,
                "key_columns": [col["column_name"] for col in analysis.column_insights 
                              if col.get("is_joinable", False)],
                "amount_columns": [col["column_name"] for col in analysis.column_insights 
                                 if "SUM" in col.get("data_operations", []) or "AVG" in col.get("data_operations", [])]
            })
        
        relationship_prompt = f"""Analyze relationships between these database tables and provide comprehensive query guidance.

TABLES OVERVIEW:
{json.dumps(tables_summary, indent=2)}

ANALYSIS INSTRUCTIONS:
1. Identify primary-foreign key relationships between tables
2. Determine the best tables and columns for common business queries
3. Provide specific guidance for payment analysis queries
4. Identify data flow patterns and table hierarchies

Respond with JSON:
{{
    "relationships": [
        {{
            "from_table": "table1",
            "to_table": "table2", 
            "join_column": "shared_column",
            "relationship_type": "foreign_key|lookup|bridge",
            "business_context": "why these tables connect",
            "confidence": 0.0-1.0
        }}
    ],
    "query_patterns": {{
        "payment_analysis": {{
            "primary_tables": ["table1", "table2"],
            "primary_amount_column": "table.column",
            "required_joins": ["join pattern"],
            "forbidden_operations": ["what not to do"]
        }},
        "provider_analysis": {{
            "primary_tables": ["table1"],
            "key_identifiers": ["column1", "column2"],
            "typical_joins": ["join pattern"]
        }}
    }},
    "business_rules": [
        {{
            "rule": "rule_name", 
            "description": "business rule description",
            "enforcement": "how to enforce in SQL"
        }}
    ],
    "data_quality_insights": {{
        "primary_entities": ["entity1", "entity2"],
        "fact_tables": ["table1", "table2"],
        "dimension_tables": ["table1", "table2"],
        "recommended_indexes": ["table.column"]
    }}
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database architecture expert. Respond only with valid JSON."},
                    {"role": "user", "content": relationship_prompt}
                ],
                temperature=0.1,
                max_tokens=2500
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
            
        except Exception as e:
            print(f"❌ Cross-table analysis failed: {e}")
            return {"relationships": [], "query_patterns": {}, "business_rules": []}
    
    def _format_columns_for_llm(self, columns_info: List[Dict]) -> str:
        """Format column information for LLM analysis"""
        formatted = []
        for col in columns_info:
            col_text = f"- {col['name']} ({col['data_type']})"
            if col.get('nullable'):
                col_text += " [NULLABLE]"
            formatted.append(col_text)
        return "\n".join(formatted)
    
    def _format_sample_data_for_llm(self, sample_data: Dict[str, List]) -> str:
        """Format sample data for LLM context"""
        if not sample_data:
            return ""
        
        sample_text = "SAMPLE DATA:\n"
        for col_name, values in sample_data.items():
            if values:
                sample_text += f"- {col_name}: {values[:3]}\n"
        return sample_text
    
    def _create_fallback_analysis(self, table_name: str, columns_info: List[Dict]) -> SchemaAnalysisResult:
        """Create basic analysis when LLM fails"""
        return SchemaAnalysisResult(
            table_name=table_name,
            business_purpose=f"Data table: {table_name.lower().replace('_', ' ')}",
            domain_classification="unknown",
            column_insights=[
                {
                    "column_name": col["name"],
                    "semantic_role": "unknown",
                    "data_operations": ["COUNT"],
                    "is_joinable": False,
                    "business_meaning": f"Data field: {col['name']}",
                    "likely_foreign_key_to": None,
                    "aggregation_priority": 5
                }
                for col in columns_info
            ],
            relationship_potential=[],
            query_guidance={},
            llm_confidence=0.1
        )

class VectorSchemaStorage:
    """Store LLM schema analysis in vector database during indexing"""
    
    def __init__(self):
        self.llm_analyzer = LLMSchemaIntelligence()
    
    async def enhanced_schema_indexing(self, tables_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced schema indexing with LLM intelligence stored in vector DB"""
        
        print("🧠 Starting LLM-driven schema analysis...")
        
        # Step 1: Analyze each table with LLM
        table_analyses = []
        for table_name, table_info in tables_metadata.items():
            print(f"🔍 LLM analyzing table: {table_name}")
            
            columns_info = table_info.get("columns", [])
            sample_data = table_info.get("sample_data", {})
            
            analysis = await self.llm_analyzer.analyze_table_with_llm(
                table_name, columns_info, sample_data
            )
            table_analyses.append(analysis)
        
        # Step 2: Cross-table relationship analysis
        print("🔗 LLM analyzing cross-table relationships...")
        cross_table_analysis = await self.llm_analyzer.analyze_cross_table_relationships(table_analyses)
        
        # Step 3: Generate comprehensive schema intelligence
        schema_intelligence = {
            "table_analyses": {analysis.table_name: {
                "business_purpose": analysis.business_purpose,
                "domain": analysis.domain_classification,
                "column_insights": analysis.column_insights,
                "relationship_potential": analysis.relationship_potential,
                "query_guidance": analysis.query_guidance,
                "confidence": analysis.llm_confidence
            } for analysis in table_analyses},
            "cross_table_intelligence": cross_table_analysis,
            "generated_timestamp": "2025-09-10",
            "llm_model": self.llm_analyzer.model
        }
        
        # Step 4: Create enhanced vector embeddings with intelligence
        enhanced_embeddings = await self._create_intelligent_embeddings(schema_intelligence)
        
        print("✅ LLM schema analysis completed and ready for vector storage")
        return {
            "schema_intelligence": schema_intelligence,
            "enhanced_embeddings": enhanced_embeddings,
            "indexing_metadata": {
                "tables_analyzed": len(table_analyses),
                "relationships_found": len(cross_table_analysis.get("relationships", [])),
                "query_patterns_identified": len(cross_table_analysis.get("query_patterns", {}))
            }
        }
    
    async def _create_intelligent_embeddings(self, schema_intelligence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create embeddings enriched with LLM intelligence"""
        embeddings = []
        
        for table_name, analysis in schema_intelligence["table_analyses"].items():
            # Create table-level embedding with business context
            table_embedding = {
                "table": table_name,
                "type": "table_analysis",
                "content": f"""
                Table: {table_name}
                Business Purpose: {analysis['business_purpose']}
                Domain: {analysis['domain']}
                Query Guidance: {json.dumps(analysis['query_guidance'])}
                """,
                "metadata": {
                    "business_purpose": analysis['business_purpose'],
                    "domain": analysis['domain'],
                    "query_guidance": analysis['query_guidance'],
                    "confidence": analysis['confidence']
                }
            }
            embeddings.append(table_embedding)
            
            # Create column-level embeddings with semantic understanding
            for col_insight in analysis['column_insights']:
                col_embedding = {
                    "table": table_name,
                    "column": col_insight['column_name'],
                    "type": "column_analysis", 
                    "content": f"""
                    Column: {table_name}.{col_insight['column_name']}
                    Semantic Role: {col_insight['semantic_role']}
                    Business Meaning: {col_insight['business_meaning']}
                    Operations: {col_insight['data_operations']}
                    Joinable: {col_insight['is_joinable']}
                    """,
                    "metadata": {
                        "semantic_role": col_insight['semantic_role'],
                        "data_operations": col_insight['data_operations'],
                        "is_joinable": col_insight['is_joinable'],
                        "business_meaning": col_insight['business_meaning'],
                        "likely_foreign_key_to": col_insight.get('likely_foreign_key_to'),
                        "aggregation_priority": col_insight.get('aggregation_priority', 5)
                    }
                }
                embeddings.append(col_embedding)
        
        # Add relationship embeddings
        relationships = schema_intelligence["cross_table_intelligence"].get("relationships", [])
        for rel in relationships:
            rel_embedding = {
                "type": "relationship_analysis",
                "content": f"""
                Relationship: {rel['from_table']} -> {rel['to_table']} via {rel['join_column']}
                Type: {rel['relationship_type']}
                Business Context: {rel['business_context']}
                """,
                "metadata": {
                    "from_table": rel['from_table'],
                    "to_table": rel['to_table'],
                    "join_column": rel['join_column'],
                    "relationship_type": rel['relationship_type'],
                    "business_context": rel['business_context'],
                    "confidence": rel['confidence']
                }
            }
            embeddings.append(rel_embedding)
        
        return embeddings

# Integration example for the main indexing process
async def integrate_with_pinecone_indexing():
    """Integration example showing how to use this in the main indexing flow"""
    
    print("""
    📋 INTEGRATION PLAN:
    
    1. During Schema Indexing (in pinecone_schema_vector_store.py):
       - Get table metadata from Snowflake
       - Run LLM analysis with VectorSchemaStorage 
       - Store enhanced embeddings in Pinecone with intelligence metadata
    
    2. During Query Time (in dynamic_agent_orchestrator.py):
       - Retrieve pre-analyzed schema intelligence from Pinecone
       - Use LLM insights for smarter SQL generation
       - No real-time analysis needed - just lookup!
    
    3. Benefits:
       - ✅ LLM does all the intelligent analysis (no hardcoding)
       - ✅ Analysis happens once during indexing (not per query)
       - ✅ Fast query-time retrieval from vector DB
       - ✅ Rich business context for SQL generation
    """)

if __name__ == "__main__":
    asyncio.run(integrate_with_pinecone_indexing())
