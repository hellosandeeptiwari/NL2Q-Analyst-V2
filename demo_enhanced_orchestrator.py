"""
Integration Guide: LLM Schema Intelligence in Main Application
This shows how to integrate the LLM-driven schema intelligence into the 
existing dynamic_agent_orchestrator.py for smarter SQL generation
"""

import os
import json
from typing import Dict, List, Any
from backend.agents.schema_embedder import SchemaEmbedder

class EnhancedDynamicAgentOrchestrator:
    """Enhanced orchestrator that uses LLM schema intelligence for better SQL generation"""
    
    def __init__(self):
        # Initialize with LLM-enhanced schema embedder
        self.schema_embedder = SchemaEmbedder()
        
        # Load pre-analyzed schema intelligence (from indexing)
        self.schema_intelligence = None
        self._load_schema_intelligence()
    
    def _load_schema_intelligence(self):
        """Load pre-analyzed LLM intelligence from vector storage"""
        print("📋 Loading pre-analyzed schema intelligence...")
        
        # This would normally come from Pinecone or your vector DB
        intelligence = self.schema_embedder.load_intelligence_cache()
        
        if intelligence:
            self.schema_intelligence = intelligence
            table_count = len(intelligence.get('table_analyses', {}))
            rel_count = len(intelligence.get('cross_table_intelligence', {}).get('relationships', []))
            print(f"✅ Loaded intelligence for {table_count} tables, {rel_count} relationships")
        else:
            print("⚠️ No cached intelligence found - run schema indexing first")
    
    def generate_sql_with_intelligence(self, user_query: str, relevant_tables: List[str]) -> str:
        """Generate SQL using LLM with intelligent schema context"""
        
        if not self.schema_intelligence:
            print("⚠️ No schema intelligence available - falling back to basic generation")
            return self._generate_basic_sql(user_query, relevant_tables)
        
        print(f"🧠 Generating SQL with LLM intelligence for: {user_query}")
        
        # Step 1: Get intelligent context for relevant tables
        intelligent_context = self._build_intelligent_context(relevant_tables)
        
        # Step 2: Create enhanced system prompt with intelligence
        enhanced_prompt = self._create_intelligent_system_prompt(intelligent_context)
        
        # Step 3: Generate SQL with LLM using intelligent context
        sql_query = self._generate_sql_with_llm(user_query, enhanced_prompt, intelligent_context)
        
        return sql_query
    
    def _build_intelligent_context(self, table_names: List[str]) -> Dict[str, Any]:
        """Build comprehensive intelligent context for SQL generation"""
        context = {
            "tables": {},
            "relationships": [],
            "query_guidance": {},
            "business_rules": []
        }
        
        # Get table-level intelligence
        table_analyses = self.schema_intelligence.get('table_analyses', {})
        for table_name in table_names:
            if table_name in table_analyses:
                analysis = table_analyses[table_name]
                context["tables"][table_name] = {
                    "business_purpose": analysis.get('business_purpose', ''),
                    "domain": analysis.get('domain', ''),
                    "columns": self._get_intelligent_columns(table_name, analysis),
                    "amount_fields": self._get_amount_fields(analysis),
                    "identifier_fields": self._get_identifier_fields(analysis),
                    "forbidden_operations": analysis.get('query_guidance', {}).get('forbidden_operations', [])
                }
        
        # Get relationship intelligence
        cross_table = self.schema_intelligence.get('cross_table_intelligence', {})
        relationships = cross_table.get('relationships', [])
        
        for rel in relationships:
            if rel['from_table'] in table_names or rel['to_table'] in table_names:
                context["relationships"].append({
                    "from_table": rel['from_table'],
                    "to_table": rel['to_table'],
                    "join_column": rel['join_column'],
                    "business_context": rel['business_context']
                })
        
        # Get query patterns
        query_patterns = cross_table.get('query_patterns', {})
        context["query_guidance"] = query_patterns
        
        return context
    
    def _get_intelligent_columns(self, table_name: str, analysis: Dict) -> List[Dict]:
        """Get enhanced column information with semantic intelligence"""
        intelligent_columns = []
        
        for col_insight in analysis.get('column_insights', []):
            col_info = {
                "name": col_insight['column_name'],
                "semantic_role": col_insight['semantic_role'],
                "business_meaning": col_insight['business_meaning'],
                "data_operations": col_insight['data_operations'],
                "is_joinable": col_insight.get('is_joinable', False),
                "aggregation_priority": col_insight.get('aggregation_priority', 5)
            }
            intelligent_columns.append(col_info)
        
        return intelligent_columns
    
    def _get_amount_fields(self, analysis: Dict) -> List[str]:
        """Get fields that can be used for mathematical operations"""
        amount_fields = []
        for col in analysis.get('column_insights', []):
            if 'SUM' in col.get('data_operations', []) or 'AVG' in col.get('data_operations', []):
                amount_fields.append(col['column_name'])
        return amount_fields
    
    def _get_identifier_fields(self, analysis: Dict) -> List[str]:
        """Get identifier fields for grouping and joining"""
        id_fields = []
        for col in analysis.get('column_insights', []):
            if col.get('semantic_role') == 'identifier' or col.get('is_joinable'):
                id_fields.append(col['column_name'])
        return id_fields
    
    def _create_intelligent_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create system prompt enriched with LLM intelligence"""
        
        prompt_parts = [
            "You are an expert SQL analyst with deep understanding of database semantics.",
            "",
            "DATABASE INTELLIGENCE:",
        ]
        
        # Add table intelligence
        for table_name, table_info in context["tables"].items():
            prompt_parts.extend([
                f"",
                f"TABLE: {table_name}",
                f"Purpose: {table_info['business_purpose']}",
                f"Domain: {table_info['domain']}",
                "",
                "COLUMNS WITH SEMANTIC UNDERSTANDING:"
            ])
            
            for col in table_info["columns"]:
                col_desc = f"- {col['name']}: {col['semantic_role']} - {col['business_meaning']}"
                operations = ', '.join(col['data_operations'])
                col_desc += f" [Operations: {operations}]"
                
                if col['aggregation_priority'] <= 2:
                    col_desc += " ⭐ PRIMARY AMOUNT FIELD"
                
                prompt_parts.append(col_desc)
            
            # Add operation guidance
            if table_info["amount_fields"]:
                prompt_parts.append(f"💰 AMOUNT FIELDS (use for SUM/AVG): {', '.join(table_info['amount_fields'])}")
            
            if table_info["identifier_fields"]:
                prompt_parts.append(f"🔑 IDENTIFIER FIELDS (use for JOIN/GROUP BY): {', '.join(table_info['identifier_fields'])}")
            
            if table_info["forbidden_operations"]:
                prompt_parts.append(f"🚫 FORBIDDEN: {', '.join(table_info['forbidden_operations'])}")
        
        # Add relationship intelligence
        if context["relationships"]:
            prompt_parts.extend([
                "",
                "RELATIONSHIP INTELLIGENCE:"
            ])
            for rel in context["relationships"]:
                prompt_parts.append(f"🔗 {rel['from_table']} ↔ {rel['to_table']} via {rel['join_column']}")
                prompt_parts.append(f"   Context: {rel['business_context']}")
        
        # Add query patterns
        query_guidance = context.get("query_guidance", {})
        if query_guidance:
            prompt_parts.extend([
                "",
                "QUERY PATTERN GUIDANCE:"
            ])
            
            payment_pattern = query_guidance.get("payment_analysis", {})
            if payment_pattern:
                prompt_parts.append("💵 For payment/amount analysis:")
                prompt_parts.append(f"   - Primary tables: {payment_pattern.get('primary_tables', [])}")
                prompt_parts.append(f"   - Primary amount column: {payment_pattern.get('primary_amount_column', '')}")
                prompt_parts.append(f"   - Required joins: {payment_pattern.get('required_joins', [])}")
        
        prompt_parts.extend([
            "",
            "SQL GENERATION RULES:",
            "1. Use semantic understanding to select correct columns",
            "2. Only use amount fields for mathematical operations (SUM, AVG)",
            "3. Use identifier fields for grouping and joining", 
            "4. Follow relationship intelligence for proper JOINs",
            "5. Never attempt mathematical operations on text/description fields",
            "6. Prioritize primary amount fields for financial calculations",
            "",
            "Generate precise SQL that respects the semantic intelligence provided."
        ])
        
        return "\\n".join(prompt_parts)
    
    def _generate_sql_with_llm(self, user_query: str, system_prompt: str, context: Dict) -> str:
        """Generate SQL using LLM with intelligent context"""
        
        # This would call your OpenAI API
        user_prompt = f"""
        User Query: {user_query}
        
        Available Tables: {list(context['tables'].keys())}
        
        Generate SQL that uses the semantic intelligence provided in the system prompt.
        Focus on using the correct amount fields and proper relationships.
        """
        
        print("🤖 LLM Prompt Preview:")
        print("System Prompt (truncated):")
        print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
        print("")
        print(f"User Query: {user_query}")
        
        # Simulate LLM response for demo
        if "average" in user_query.lower() and "payment" in user_query.lower():
            # LLM would generate this based on the intelligent context
            suggested_sql = """
            SELECT 
                pr.PROVIDER_NAME,
                AVG(nr.NEGOTIATED_RATE) as avg_payment_amount
            FROM NEGOTIATED_RATES nr
            JOIN PROVIDER_REFERENCES pr ON nr.PROVIDER_ID = pr.PROVIDER_ID
            GROUP BY pr.PROVIDER_NAME
            ORDER BY avg_payment_amount DESC;
            """
            
            print("🎯 LLM-Generated SQL (using intelligence):")
            print("✅ Uses NEGOTIATED_RATE (amount field) for AVG - correct!")
            print("✅ Avoids PAYER (text field) for mathematical operations")
            print("✅ Uses proper JOIN relationship via PROVIDER_ID")
            print("✅ Groups by PROVIDER_NAME (identifier field)")
            
            return suggested_sql.strip()
        
        return "-- SQL would be generated here using LLM with intelligent context"
    
    def _generate_basic_sql(self, user_query: str, relevant_tables: List[str]) -> str:
        """Fallback basic SQL generation without intelligence"""
        return f"-- Basic SQL generation for: {user_query} using tables: {relevant_tables}"

def demo_integration():
    """Demonstrate how the enhanced orchestrator works"""
    
    print("🚀 Enhanced Dynamic Agent Orchestrator Demo")
    print("=" * 60)
    
    # Initialize enhanced orchestrator
    orchestrator = EnhancedDynamicAgentOrchestrator()
    
    # Example user query
    user_query = "What are the average payment amounts by provider?"
    relevant_tables = ["NEGOTIATED_RATES", "PROVIDER_REFERENCES"]
    
    print(f"🔍 User Query: {user_query}")
    print(f"📊 Relevant Tables: {relevant_tables}")
    print("")
    
    # Generate SQL with intelligence
    sql_result = orchestrator.generate_sql_with_intelligence(user_query, relevant_tables)
    
    print("🎉 Final SQL:")
    print(sql_result)
    
    print("\\n💡 Integration Benefits:")
    print("✅ LLM uses pre-analyzed schema intelligence")
    print("✅ No real-time schema analysis needed")
    print("✅ Prevents common SQL generation errors")
    print("✅ Understands business context and relationships")
    print("✅ Fast query-time execution")

if __name__ == "__main__":
    demo_integration()
