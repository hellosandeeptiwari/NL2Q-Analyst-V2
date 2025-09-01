"""
Agent Orchestrator - Coordinates multiple agents for intelligent task execution
Now using OpenAI embeddings for superior semantic understanding
"""
from typing import Dict, List, Any, Optional
from .openai_vector_matcher import OpenAIVectorMatcher
from .llm_agent import LLMAgent
import time
import json
import os

class AgentOrchestrator:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if self.openai_api_key:
            self.vector_matcher = OpenAIVectorMatcher(self.openai_api_key)
            self.llm_agent = LLMAgent(self.openai_api_key)
            print("🤖 OpenAI-powered agents initialized")
        else:
            print("⚠️ No OpenAI API key found - running in limited mode")
            self.vector_matcher = None
            self.llm_agent = None
        
        self.table_names = []
        self.is_initialized = False
        
    def initialize(self, adapter, force_rebuild: bool = False):
        """Initialize the agent system with database adapter for schema analysis"""
        if not self.vector_matcher:
            print("⚠️ Vector matcher not available - skipping initialization")
            return
            
        print("🚀 Initializing Agent Orchestrator with OpenAI embeddings...")
        start_time = time.time()
        
        # Initialize vector embeddings from database schema
        self.vector_matcher.initialize_from_database(adapter, force_rebuild)
        
        # Get table names for compatibility
        self.table_names = list(self.vector_matcher.table_embeddings.keys())
        
        self.is_initialized = True
        init_time = time.time() - start_time
        print(f"✅ Agent Orchestrator initialized in {init_time:.2f}s with {len(self.table_names)} tables")
        
    def intelligent_table_suggestion(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Provide intelligent table suggestions using OpenAI vector embeddings and LLM analysis
        """
        if not self.is_initialized or not self.vector_matcher:
            return {"error": "Agent system not initialized or not available"}
            
        start_time = time.time()
        
        # Step 1: Get semantic matches using OpenAI embeddings
        print("🔍 Running OpenAI vector similarity search...")
        vector_results = self.vector_matcher.hybrid_search(query, top_k * 2)
        
        # Step 2: Analyze query intent with LLM (if available)
        intent_analysis = {}
        if self.llm_agent:
            print("🧠 Analyzing query intent with LLM...")
            intent_analysis = self.llm_agent.analyze_query_intent(query, self.table_names)
        
        # Step 3: Combine results intelligently
        combined_suggestions = self._combine_vector_suggestions(
            vector_results, intent_analysis, query
        )
        
        # Step 4: Get query execution plan (if LLM available)
        execution_plan = {}
        if self.llm_agent and combined_suggestions:
            print("📋 Planning query execution strategy...")
            execution_plan = self.llm_agent.plan_query_strategy(query, combined_suggestions[:3])
        
        processing_time = time.time() - start_time
        
        result = {
            "query": query,
            "processing_time_ms": round(processing_time * 1000, 2),
            "suggested_tables": combined_suggestions[:top_k],
            "vector_analysis": vector_results,
            "intent_analysis": intent_analysis,
            "execution_plan": execution_plan,
            "metadata": {
                "total_tables_searched": len(self.table_names),
                "vector_matches_found": len(vector_results.get('similar_tables', [])),
                "agents_used": ["openai_vector_matcher", "llm_agent"] if self.llm_agent else ["openai_vector_matcher"]
            }
        }
        
        return result
    
    def generate_intelligent_insights(self, query: str, data_analysis: Dict[str, Any], 
                                    table_name: str) -> str:
        """Generate sophisticated insights using LLM agent"""
        print("💡 Generating intelligent insights...")
        
        # Prepare data summary for LLM
        data_summary = {
            "total_rows": len(data_analysis.get("rows", [])),
            "columns": data_analysis.get("columns", []),
            "sample_data": data_analysis.get("rows", [])[:5],
            "table_name": table_name,
            "statistical_analysis": self._perform_statistical_analysis(data_analysis)
        }
        
        # Generate insights using LLM
        insights = self.llm_agent.generate_insights(query, data_summary, table_name)
        
        return insights
    
    def _combine_vector_suggestions(self, vector_results: Dict[str, Any], 
                                   intent_analysis: Dict[str, Any], query: str) -> List[Dict]:
        """Intelligently combine OpenAI vector results with LLM analysis"""
        
        # Create a scoring system
        table_scores = {}
        
        # Score from vector similarity
        similar_tables = vector_results.get('similar_tables', [])
        for match in similar_tables:
            table_name = match['table_name']
            score = match['similarity_score']
            confidence_bonus = {
                'very_high': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0.0
            }.get(match['confidence'], 0.0)
            
            table_scores[table_name] = {
                'total_score': score + confidence_bonus,
                'vector_score': score,
                'confidence': match['confidence'],
                'reasons': [f"Vector similarity: {score:.3f}"],
                'sources': ['openai_embeddings'],
                'relevant_columns': []
            }
        
        # Add relevant columns information
        table_columns = vector_results.get('table_specific_columns', {})
        for table_name, columns in table_columns.items():
            if table_name in table_scores and columns:
                table_scores[table_name]['relevant_columns'] = [
                    f"{col['column_name']} ({col['similarity_score']:.2f})" 
                    for col in columns[:3]
                ]
                table_scores[table_name]['reasons'].append(
                    f"Relevant columns found: {len(columns)}"
                )
        
        # Boost tables recommended by LLM
        llm_recommendations = intent_analysis.get('recommended_tables', [])
        for rec in llm_recommendations:
            table_name = rec.get('table', '')
            if table_name in table_scores:
                # Boost existing score
                llm_bonus = {'high': 0.4, 'medium': 0.3, 'low': 0.2}.get(rec.get('confidence', 'low'), 0.2)
                table_scores[table_name]['total_score'] += llm_bonus
                table_scores[table_name]['reasons'].append(f"LLM recommendation: {rec.get('reason', 'No reason')}")
                table_scores[table_name]['sources'].append('llm')
            else:
                # Add new table from LLM
                table_scores[table_name] = {
                    'total_score': {'high': 0.7, 'medium': 0.5, 'low': 0.3}.get(rec.get('confidence', 'low'), 0.3),
                    'vector_score': 0.0,
                    'confidence': rec.get('confidence', 'medium'),
                    'reasons': [f"LLM recommendation: {rec.get('reason', 'No reason')}"],
                    'sources': ['llm'],
                    'relevant_columns': []
                }
        
        # Apply domain-specific bonuses
        query_lower = query.lower()
        if 'nba' in query_lower:
            for table_name in table_scores:
                if 'nba' in table_name.lower():
                    table_scores[table_name]['total_score'] += 0.5
                    table_scores[table_name]['reasons'].append("NBA domain match bonus")
        
        # Convert to sorted list
        suggestions = []
        for table_name, data in table_scores.items():
            suggestions.append({
                'table_name': table_name,
                'total_score': data['total_score'],
                'vector_score': data['vector_score'],
                'confidence': data['confidence'],
                'reasons': data['reasons'],
                'sources': data['sources'],
                'relevant_columns': data['relevant_columns'],
                'match_type': 'hybrid_vector_llm'
            })
        
        # Sort by total score
        suggestions.sort(key=lambda x: x['total_score'], reverse=True)
        
        return suggestions
    
    def _perform_statistical_analysis(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic statistical analysis on the data"""
        rows = data_analysis.get("rows", [])
        columns = data_analysis.get("columns", [])
        
        if not rows or not columns:
            return {}
        
        analysis = {
            "row_count": len(rows),
            "column_count": len(columns),
            "sample_size": min(len(rows), 10)
        }
        
        # Analyze column types and sample values
        if rows:
            first_row = rows[0]
            column_analysis = {}
            
            for i, col in enumerate(columns):
                if i < len(first_row):
                    value = first_row[i]
                    value_type = type(value).__name__
                    column_analysis[col] = {
                        "type": value_type,
                        "sample_value": str(value)[:50] if value is not None else "NULL"
                    }
            
            analysis["columns_detail"] = column_analysis
        
        return analysis
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current status of the agent system"""
        # Check if vector matcher is ready
        vector_ready = self.vector_matcher and bool(self.vector_matcher.table_embeddings)
        llm_ready = self.llm_agent and self.llm_agent.api_key is not None
        agents_ready = vector_ready and llm_ready
        
        status = {
            "initialized": self.is_initialized,
            "connected": self.is_initialized,
            "total_tables": len(self.table_names),
            "openai_api_available": self.openai_api_key is not None,
            
            # Frontend expected format
            "database": {
                "connected": self.is_initialized,
                "status": "connected" if self.is_initialized else "disconnected"
            },
            "agents": {
                "ready": agents_ready,
                "llm_connected": llm_ready
            },
            "tables": {
                "count": len(self.table_names)
            },
            
            # Additional detailed info
            "agent_details": {}
        }
        
        if self.vector_matcher:
            status["vector_matcher"] = self.vector_matcher.get_status()
            status["agent_details"]["openai_vector_matcher"] = "ready" if self.vector_matcher.table_embeddings else "not_ready"
            if hasattr(self.vector_matcher, 'table_embeddings'):
                status["openai_vector_ready"] = "Available" if self.vector_matcher.table_embeddings else "Not Available"
        else:
            status["agent_details"]["openai_vector_matcher"] = "not_available"
            
        if self.llm_agent:
            status["llm_agent_ready"] = self.llm_agent.api_key is not None
            status["context_memory_size"] = len(self.llm_agent.context_memory)
            status["agent_details"]["llm_agent"] = "ready" if self.llm_agent.api_key else "fallback_mode"
        else:
            status["agent_details"]["llm_agent"] = "not_available"
            
        return status
