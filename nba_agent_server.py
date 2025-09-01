#!/usr/bin/env python3
"""
Enhanced NBA HTTP server with LLM Agents and FAISS-based similarity search
Features:
- Intelligent table matching using FAISS semantic embeddings
- LLM-powered query analysis and insights generation
- Multi-agent orchestration for complex tasks
- Context-aware memory system
"""
import sys
from pathlib import Path
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.parse
from datetime import datetime
from typing import Dict, List, Any
import difflib
import re
import pandas as pd
import numpy as np
import traceback
import os

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv()

# Initialize database
from db.engine import get_adapter
adapter = get_adapter()

# Initialize Agent Orchestrator with optimized settings
from agents.orchestrator import AgentOrchestrator
agent_orchestrator = AgentOrchestrator(openai_api_key=os.getenv('OPENAI_API_KEY'))

# Initialize optimized schema embedder
from agents.schema_embedder import SchemaEmbedder
schema_embedder = SchemaEmbedder(
    api_key=os.getenv('OPENAI_API_KEY'),
    batch_size=25,  # Optimal batch size
    max_workers=3   # Conservative threading
)

# Initialize agents with database adapter
def initialize_agents():
    """Initialize the agent system with database adapter for schema analysis"""

# Conversation context storage
conversation_contexts = {}

class ConversationContext:
    """Manages conversation context for continuous chat functionality"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.queries = []
        self.last_table = None
        self.last_columns = []
        self.last_analysis_type = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_query(self, query: str, table_name: str = None, columns: List[str] = None, analysis_type: str = None):
        """Add a query to the conversation history"""
        self.queries.append({
            'query': query,
            'table_name': table_name,
            'columns': columns,
            'analysis_type': analysis_type,
            'timestamp': datetime.now()
        })
        
        # Update context
        if table_name:
            self.last_table = table_name
        if columns:
            self.last_columns = columns
        if analysis_type:
            self.last_analysis_type = analysis_type
            
        self.updated_at = datetime.now()
        
        # Keep only last 10 queries to manage memory
        if len(self.queries) > 10:
            self.queries = self.queries[-10:]
    
    def get_context_summary(self) -> str:
        """Generate a context summary for LLM"""
        if not self.queries:
            return "No previous conversation context."
        
        context = "Previous conversation context:\n"
        for i, q in enumerate(self.queries[-3:], 1):  # Last 3 queries
            context += f"{i}. Query: '{q['query']}'"
            if q['table_name']:
                context += f" (Table: {q['table_name']})"
            if q['columns']:
                context += f" (Columns: {', '.join(q['columns'])})"
            context += "\n"
        
        if self.last_table:
            context += f"Current working table: {self.last_table}\n"
        if self.last_columns:
            context += f"Recent columns: {', '.join(self.last_columns)}\n"
            
        return context
    
    def get_continuation_suggestions(self) -> List[str]:
        """Get suggestions for follow-up queries"""
        suggestions = []
        
        if self.last_table and self.last_columns:
            suggestions.extend([
                f"Show more details about {self.last_table}",
                f"Compare {self.last_columns[0]} with other columns" if self.last_columns else "",
                f"Filter {self.last_table} by specific criteria",
                "Show top 10 records",
                "Create a different visualization"
            ])
        
        return [s for s in suggestions if s]  # Remove empty strings

def get_or_create_context(session_id: str) -> ConversationContext:
    """Get existing context or create new one"""
    if session_id not in conversation_contexts:
        conversation_contexts[session_id] = ConversationContext(session_id)
    return conversation_contexts[session_id]
    try:
        if agent_orchestrator.vector_matcher:
            print("🚀 Initializing OpenAI vector embeddings...")
            agent_orchestrator.initialize(adapter)
            print("🤖 Agent Orchestrator initialized with OpenAI embeddings")
        else:
            print("⚠️ OpenAI API key not available, agents running in limited mode")
    except Exception as e:
        print(f"⚠️ Agent initialization failed: {e}")
        print("🔄 Running without agent enhancements")

def get_available_tables():
    """Get all available tables from the database"""
    try:
        result = adapter.run("SHOW TABLES")
        if result.error:
            print(f"❌ Error getting tables: {result.error}")
            return []
        tables = [row[1] for row in result.rows]  # Table name is usually in second column
        print(f"📋 Found {len(tables)} tables")
        return tables
    except Exception as e:
        print(f"❌ Exception getting tables: {e}")
        return []

def initialize_agents():
    """Initialize the agent system with available tables"""
    try:
        tables = get_available_tables()
        if tables:
            agent_orchestrator.initialize(tables)
            print("🤖 Agent Orchestrator initialized successfully")
        else:
            print("⚠️ No tables found, agents running in limited mode")
    except Exception as e:
        print(f"⚠️ Agent initialization failed: {e}")
        print("🔄 Running without agent enhancements")

def intelligent_table_suggestion(query_text, available_tables, max_suggestions=5):
    """
    Use Agent Orchestrator for intelligent table suggestions
    """
    try:
        if agent_orchestrator.is_initialized:
            print("🤖 Using intelligent agent-based table suggestion...")
            suggestion_result = agent_orchestrator.intelligent_table_suggestion(
                query_text, max_suggestions
            )
            return suggestion_result
        else:
            print("⚠️ Agents not initialized, using basic suggestion")
            return {"suggested_tables": [], "agent_enhanced": False}
    except Exception as e:
        print(f"⚠️ Agent suggestion failed: {e}")
        return {"suggested_tables": [], "agent_enhanced": False}

class NBAHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        print(f"🔍 Received GET request to: {self.path}")
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            health = adapter.health()
            self._send_json_response(health)
        
        elif parsed_path.path == '/test-nba':
            result = adapter.run('SELECT COUNT(*) FROM "Final_NBA_Output_python_20250519"')
            if result.error:
                self._send_json_response({"error": result.error}, 500)
            else:
                self._send_json_response({
                    "table": "Final_NBA_Output_python_20250519",
                    "total_records": result.rows[0][0],
                    "status": "accessible"
                })
        
        elif parsed_path.path == '/tables':
            tables = get_available_tables()
            self._send_json_response({
                "tables": tables,
                "count": len(tables)
            })
        
        elif parsed_path.path == '/agent-status':
            status = agent_orchestrator.get_system_status()
            status['agent_endpoints'] = {
                'intelligent_suggestions': '/query (enhanced)',
                'llm_insights': '/insights (enhanced)',
                'system_status': '/agent-status'
            }
            self._send_json_response(status)
        
        else:
            self._send_json_response({"error": "Not found"}, 404)
    
    def _get_session_context(self, session_id):
        """Get or create conversation context for a session"""
        return get_or_create_context(session_id)
    
    def do_POST(self):
        print(f"🔍 Received POST request to: {self.path}")
        
        if self.path == '/query':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                print(f"📝 Request data: {request_data}")
                natural_language = request_data.get('natural_language', '')
                job_id = request_data.get('job_id', 'default')
                selected_tables = request_data.get('selected_tables', [])
                session_id = request_data.get('session_id', job_id)  # Use session for context
                
                # Get conversation context
                context = get_or_create_context(session_id)
                context_summary = context.get_context_summary()
                
                print(f"🔍 Processing query: {natural_language}")
                print(f"💭 Context: {context_summary}")
                
                # Enhance query with context for better understanding
                enhanced_query = natural_language
                if context.last_table and not any(word in natural_language.lower() for word in ['table', 'from']):
                    # If user doesn't specify table but we have context, add it
                    if any(word in natural_language.lower() for word in ['more', 'also', 'and', 'additionally', 'further']):
                        enhanced_query = f"{natural_language} from {context.last_table}"
                        print(f"🔄 Enhanced query with context: {enhanced_query}")
                
                # Get available tables
                available_tables = get_available_tables()
                
                # Check if user provided selected tables
                if selected_tables:
                    print(f"👤 User selected tables: {selected_tables}")
                    table_name = selected_tables[0]
                    context.add_query(natural_language, table_name, analysis_type="user_selected")
                    self._process_query_with_table(enhanced_query, job_id, table_name, context)
                elif context.last_table and any(word in natural_language.lower() for word in ['more', 'also', 'continue', 'further', 'next', 'another']):
                    # User wants to continue with the same table
                    print(f"🔄 Continuing with previous table: {context.last_table}")
                    context.add_query(natural_language, context.last_table, analysis_type="continuation")
                    self._process_query_with_table(enhanced_query, job_id, context.last_table, context)
                else:
                    # Use intelligent suggestions with automatic execution
                    suggestion_result = intelligent_table_suggestion(enhanced_query, available_tables)
                    
                    if suggestion_result.get('suggested_tables'):
                        suggested_tables = suggestion_result['suggested_tables']
                        
                        # Check confidence level - auto-execute if high confidence
                        best_suggestion = suggested_tables[0] if suggested_tables else None
                        
                        if best_suggestion:
                            # Extract table name properly
                            if isinstance(best_suggestion, dict):
                                table_name = best_suggestion.get('table_name', '')
                                confidence_val = best_suggestion.get('confidence', 0.0)
                                # Ensure confidence is a float
                                try:
                                    confidence = float(confidence_val) if confidence_val is not None else 0.0
                                except (ValueError, TypeError):
                                    confidence = 0.0
                            else:
                                table_name = str(best_suggestion)
                                confidence = 0.7  # Default moderate confidence
                            
                            # Auto-execute if confidence is high enough (>= 80%)
                            if confidence >= 0.8 and table_name:
                                print(f"🚀 Auto-executing with high confidence table: {table_name} ({confidence:.1%})")
                                context.add_query(natural_language, table_name, analysis_type="auto_high_confidence")
                                self._process_query_with_table(enhanced_query, job_id, table_name, context)
                                return
                            # Medium confidence - show suggestions but auto-execute with best
                            elif confidence >= 0.6 and table_name:
                                print(f"🔄 Medium confidence - executing with notification: {table_name} ({confidence:.1%})")
                                context.add_query(natural_language, table_name, analysis_type="auto_medium_confidence")
                                # Execute but also inform user of other options
                                self._process_query_with_table_and_alternatives(enhanced_query, job_id, table_name, suggested_tables, context)
                                return
                        
                        # Only ask for selection if confidence is low (<60%) or ambiguous
                        formatted_suggestions = []
                        for suggestion in suggested_tables:
                            if isinstance(suggestion, dict):
                                formatted_suggestions.append(suggestion.get('table_name', ''))
                            else:
                                formatted_suggestions.append(str(suggestion))
                        
                        response = {
                            "job_id": job_id,
                            "status": "needs_table_selection",
                            "message": "Multiple relevant tables found. Please confirm your choice:",
                            "suggested_tables": formatted_suggestions,
                            "intent_analysis": suggestion_result.get('intent_analysis', {}),
                            "execution_plan": suggestion_result.get('execution_plan', {}),
                            "all_tables": available_tables[:20],
                            "query": natural_language,
                            "agent_enhanced": True
                        }
                        self._send_json_response(response)
                    else:
                        # No suggestions - show all tables
                        response = {
                            "job_id": job_id,
                            "status": "needs_table_selection", 
                            "message": "Please select from available tables:",
                            "suggested_tables": available_tables[:10],
                            "all_tables": available_tables[:20],
                            "query": natural_language,
                            "agent_enhanced": False
                        }
                        self._send_json_response(response)
                    
            except Exception as e:
                print(f"🚨 Error processing query: {str(e)}")
                self._send_json_response({"error": f"Request processing failed: {str(e)}"}, 400)
        
        elif self.path == '/query-with-table':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                natural_language = request_data.get('natural_language', '')
                job_id = request_data.get('job_id', 'default')
                table_name = request_data.get('table_name', '')
                selected_tables = request_data.get('selected_tables', [])
                
                # Support both table_name (single) and selected_tables (multiple)
                if table_name:
                    self._process_query_with_table(natural_language, job_id, table_name)
                elif selected_tables and len(selected_tables) > 0:
                    # Use the first selected table for now
                    primary_table = selected_tables[0]
                    self._process_query_with_table(natural_language, job_id, primary_table)
                else:
                    self._send_json_response({"error": "Table name or selected tables required"}, 400)
                    
            except Exception as e:
                print(f"🚨 Error processing query with table: {str(e)}")
                self._send_json_response({"error": f"Request processing failed: {str(e)}"}, 400)
        
        elif self.path == '/execute-choice':
            # New endpoint for executing user's choice from suggestions
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                natural_language = request_data.get('natural_language', '')
                job_id = request_data.get('job_id', 'default')
                table_name = request_data.get('table_name', '')
                chosen_column = request_data.get('chosen_column', '')
                
                print(f"🎯 Executing user choice: {chosen_column} for query: {natural_language}")
                
                # Force execute with the user's chosen column
                self._execute_analysis_with_column(natural_language, job_id, table_name, chosen_column)
                
            except Exception as e:
                print(f"🚨 Error executing choice: {str(e)}")
                self._send_json_response({"error": f"Choice execution failed: {str(e)}"}, 400)
        
        elif self.path == '/insights':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                query = request_data.get('query', '')
                data_rows = request_data.get('data_rows', [])
                columns = request_data.get('columns', [])
                table_name = request_data.get('table_name', '')
                
                insights = self._generate_insights(query, data_rows=data_rows, 
                                                columns=columns, table_name=table_name)
                
                self._send_json_response({"insight": insights})
                
            except Exception as e:
                print(f"🚨 Error generating insights: {str(e)}")
                self._send_json_response({"error": f"Insight generation failed: {str(e)}"}, 400)
        
        else:
            self._send_json_response({"error": "Endpoint not found"}, 404)
    
    def _process_query_with_table_and_alternatives(self, natural_language, job_id, table_name, alternative_tables, context=None):
        """Process query with chosen table but show alternatives for medium confidence"""
        try:
            print(f"🔄 Processing with medium confidence: {table_name}")
            
            # For now, just execute normally - we could enhance this later to include alternatives in response
            self._process_query_with_table(natural_language, job_id, table_name, context)
            
        except Exception as e:
            print(f"🚨 Error in medium confidence processing: {str(e)}")
            # Fallback to normal processing
            self._process_query_with_table(natural_language, job_id, table_name, context)
    
    def _process_query_with_table(self, natural_language, job_id, table_name, context=None):
        """Process query with specific table"""
        try:
            # Handle frequency analysis
            if 'frequency' in natural_language.lower():
                self._handle_frequency_analysis(natural_language, job_id, table_name, context)
            else:
                # Default query - show sample data
                sql = f'SELECT * FROM "{table_name}" LIMIT 10'
                
                result = adapter.run(sql)
                if result.error:
                    self._send_json_response({"error": f"Query failed: {result.error}"}, 500)
                    return
                
                # Update context
                if context and result.columns:
                    context.add_query(natural_language, table_name, result.columns, "sample_data")
                
                response = {
                    "job_id": job_id,
                    "status": "completed",
                    "rows": result.rows,
                    "columns": list(result.rows[0].keys()) if result.rows and len(result.rows) > 0 and isinstance(result.rows[0], dict) else [],
                    "table_used": table_name,
                    "message": f"Sample data from {table_name}",
                    "context_suggestions": context.get_continuation_suggestions() if context else [],
                    "session_id": context.session_id if context else job_id
                }
                self._send_json_response(response)
                
        except Exception as e:
            print(f"🚨 Error in _process_query_with_table: {str(e)}")
            self._send_json_response({"error": f"Query processing failed: {str(e)}"}, 500)
    
    def _find_similar_columns(self, target_column, available_columns, threshold=0.5):
        """Enhanced column matching with intelligent suggestions"""
        import difflib
        import re
        
        target_lower = target_column.lower().strip()
        matches = []
        
        for col in available_columns:
            col_lower = col.lower().strip()
            
            # Exact match
            if target_lower == col_lower:
                return [col], 1.0
            
            # Exact match ignoring underscores and spaces
            target_normalized = re.sub(r'[_\s]+', '', target_lower)
            col_normalized = re.sub(r'[_\s]+', '', col_lower)
            if target_normalized == col_normalized:
                return [col], 0.95
            
            # Substring match (both directions)
            if target_lower in col_lower:
                matches.append((col, 0.85))
                continue
            elif col_lower in target_lower:
                matches.append((col, 0.80))
                continue
            
            # Word-based matching (split by common separators)
            target_words = set(re.split(r'[_\s\-\.]+', target_lower))
            col_words = set(re.split(r'[_\s\-\.]+', col_lower))
            
            # Check for word overlap
            common_words = target_words.intersection(col_words)
            if common_words:
                word_score = len(common_words) / max(len(target_words), len(col_words))
                if word_score >= 0.5:
                    matches.append((col, 0.70 + word_score * 0.15))
                    continue
            
            # Fuzzy matching with sequence matcher
            similarity = difflib.SequenceMatcher(None, target_lower, col_lower).ratio()
            if similarity >= threshold:
                matches.append((col, similarity))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches[:5]], matches[0][1] if matches else 0.0

    def _extract_requested_columns(self, natural_language):
        """Extract column names mentioned in the natural language query"""
        import re
        
        requested_columns = []
        
        # Enhanced pattern to capture multiple columns with "and" connector
        multi_column_patterns = [
            r'frequency of (.+?)(?:\s+from|\s+in|\s+table|$)',
            r'analyze (.+?)(?:\s+from|\s+in|\s+table|$)',
            r'group by (.+?)(?:\s+from|\s+in|\s+table|$)',
            r'count (.+?)(?:\s+from|\s+in|\s+table|$)',
        ]
        
        for pattern in multi_column_patterns:
            matches = re.findall(pattern, natural_language.lower(), re.IGNORECASE)
            for match in matches:
                # Split on "and" to get multiple columns
                column_parts = re.split(r'\s+and\s+', match.strip())
                for part in column_parts:
                    # Clean up each part
                    clean_part = part.strip().replace(' ', '_')
                    # Remove common words that aren't column names
                    clean_part = re.sub(r'^(the|a|an)\s+', '', clean_part)
                    if clean_part and clean_part not in requested_columns:
                        requested_columns.append(clean_part)
        
        # Also look for individual column patterns if multi-column didn't match
        if not requested_columns:
            individual_patterns = [
                r'frequency of (\w+(?:\s+\w+)*)',
                r'analyze (\w+(?:\s+\w+)*)',
                r'group by (\w+(?:\s+\w+)*)',
                r'count (\w+(?:\s+\w+)*)',
                r'column (\w+(?:\s+\w+)*)',
                r'field (\w+(?:\s+\w+)*)',
            ]
            
            for pattern in individual_patterns:
                matches = re.findall(pattern, natural_language.lower())
                for match in matches:
                    clean_match = match.strip().replace(' ', '_')
                    if clean_match not in requested_columns:
                        requested_columns.append(clean_match)
        
        # Look for quoted column names
        quoted_matches = re.findall(r'["\'](\w+(?:\s+\w+)*)["\']', natural_language)
        for match in quoted_matches:
            clean_match = match.strip().replace(' ', '_')
            if clean_match not in requested_columns:
                requested_columns.append(clean_match)
        
        # Look for common column-like phrases in the query
        common_column_phrases = [
            r'recommended\s+message',
            r'provider\s+input',
            r'input\s+provider',
            r'message\s+type',
            r'output\s+type',
            r'data\s+source'
        ]
        
        for phrase_pattern in common_column_phrases:
            matches = re.findall(phrase_pattern, natural_language.lower())
            for match in matches:
                clean_match = match.replace(' ', '_')
                if clean_match not in requested_columns:
                    requested_columns.append(clean_match)
        
        print(f"🔍 Extracted columns from query: {requested_columns}")
        return requested_columns

    def _handle_frequency_analysis(self, natural_language, job_id, table_name, context=None):
        """Enhanced frequency analysis with intelligent column matching"""
        import re
        
        try:
            # Get table schema and available columns
            table_info_sql = f'SELECT * FROM "{table_name}" LIMIT 1'
            table_info = adapter.run(table_info_sql)
            
            if table_info.error:
                self._send_json_response({"error": f"Cannot access table: {table_info.error}"}, 500)
                return
            
            # Get columns from the first row if data exists
            columns = []
            if table_info.rows and len(table_info.rows) > 0:
                if isinstance(table_info.rows[0], dict):
                    columns = list(table_info.rows[0].keys())
                else:
                    # Try to get schema information
                    try:
                        from backend.agents.schema_embedder import SchemaEmbedder
                        embedder = SchemaEmbedder()
                        schema_data = embedder._get_table_metadata(table_name)
                        if schema_data and 'columns' in schema_data:
                            columns = [col['column_name'] for col in schema_data['columns']]
                    except:
                        # Fallback: try DESCRIBE query
                        try:
                            desc_sql = f'DESCRIBE TABLE "{table_name}"'
                            desc_result = adapter.run(desc_sql)
                            if not desc_result.error and desc_result.rows:
                                columns = [row[0] if isinstance(row, (list, tuple)) else row.get('name', '') for row in desc_result.rows]
                        except:
                            columns = []
            
            if not columns:
                self._send_json_response({"error": f"Could not determine columns for table {table_name}"}, 500)
                return
            
            # Extract requested columns from the query
            requested_columns = self._extract_requested_columns(natural_language)
            
            # Find the best matching columns for frequency analysis
            frequency_targets = []
            suggestions = []
            
            # Check for requested columns
            for requested_col in requested_columns:
                similar_cols, similarity = self._find_similar_columns(requested_col, columns)
                
                if similar_cols and similarity >= 0.8:
                    if similar_cols[0] not in frequency_targets:
                        frequency_targets.append(similar_cols[0])
                elif similar_cols:
                    suggestions.extend([(requested_col, similar_cols)])
            
            # If no specific columns were found and we have suggestions, use LLM for better analysis
            if not frequency_targets and (suggestions or requested_columns):
                # Use LLM agent for intelligent column analysis
                try:
                    from backend.agents.llm_agent import LLMAgent
                    llm_agent = LLMAgent()
                    
                    column_analysis = llm_agent.analyze_column_requirements(
                        natural_language, table_name, columns
                    )
                    
                    # Try to use LLM recommendations with execution strategy
                    if column_analysis.get('execution_strategy'):
                        exec_strategy = column_analysis['execution_strategy']
                        strategy_confidence = exec_strategy.get('confidence', 0)
                        try:
                            strategy_confidence = float(strategy_confidence) if strategy_confidence is not None else 0.0
                        except (ValueError, TypeError):
                            strategy_confidence = 0.0
                            
                        if strategy_confidence >= 0.8 and exec_strategy.get('primary_column'):
                            primary_col = exec_strategy['primary_column']
                            if primary_col not in frequency_targets:
                                frequency_targets.append(primary_col)
                            print(f"✅ Auto-executing with LLM strategy: {primary_col} ({strategy_confidence:.1%} confidence)")
                    
                    # Fallback to column matches
                    if not frequency_targets and column_analysis.get('column_matches'):
                        for match in column_analysis['column_matches']:
                            match_confidence = match.get('confidence', 0)
                            try:
                                match_confidence = float(match_confidence) if match_confidence is not None else 0.0
                            except (ValueError, TypeError):
                                match_confidence = 0.0
                                
                            if match_confidence >= 0.8:
                                matched_col = match['matched']
                                if matched_col not in frequency_targets:
                                    frequency_targets.append(matched_col)
                                print(f"✅ Auto-executing with column match: {matched_col} ({match_confidence:.1%} confidence)")
                        
                    # Use suggested alternatives as last resort
                    if not frequency_targets and column_analysis.get('suggested_alternatives'):
                        for alt in column_analysis['suggested_alternatives'][:2]:  # Take top 2
                            alt_col = alt['column']
                            if alt_col not in frequency_targets:
                                frequency_targets.append(alt_col)
                        print(f"✅ Auto-executing with suggested alternatives: {frequency_targets}")
                    
                    # Only show suggestions if we truly can't proceed
                    if not frequency_targets:
                        recommended_action = column_analysis.get('recommended_action', '')
                        if 'clarification needed' in recommended_action.lower():
                            # Show user the options
                            error_msg = f"🤔 Need clarification to proceed with your analysis.\n\n"
                            error_msg += f"Goal: {column_analysis.get('user_goal', 'Unknown')}\n\n"
                            
                            if column_analysis.get('column_matches'):
                                error_msg += "📊 Possible data matches:\n"
                                for match in column_analysis['column_matches'][:3]:
                                    error_msg += f"  • '{match['matched']}' - {match['reason']}\n"
                            
                            if column_analysis.get('suggested_alternatives'):
                                error_msg += "\n💡 Alternative approaches:\n"
                                for alt in column_analysis['suggested_alternatives'][:3]:
                                    error_msg += f"  • '{alt['column']}' - {alt['business_value']}\n"
                            
                            error_msg += f"\n🎯 Recommended: {recommended_action}"
                            
                            self._send_json_response({
                                "status": "clarification_needed",
                                "message": error_msg,
                                "execution_options": column_analysis.get('suggested_alternatives', [])[:3],
                                "user_goal": column_analysis.get('user_goal'),
                                "available_columns": columns,
                                "query": natural_language
                            }, 200)
                            return
                        else:
                            # Auto-execute with first available column as last resort
                            if columns:
                                frequency_targets.append(columns[0])
                
                except Exception as e:
                    print(f"🚨 LLM column analysis failed: {e}")
                    # Fall back to basic suggestions
                    pass
            
            # Enhanced fallback logic for finding suitable columns
            if not frequency_targets:
                query_lower = natural_language.lower()
                all_suggestions = []
                
                # Look for any column mentioned in the query
                for col in columns:
                    if col.lower() in query_lower:
                        if col not in frequency_targets:
                            frequency_targets.append(col)
                        break
                
                # If no direct match, collect intelligent suggestions
                if not frequency_targets:
                    # Check for partial matches with the query terms
                    query_words = set(re.split(r'[_\s\-\.]+', query_lower))
                    
                    for col in columns:
                        col_words = set(re.split(r'[_\s\-\.]+', col.lower()))
                        
                        # Word overlap scoring
                        common_words = query_words.intersection(col_words)
                        if common_words:
                            overlap_score = len(common_words) / len(query_words)
                            all_suggestions.append({
                                'column': col,
                                'score': overlap_score,
                                'reason': f"Contains {len(common_words)} matching words: {', '.join(common_words)}"
                            })
                    
                    # Smart fallback to common healthcare/commercial patterns
                    if not all_suggestions:
                        priority_keywords = [
                            ('provider', 'healthcare provider information'),
                            ('patient', 'patient-related data'),
                            ('drug', 'pharmaceutical data'),
                            ('therapy', 'treatment information'),
                            ('diagnosis', 'diagnostic information'),
                            ('prescription', 'prescription data'),
                            ('nps', 'net promoter score'),
                            ('satisfaction', 'satisfaction metrics'),
                            ('outcome', 'patient outcomes'),
                            ('cost', 'cost analysis'),
                            ('revenue', 'revenue metrics'),
                            ('channel', 'distribution channels'),
                            ('market', 'market data'),
                            ('share', 'market share'),
                            ('recommend', 'recommendation data'),
                            ('message', 'message/communication data'),
                            ('type', 'categorization data'),
                            ('category', 'category information'),
                            ('status', 'status tracking'),
                            ('input', 'input parameters'),
                            ('output', 'output results')
                        ]
                        
                        for keyword, description in priority_keywords:
                            for col in columns:
                                if keyword in col.lower():
                                    all_suggestions.append({
                                        'column': col,
                                        'score': 0.7,
                                        'reason': f"Healthcare/commercial keyword match: {description}"
                                    })
                
                # Sort suggestions by score and use the best ones
                if all_suggestions:
                    all_suggestions.sort(key=lambda x: x['score'], reverse=True)
                    # Take top 2 columns for multi-column analysis
                    for suggestion in all_suggestions[:2]:
                        if suggestion['column'] not in frequency_targets:
                            frequency_targets.append(suggestion['column'])
                    
                    # If score is low, provide user with suggestions
                    if all_suggestions[0]['score'] < 0.5:
                        suggestion_msg = f"🤔 No exact column match found. Using {frequency_targets} as best guess.\n\n"
                        suggestion_msg += "💡 Better column suggestions:\n"
                        
                        for i, suggestion in enumerate(all_suggestions[:5]):
                            suggestion_msg += f"  {i+1}. '{suggestion['column']}' - {suggestion['reason']}\n"
                        
                        suggestion_msg += f"\n📋 All available columns: {', '.join(columns[:10])}"
                        if len(columns) > 10:
                            suggestion_msg += f" ... and {len(columns)-10} more"
                        
                        self._send_json_response({
                            "status": "column_suggestion",
                            "message": suggestion_msg,
                            "using_columns": frequency_targets,
                            "suggestions": all_suggestions[:5],
                            "available_columns": columns,
                            "query": natural_language
                        }, 200)
                        # Continue with analysis using the suggested columns
                
                # Last resort - use first suitable column
                if not frequency_targets and columns:
                    frequency_targets.append(columns[0])
            
            # Execute frequency analysis (single or multi-column)
            if len(frequency_targets) == 1:
                # Single column analysis
                frequency_target = frequency_targets[0]
                sql = f'''
                SELECT "{frequency_target}", COUNT(*) as frequency 
                FROM "{table_name}" 
                WHERE "{frequency_target}" IS NOT NULL 
                GROUP BY "{frequency_target}" 
                ORDER BY frequency DESC 
                LIMIT 10
                '''
                print(f"📊 Single-column analysis: {frequency_target}")
                
            elif len(frequency_targets) > 1:
                # Multi-column analysis - create a combined frequency analysis
                frequency_target = frequency_targets[0]  # Primary for response
                column_list = '", "'.join(frequency_targets)
                sql = f'''
                SELECT "{frequency_targets[0]}", "{frequency_targets[1]}", COUNT(*) as frequency 
                FROM "{table_name}" 
                WHERE "{frequency_targets[0]}" IS NOT NULL 
                AND "{frequency_targets[1]}" IS NOT NULL 
                GROUP BY "{frequency_targets[0]}", "{frequency_targets[1]}" 
                ORDER BY frequency DESC 
                LIMIT 10
                '''
                print(f"📊 Multi-column analysis: {frequency_targets}")
                
            else:
                self._send_json_response({"error": "No suitable columns found for analysis"}, 400)
                return
            
            result = adapter.run(sql)
            if result.error:
                self._send_json_response({"error": f"Frequency query failed: {result.error}"}, 500)
                return
            
            # Create enhanced visualization based on column count
            if len(frequency_targets) == 1:
                # Single column visualization
                labels = [str(row[0]) for row in result.rows]
                values = [row[-1] for row in result.rows]  # Last column is always frequency
                
                plotly_spec = {
                    "data": [{
                        "x": labels,
                        "y": values,
                        "type": "bar",
                        "name": "Frequency",
                        "marker": {"color": "#1f77b4"},
                        "text": values,
                        "textposition": "auto"
                    }],
                    "layout": {
                        "title": f"Frequency Analysis: {frequency_targets[0]}",
                        "xaxis": {"title": frequency_targets[0]},
                        "yaxis": {"title": "Frequency"},
                        "margin": {"l": 50, "r": 50, "t": 80, "b": 100}
                    }
                }
                analysis_message = f"Frequency analysis of '{frequency_targets[0]}' from {table_name}"
                
            else:
                # Multi-column visualization - create a grouped bar chart or heatmap
                col1_values = [str(row[0]) for row in result.rows]
                col2_values = [str(row[1]) for row in result.rows]
                frequencies = [row[-1] for row in result.rows]
                
                # Create labels combining both columns
                combined_labels = [f"{c1} & {c2}" for c1, c2 in zip(col1_values, col2_values)]
                
                plotly_spec = {
                    "data": [{
                        "x": combined_labels,
                        "y": frequencies,
                        "type": "bar",
                        "name": "Combined Frequency",
                        "marker": {"color": "#2ca02c"},
                        "text": frequencies,
                        "textposition": "auto"
                    }],
                    "layout": {
                        "title": f"Combined Frequency Analysis: {' & '.join(frequency_targets)}",
                        "xaxis": {"title": f"{frequency_targets[0]} & {frequency_targets[1]}", "tickangle": -45},
                        "yaxis": {"title": "Frequency"},
                        "margin": {"l": 50, "r": 50, "t": 80, "b": 150}
                    }
                }
                analysis_message = f"Combined frequency analysis of '{' & '.join(frequency_targets)}' from {table_name}"
            
            # Prepare response with helpful information
            if len(frequency_targets) == 1:
                response_columns = [frequency_targets[0], "frequency"]
                message = analysis_message
                if requested_columns and frequency_targets[0].lower() not in [req.lower() for req in requested_columns]:
                    message += f" (used '{frequency_targets[0]}' as best match)"
            else:
                response_columns = frequency_targets + ["frequency"]
                message = analysis_message
                
            # Update conversation context
            context = self._get_session_context(job_id)
            context.add_query(natural_language, table_name, frequency_targets)
            context_suggestions = context.get_continuation_suggestions()
            
            response = {
                "job_id": job_id,
                "status": "completed", 
                "rows": result.rows,
                "columns": response_columns,
                "table_used": table_name,
                "plotly_spec": plotly_spec,
                "message": message,
                "columns_used": frequency_targets,
                "available_columns": columns,
                "context_suggestions": context_suggestions
            }
            self._send_json_response(response)
                
        except Exception as e:
            print(f"🚨 Error in frequency analysis: {str(e)}")
            self._send_json_response({"error": f"Frequency analysis failed: {str(e)}"}, 500)
    
    def _execute_analysis_with_column(self, natural_language, job_id, table_name, column_name):
        """Execute analysis with user's chosen column"""
        try:
            print(f"🚀 Executing forced analysis: {column_name} on {table_name}")
            
            # Execute frequency analysis with the chosen column
            sql = f'''
            SELECT "{column_name}", COUNT(*) as frequency 
            FROM "{table_name}" 
            WHERE "{column_name}" IS NOT NULL 
            GROUP BY "{column_name}" 
            ORDER BY frequency DESC 
            LIMIT 10
            '''
            
            print(f"📊 Executing SQL: {sql}")
            result = adapter.run(sql)
            
            if result.error:
                self._send_json_response({
                    "error": f"Query execution failed: {result.error}",
                    "sql": sql
                }, 500)
                return
            
            # Generate insights for the results
            insights = self._generate_insights(
                natural_language, 
                result.rows, 
                result.columns, 
                table_name
            )
            
            response = {
                "job_id": job_id,
                "status": "completed",
                "message": f"✅ Analysis completed using '{column_name}' column",
                "sql": sql,
                "table_name": table_name,
                "column_used": column_name,
                "row_count": len(result.rows),
                "columns": result.columns,
                "rows": result.rows,
                "insights": insights,
                "analysis_type": "frequency_analysis",
                "execution_method": "user_choice"
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            print(f"🚨 Error in forced execution: {str(e)}")
            self._send_json_response({
                "error": f"Analysis execution failed: {str(e)}",
                "column": column_name,
                "table": table_name
            }, 500)
    
    def _generate_insights(self, query, data_rows=None, columns=None, table_name=""):
        """Generate intelligent insights using Agent Orchestrator"""
        
        if not data_rows or not columns:
            return "No data available for insight generation."
        
        try:
            # Use Agent Orchestrator for intelligent insights
            if agent_orchestrator.is_initialized:
                print("🤖 Generating intelligent insights with LLM Agent...")
                
                data_analysis = {
                    "rows": data_rows,
                    "columns": columns,
                    "table_name": table_name
                }
                
                insights = agent_orchestrator.generate_intelligent_insights(
                    query, data_analysis, table_name
                )
                
                return insights
            
        except Exception as e:
            print(f"⚠️ Agent insight generation failed: {e}")
            print("🔄 Falling back to basic analysis...")
        
        # Fallback to basic insights
        try:
            df = pd.DataFrame(data_rows, columns=columns)
            
            insights = []
            insights.append(f"📊 Dataset Analysis for {table_name}:")
            insights.append(f"• Contains {len(df)} records with {len(df.columns)} attributes")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                insights.append(f"• Found {len(numeric_cols)} numeric columns")
            if text_cols:
                insights.append(f"• Found {len(text_cols)} text columns")
            
            # Basic recommendations
            insights.append("🔍 Recommendations:")
            insights.append("• Explore data patterns with filtering and grouping")
            insights.append("• Consider visualizing trends with charts")
            
            return "\n".join(insights)
            
        except Exception as e:
            return f"Basic data summary: {len(data_rows)} rows, {len(columns)} columns from {table_name}"
    
    def _send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

def run_server(port=8003):
    server_address = ('', port)
    
    # Initialize agents before starting server
    print("🔧 Initializing AI agents...")
    initialize_agents()
    
    httpd = HTTPServer(server_address, NBAHandler)
    print(f"🚀 Enhanced NBA Query Agent running on http://localhost:{port}")
    print(f"✅ Database: {type(adapter).__name__}")
    
    # Show agent status
    agent_status = agent_orchestrator.get_system_status()
    if agent_status['initialized']:
        print(f"🤖 AI Agents: READY ({agent_status['total_tables']} tables indexed)")
        if agent_status['llm_agent_ready']:
            print(f"🧠 LLM Agent: OpenAI Connected")
        else:
            print(f"🧠 LLM Agent: Fallback Mode (no API key)")
        # Check for vector search capability
        if 'vector_search_ready' in agent_status:
            print(f"🔍 Vector Search: {agent_status['vector_search_ready']}")
        elif 'openai_vector_ready' in agent_status:
            print(f"🔍 OpenAI Vector Search: {agent_status['openai_vector_ready']}")
        else:
            print(f"🔍 Vector Search: Available")
    else:
        print(f"⚠️ AI Agents: Limited Mode")
    
    print(f"📊 Ready for NBA table queries with intelligent suggestions")
    print(f"🔗 Test endpoints:")
    print(f"   GET  http://localhost:{port}/health")
    print(f"   GET  http://localhost:{port}/test-nba")
    print(f"   GET  http://localhost:{port}/tables")
    print(f"   GET  http://localhost:{port}/agent-status")
    print(f"   POST http://localhost:{port}/query")
    print(f"   POST http://localhost:{port}/query-with-table")
    print(f"   POST http://localhost:{port}/insights")
    print("\n📝 Example queries:")
    print('   "Show me NBA data"')
    print('   "Get frequency analysis"')
    print('   "Top 10 records from any table"')
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.server_close()

if __name__ == "__main__":
    print("🚀 Starting NBA Agent Server with OpenAI Embeddings...")
    
    # Initialize agents first
    initialize_agents()
    
    # Start server
    run_server()
