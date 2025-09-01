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

# Initialize Agent Orchestrator
from agents.orchestrator import AgentOrchestrator
agent_orchestrator = AgentOrchestrator(openai_api_key=os.getenv('OPENAI_API_KEY'))

# Initialize agents with available tables
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

def get_available_tables():
    """Get all available tables from the database"""
    try:
        result = adapter.run("SHOW TABLES")
        if result.error:
            print(f"❌ Error getting tables: {result.error}")
            return []
        tables = [row[1] for row in result.rows]  # Table name is usually in second column
        print(f"📋 Found {len(tables)} tables: {tables[:5]}...")  # Show first 5
        return tables
    except Exception as e:
        print(f"❌ Exception getting tables: {e}")
        return []

def suggest_similar_tables(query_text, available_tables, max_suggestions=5):
    """
    Intelligent table suggestion using Agent Orchestrator with FAISS and LLM
    Falls back to simple matching if agents are not available
    """
    try:
        if agent_orchestrator.is_initialized:
            print("🤖 Using intelligent agent-based table suggestion...")
            suggestion_result = agent_orchestrator.intelligent_table_suggestion(
                query_text, max_suggestions
            )
            
            # Extract table suggestions in the expected format
            suggestions = []
            for table_data in suggestion_result.get('suggested_tables', []):
                suggestions.append({
                    'table': table_data['table_name'],
                    'score': table_data['total_score'],
                    'confidence': table_data['confidence'],
                    'reasons': table_data.get('reasons', []),
                    'match_type': table_data.get('match_type', 'hybrid')
                })
            
            print(f"🎯 Agent suggestions: {[s['table'] for s in suggestions[:3]]}")
            return suggestions, suggestion_result.get('intent_analysis', {}), suggestion_result.get('execution_plan', {})
        
    except Exception as e:
        print(f"⚠️ Agent suggestion failed: {e}")
        print("🔄 Falling back to simple matching...")
    
    # Fallback to simple matching
    return _fallback_table_suggestion(query_text, available_tables, max_suggestions), {}, {}


def _fallback_table_suggestion(query_text, available_tables, max_suggestions):
    """Fallback table suggestion when agents are not available"""
    # Extract potential table references from query
    potential_names = []
    
    # Enhanced patterns for table extraction
    patterns = [
        r'table\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # "table tablename"
        r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)',   # "from tablename"
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s+table',  # "tablename table"
        r'([a-zA-Z_]+[a-zA-Z0-9_]*)',         # Any identifier
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, query_text, re.IGNORECASE)
        potential_names.extend(matches)
    
    # Extract key words from query (filter out common words)
    common_words = {'read', 'table', 'from', 'select', 'show', 'get', 'fetch', 'and', 'with', 'create', 'visualization', 'frequency', 'top', 'rows'}
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query_text.lower())
    meaningful_words = [word for word in words if len(word) >= 3 and word not in common_words]
    potential_names.extend(meaningful_words)
    
    print(f"🔍 Extracted potential names: {potential_names}")
    print(f"🔍 Meaningful words: {meaningful_words}")
    
    # Score-based matching with multiple criteria
    table_scores = []
    
    # Special handling for NBA-related queries
    nba_related = any(word in query_text.lower() for word in ['nba', 'final', 'output', 'python'])
    
    for table in available_tables:
        score = 0
        table_lower = table.lower()
        reasons = []
        
        # Exact name matching (with fuzzy tolerance)
        for name in potential_names:
            name_lower = name.lower()
            
            if name_lower == table_lower:
                score += 500  # Perfect match
                reasons.append(f"Exact match with '{name}'")
            elif name_lower in table_lower:
                score += 100
                reasons.append(f"Contains '{name}'")
            elif table_lower in name_lower:
                score += 90
                reasons.append(f"Reverse match with '{name}'")
            else:
                # Fuzzy matching with difflib
                similarity = difflib.SequenceMatcher(None, name_lower, table_lower).ratio()
                if similarity > 0.5:
                    score += int(similarity * 50)
                    reasons.append(f"Fuzzy match with '{name}' ({similarity:.2f})")
        
        # NBA-specific scoring
        if nba_related and 'nba' in table_lower:
            score += 150
            reasons.append("NBA domain match")
        
        # Meaningful word bonus
        matching_words = sum(1 for word in meaningful_words if word in table_lower)
        if matching_words > 0:
            score += matching_words * 25
            reasons.append(f"Contains {matching_words} meaningful words")
        
        if score > 0:
            table_scores.append({
                'table': table,
                'score': score / 100,  # Normalize
                'confidence': 'high' if score > 200 else 'medium' if score > 100 else 'low',
                'reasons': reasons,
                'match_type': 'pattern'
            })
    
    # Sort by score and return top suggestions
    table_scores.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"📋 Table scores (top 5): {[(t['table'], t['score']) for t in table_scores[:5]]}")
    
    return table_scores[:max_suggestions]


def find_exact_table_match(query_text, available_tables):
    """Find exact table matches in the query with improved precision"""
    query_lower = query_text.lower()
    
    # Sort tables by length (longest first) to avoid partial matches
    sorted_tables = sorted(available_tables, key=len, reverse=True)
    
    # First, look for very specific NBA table patterns
    nba_keywords = ['final_nba_output_python', 'nba_output_python', 'final_nba_output']
    for keyword in nba_keywords:
        for table in sorted_tables:
            if keyword in table.lower():
                # Check if the query contains elements of this table name
                table_parts = table.lower().replace('_', ' ').split()
                query_parts = query_lower.replace('_', ' ').split()
                
                # Check if key parts of table name appear in query
                matches = sum(1 for part in table_parts if part in query_parts)
                if matches >= 3:  # Need at least 3 matching parts
                    print(f"✅ Found specific NBA table match: '{table}' (matched {matches} parts)")
                    return table
    
    # Check for exact table name matches (case insensitive) - but be more careful
    for table in sorted_tables:
        table_lower = table.lower()
        
        # Skip very short table names that might cause false positives
        if len(table) <= 3:
            continue
            
        # Check if full table name appears in query
        if table_lower in query_lower:
            # Additional validation - make sure it's not a substring of a larger word
            import re
            pattern = r'\b' + re.escape(table_lower) + r'\b'
            if re.search(pattern, query_lower):
                print(f"✅ Found exact word-boundary match: '{table}' in query")
                return table
        
        # Also check quoted versions
        quoted_variations = [f'"{table}"', f"'{table}'", f"`{table}`"]
        for quoted in quoted_variations:
            if quoted.lower() in query_lower:
                print(f"✅ Found exact quoted match: '{quoted}' in query")
                return table
    
    return None

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
            # New endpoint to get available tables
            tables = get_available_tables()
            self._send_json_response({
                "tables": tables,
                "count": len(tables)
            })
        
        elif parsed_path.path == '/agent-status':
            # Get agent system status
            status = agent_orchestrator.get_system_status()
            status['agent_endpoints'] = {
                'intelligent_suggestions': '/query (enhanced)',
                'llm_insights': '/insights (enhanced)',
                'system_status': '/agent-status'
            }
            self._send_json_response(status)
        
        else:
            self._send_json_response({"error": "Not found"}, 404)
    
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
                
                print(f"🔍 Processing query: {natural_language}")
                
                # Get available tables
                available_tables = get_available_tables()
                
                # Check if user provided selected tables
                if selected_tables:
                    print(f"👤 User selected tables: {selected_tables}")
                    # Use the selected tables for the query
                    if len(selected_tables) == 1:
                        table_name = selected_tables[0]
                        sql = f'SELECT * FROM "{table_name}" LIMIT 10'
                        
                        result = adapter.run(sql)
                        if result.error:
                            self._send_json_response({"error": f"Query failed: {result.error}"}, 500)
                            return
                        
                        response = {
                            "job_id": job_id,
                            "status": "completed",
                            "rows": result.rows,
                            "table_used": table_name,
                            "message": f"Retrieved data from {table_name}"
                        }
                        self._send_json_response(response)
                        return
                
                # Check for exact table match first
                exact_table = find_exact_table_match(natural_language, available_tables)
                
                if exact_table:
                    print(f"✅ Found exact table match: {exact_table}")
                    # Process with exact table
                    self._process_query_with_table(natural_language, job_id, exact_table)
                else:
                    # No exact match - use intelligent agent-based suggestions
                    suggestions, intent_analysis, execution_plan = suggest_similar_tables(
                        natural_language, available_tables
                    )
                    print(f"💡 Agent-based suggestions: {[s.get('table', s) for s in suggestions]}")
                    
                    if suggestions:
                        # Format suggestions for frontend
                        formatted_suggestions = []
                        for suggestion in suggestions:
                            if isinstance(suggestion, dict):
                                formatted_suggestions.append({
                                    'table': suggestion.get('table_name', suggestion.get('table', '')),
                                    'score': suggestion.get('total_score', suggestion.get('score', 0)),
                                    'confidence': suggestion.get('confidence', 'medium'),
                                    'reasons': suggestion.get('reasons', []),
                                    'match_type': suggestion.get('match_type', 'pattern')
                                })
                            else:
                                # Handle simple string suggestions (fallback)
                                formatted_suggestions.append({
                                    'table': suggestion,
                                    'score': 0.5,
                                    'confidence': 'medium',
                                    'reasons': ['Basic pattern match'],
                                    'match_type': 'simple'
                                })
                        
                        response = {
                            "job_id": job_id,
                            "status": "needs_table_selection",
                            "message": "Intelligent analysis complete. Select from suggested tables:",
                            "suggested_tables": [s['table'] for s in formatted_suggestions],  # Simple format for compatibility
                            "detailed_suggestions": formatted_suggestions,  # Detailed format for enhanced UI
                            "intent_analysis": intent_analysis,
                            "execution_plan": execution_plan,
                            "all_tables": available_tables[:20],  # Limit to first 20
                            "query": natural_language,
                            "agent_enhanced": True
                        }
                        self._send_json_response(response)
                    else:
                        # No suggestions either - show all tables
                        response = {
                            "job_id": job_id,
                            "status": "needs_table_selection", 
                            "message": "No table matches found. Please select from available tables:",
                            "suggested_tables": available_tables[:10],  # Show top 10
                            "all_tables": available_tables[:20],
                            "query": natural_language,
                            "agent_enhanced": False
                        }
                        self._send_json_response(response)
                    
            except Exception as e:
                print(f"🚨 Error processing query: {str(e)}")
                self._send_json_response({"error": f"Request processing failed: {str(e)}"}, 400)
        
        elif self.path == '/query-with-table':
            # New endpoint for queries with selected tables
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                natural_language = request_data.get('natural_language', '')
                job_id = request_data.get('job_id', 'default')
                selected_tables = request_data.get('selected_tables', [])
                
                if not selected_tables:
                    self._send_json_response({"error": "No tables selected"}, 400)
                    return
                
                # Process query with selected tables
                table_name = selected_tables[0]  # Use first selected table
                self._process_query_with_table(natural_language, job_id, table_name)
                
            except Exception as e:
                print(f"🚨 Error processing table query: {str(e)}")
                self._send_json_response({"error": f"Request processing failed: {str(e)}"}, 400)
        
        elif self.path == '/insights':
            # Add insights endpoint with data-driven analysis
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                location = request_data.get('location', '')
                query = request_data.get('query', '')
                
                # Try to extract data from the request if available
                data_rows = request_data.get('data_rows', [])
                columns = request_data.get('columns', [])
                table_name = request_data.get('table_name', '')
                
                # If no data provided, try to get some sample data for analysis
                if not data_rows and table_name:
                    try:
                        sample_result = adapter.run(f'SELECT * FROM "{table_name}" LIMIT 100')
                        if not sample_result.error:
                            data_rows = sample_result.rows
                            # Get column names
                            desc_result = adapter.run(f'DESCRIBE TABLE "{table_name}"')
                            if not desc_result.error:
                                columns = [row[0] for row in desc_result.rows]
                    except Exception as e:
                        print(f"Could not fetch sample data: {e}")
                
                # Generate data-driven insights
                insight = self._generate_insights(query, location, data_rows, columns, table_name)
                
                response = {
                    "insight": insight,
                    "query": query,
                    "location": location,
                    "data_analyzed": len(data_rows) if data_rows else 0,
                    "columns_analyzed": len(columns) if columns else 0
                }
                self._send_json_response(response)
                
            except Exception as e:
                print(f"🚨 Error generating insights: {str(e)}")
                self._send_json_response({"error": f"Insights generation failed: {str(e)}"}, 400)
        
        else:
            print(f"❌ Unknown POST path: {self.path}")
            self._send_json_response({"error": "Not found"}, 404)
    
    def _process_query_with_table(self, natural_language, job_id, table_name):
        """Process a query with a specific table"""
        print(f"🔧 Processing query with table: {table_name}")
        
        # Default simple query for testing
        if not natural_language.strip():
            natural_language = "show top 10 records"
        
        # Handle general queries
        if "top" in natural_language.lower() and ("5" in natural_language or "10" in natural_language):
            limit = "5" if "5" in natural_language else "10"
            sql = f'SELECT * FROM "{table_name}" LIMIT {limit}'
            
            result = adapter.run(sql)
            if result.error:
                print(f"❌ Query error: {result.error}")
                self._send_json_response({"error": f"Query failed: {result.error}"}, 500)
                return
            
            # Simple response for top records
            response = {
                "job_id": job_id,
                "status": "completed", 
                "rows": result.rows[:int(limit)],
                "table_used": table_name,
                "plotly_spec": None,
                "message": f"Retrieved top {limit} records from {table_name}"
            }
            print(f"✅ Query successful, returning {len(result.rows)} rows")
            self._send_json_response(response)
            return
        
        # Handle frequency analysis
        elif "frequency" in natural_language.lower():
            # Parse what specific frequency analysis is requested
            frequency_target = None
            
            if "recommended message" in natural_language.lower():
                # Look for recommended message columns
                desc_result = adapter.run(f'DESCRIBE TABLE "{table_name}"')
                if desc_result.error:
                    self._send_json_response({"error": f"Cannot describe table: {desc_result.error}"}, 500)
                    return
                
                columns = [row[0] for row in desc_result.rows]
                
                # Find recommended message column
                for col in columns:
                    if "recommend" in col.lower() and "msg" in col.lower():
                        frequency_target = col
                        break
                    elif "recommended_msg" in col.lower():
                        frequency_target = col
                        break
                
                if frequency_target:
                    print(f"🎯 Found recommended message column: {frequency_target}")
                    
                    sql = f'''
                    SELECT "{frequency_target}", COUNT(*) as frequency 
                    FROM "{table_name}" 
                    WHERE "{frequency_target}" IS NOT NULL 
                    AND "{frequency_target}" != '' 
                    AND "{frequency_target}" != '{{}}' 
                    GROUP BY "{frequency_target}" 
                    ORDER BY frequency DESC 
                    LIMIT 15
                    '''
                    
                    result = adapter.run(sql)
                    if result.error:
                        self._send_json_response({"error": f"Frequency query failed: {result.error}"}, 500)
                        return
                    
                    # Create visualization
                    labels = [str(row[0])[:50] + "..." if len(str(row[0])) > 50 else str(row[0]) for row in result.rows]  # Truncate long labels
                    values = [row[1] for row in result.rows]
                    
                    plotly_spec = {
                        "data": [{
                            "x": labels,
                            "y": values,
                            "type": "bar",
                            "name": "Frequency",
                            "marker": {"color": "#1f77b4"},
                            "hovertemplate": "<b>%{x}</b><br>Count: %{y}<extra></extra>"
                        }],
                        "layout": {
                            "title": f"Frequency Analysis: {frequency_target}",
                            "xaxis": {
                                "title": "Recommended Messages", 
                                "tickangle": -45,
                                "automargin": True
                            },
                            "yaxis": {"title": "Frequency"},
                            "margin": {"l": 60, "r": 30, "t": 80, "b": 150},
                            "height": 600
                        }
                    }
                    
                    response = {
                        "job_id": job_id,
                        "status": "completed",
                        "rows": result.rows,
                        "table_used": table_name,
                        "plotly_spec": plotly_spec,
                        "message": f"Frequency analysis of recommended messages from {table_name}. Found {len(result.rows)} unique message types.",
                        "columns": [frequency_target, "frequency"],  # For insights
                        "data_rows": result.rows[:100]  # Sample for insights
                    }
                    self._send_json_response(response)
                    return
                else:
                    # Fallback: look for any message-related column
                    message_cols = [col for col in columns if "msg" in col.lower() or "message" in col.lower()]
                    if message_cols:
                        frequency_target = message_cols[0]
                        print(f"🔍 Using fallback message column: {frequency_target}")
                    else:
                        self._send_json_response({"error": "No recommended message column found in the table"}, 400)
                        return
            
            elif "provider input" in natural_language.lower():
                # Look for provider input columns
                desc_result = adapter.run(f'DESCRIBE TABLE "{table_name}"')
                if desc_result.error:
                    self._send_json_response({"error": f"Cannot describe table: {desc_result.error}"}, 500)
                    return
                
                columns = [row[0] for row in desc_result.rows]
                
                # Find provider input column
                for col in columns:
                    if "provider" in col.lower() and "input" in col.lower():
                        frequency_target = col
                        break
                    elif "provider_input" in col.lower():
                        frequency_target = col
                        break
                
                if not frequency_target:
                    # Fallback: look for any provider column
                    provider_cols = [col for col in columns if "provider" in col.lower()]
                    if provider_cols:
                        frequency_target = provider_cols[0]
                        print(f"🔍 Using fallback provider column: {frequency_target}")
                    else:
                        self._send_json_response({"error": "No provider input column found in the table"}, 400)
                        return
            
            # If no specific target found, try to find a suitable column for frequency analysis
            if not frequency_target:
                # Try to find a suitable column for frequency analysis
                desc_result = adapter.run(f'DESCRIBE TABLE "{table_name}"')
                if desc_result.error:
                    self._send_json_response({"error": f"Cannot describe table: {desc_result.error}"}, 500)
                    return
                
                columns = [row[0] for row in desc_result.rows]
                
                # Look for likely categorical columns
                for col in columns:
                    if any(keyword in col.lower() for keyword in ['message', 'type', 'category', 'status', 'action', 'recommend']):
                        frequency_target = col
                        break
                
                if not frequency_target and columns:
                    frequency_target = columns[0]  # Use first column as fallback
            
            if frequency_target:
                sql = f'''
                SELECT "{frequency_target}", COUNT(*) as frequency 
                FROM "{table_name}" 
                WHERE "{frequency_target}" IS NOT NULL 
                GROUP BY "{frequency_target}" 
                ORDER BY frequency DESC 
                LIMIT 10
                '''
                
                result = adapter.run(sql)
                if result.error:
                    self._send_json_response({"error": f"Frequency query failed: {result.error}"}, 500)
                    return
                
                # Create visualization
                labels = [str(row[0]) for row in result.rows]
                values = [row[1] for row in result.rows]
                
                plotly_spec = {
                    "data": [{
                        "x": labels,
                        "y": values,
                        "type": "bar",
                        "name": "Frequency",
                        "marker": {"color": "#1f77b4"}
                    }],
                    "layout": {
                        "title": f"Frequency Analysis: {frequency_target}",
                        "xaxis": {"title": frequency_target, "tickangle": -45},
                        "yaxis": {"title": "Count"},
                        "margin": {"l": 60, "r": 30, "t": 60, "b": 120}
                    }
                }
                
                response = {
                    "job_id": job_id,
                    "status": "completed",
                    "rows": result.rows,
                    "table_used": table_name,
                    "plotly_spec": plotly_spec,
                    "message": f"Frequency analysis of {frequency_target} from {table_name}"
                }
                self._send_json_response(response)
            else:
                self._send_json_response({"error": "No suitable column found for frequency analysis"}, 400)
        
        else:
            # Default query - show sample data
            sql = f'SELECT * FROM "{table_name}" LIMIT 10'
            result = adapter.run(sql)
            
            if result.error:
                self._send_json_response({"error": f"Query failed: {result.error}"}, 500)
                return
            
            response = {
                "job_id": job_id,
                "status": "completed",
                "rows": result.rows,
                "table_used": table_name,
                "message": f"Sample data from {table_name}"
            }
            self._send_json_response(response)
    
    def _generate_insights(self, query, location="", data_rows=None, columns=None, table_name=""):
        """Generate intelligent insights using LLM Agent with statistical analysis"""
        
        if not data_rows or not columns:
            return "No data available for insight generation."
        
        try:
            # Use Agent Orchestrator for intelligent insights
            if agent_orchestrator.is_initialized:
                print("🤖 Generating intelligent insights with LLM Agent...")
                
                # Prepare data analysis summary
                data_analysis = {
                    "rows": data_rows,
                    "columns": columns,
                    "table_name": table_name,
                    "location": location
                }
                
                # Generate insights using the agent
                insights = agent_orchestrator.generate_intelligent_insights(
                    query, data_analysis, table_name
                )
                
                return insights
            
        except Exception as e:
            print(f"⚠️ Agent insight generation failed: {e}")
            print("🔄 Falling back to statistical analysis...")
        
        # Fallback to statistical analysis
        return self._fallback_insights_generation(query, data_rows, columns, table_name)
    
    def _fallback_insights_generation(self, query, data_rows, columns, table_name):
        """Fallback insights generation using pandas statistical analysis"""
        insights = []
        
        try:
            # Convert data to analyzable format
            import pandas as pd
            import numpy as np
            
            # Create DataFrame from the data
            df = pd.DataFrame(data_rows, columns=columns)
            
            print(f"🔍 Analyzing data: {len(df)} rows, {len(df.columns)} columns")
            
            # Basic data insights
            insights.append(f"📊 Dataset Analysis for {table_name}:")
            insights.append(f"• Contains {len(df)} records with {len(df.columns)} attributes")
            
            # Analyze data types and completeness
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                insights.append(f"• Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}")
                
                # Statistical analysis for numeric data
                for col in numeric_cols[:2]:  # Analyze first 2 numeric columns
                    series = pd.to_numeric(df[col], errors='coerce')
                    if not series.isna().all():
                        mean_val = series.mean()
                        std_val = series.std()
                        insights.append(f"• {col}: Mean={mean_val:.2f}, Std Dev={std_val:.2f}")
                        
                        # Outlier detection
                        outliers = series[(series > mean_val + 2*std_val) | (series < mean_val - 2*std_val)]
                        if len(outliers) > 0:
                            insights.append(f"• {col}: Found {len(outliers)} potential outliers")
            
            if text_cols:
                insights.append(f"• Found {len(text_cols)} text columns: {', '.join(text_cols[:3])}{'...' if len(text_cols) > 3 else ''}")
                
                # Categorical analysis
                for col in text_cols[:2]:  # Analyze first 2 text columns
                    unique_vals = df[col].nunique()
                    if unique_vals < len(df) * 0.8:  # If less than 80% unique, it's categorical
                        insights.append(f"• {col}: {unique_vals} unique values (categorical)")
                        
                        # Most common values
                        top_values = df[col].value_counts().head(3)
                        if len(top_values) > 0:
                            top_str = ", ".join([f"{val}({count})" for val, count in top_values.items()])
                            insights.append(f"• {col} top values: {top_str}")
            
            # Data quality insights
            null_counts = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if len(cols_with_nulls) > 0:
                insights.append(f"• Data Quality: {len(cols_with_nulls)} columns have missing values")
            
            # Duplicate analysis
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                insights.append(f"• Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)")
            
            # Query-specific insights
            query_lower = query.lower()
            if 'frequency' in query_lower:
                insights.append("💡 For frequency analysis, consider grouping by categorical columns")
            elif 'trend' in query_lower:
                insights.append("💡 For trend analysis, look for time-based columns")
            elif 'top' in query_lower or 'highest' in query_lower:
                insights.append("💡 For ranking analysis, sort by numeric columns")
            
            # Recommendations
            insights.append("🔍 Recommendations:")
            if len(numeric_cols) > 1:
                insights.append("• Explore correlations between numeric variables")
            if len(text_cols) > 0:
                insights.append("• Use categorical columns for grouping and filtering")
            insights.append("• Consider visualizing patterns with charts and graphs")
            
            return "\n".join(insights)
            
        except Exception as e:
            print(f"❌ Statistical analysis failed: {e}")
            return f"Basic data summary: {len(data_rows)} rows, {len(columns)} columns from {table_name}. Unable to perform detailed analysis."


class NBAHandler(BaseHTTPRequestHandler):
                    try:
                        col_data = pd.to_numeric(df[col], errors='coerce')
                        if not col_data.isna().all():
                            mean_val = col_data.mean()
                            std_val = col_data.std()
                            insights.append(f"{col}: Mean = {mean_val:.2f}, Std Dev = {std_val:.2f}")
                            
                            # Outlier detection
                            q1, q3 = col_data.quantile([0.25, 0.75])
                            iqr = q3 - q1
                            outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
                            if len(outliers) > 0:
                                insights.append(f"{col}: {len(outliers)} potential outliers detected.")
                    except Exception as e:
                        continue
            
            if text_cols:
                insights.append(f"Found {len(text_cols)} text columns: {', '.join(text_cols[:3])}{'...' if len(text_cols) > 3 else ''}")
                
                # Categorical analysis
                for col in text_cols[:2]:  # Analyze first 2 text columns
                    try:
                        unique_count = df[col].nunique()
                        total_count = len(df[col].dropna())
                        if total_count > 0:
                            diversity = unique_count / total_count
                            insights.append(f"{col}: {unique_count} unique values ({diversity:.2%} diversity)")
                            
                            # Most frequent values
                            top_values = df[col].value_counts().head(3)
                            if len(top_values) > 0:
                                top_value = top_values.index[0]
                                top_count = top_values.iloc[0]
                                insights.append(f"Most common {col}: '{top_value}' appears {top_count} times ({top_count/total_count:.1%})")
                    except Exception as e:
                        continue
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            cols_with_missing = missing_data[missing_data > 0]
            if len(cols_with_missing) > 0:
                total_missing = cols_with_missing.sum()
                insights.append(f"Data completeness: {total_missing} missing values across {len(cols_with_missing)} columns")
            else:
                insights.append("Data completeness: No missing values detected")
            
            # Query-specific insights
            query_lower = query.lower()
            
            if 'frequency' in query_lower:
                # Find the column that was likely used for frequency analysis
                freq_col = None
                for col in text_cols:
                    if any(keyword in col.lower() for keyword in ['message', 'recommend', 'type', 'category']):
                        freq_col = col
                        break
                
                if freq_col and freq_col in df.columns:
                    value_counts = df[freq_col].value_counts()
                    if len(value_counts) > 0:
                        total_categories = len(value_counts)
                        most_common = value_counts.index[0]
                        most_common_pct = (value_counts.iloc[0] / len(df)) * 100
                        insights.append(f"Frequency Analysis: '{most_common}' is the most frequent category ({most_common_pct:.1f}% of data)")
                        
                        # Distribution analysis
                        if total_categories > 5:
                            top_5_pct = (value_counts.head(5).sum() / len(df)) * 100
                            insights.append(f"Top 5 categories represent {top_5_pct:.1f}% of all data")
            
            if 'nba' in query_lower:
                # NBA-specific insights
                insights.append("NBA Data Analysis: This appears to be basketball-related performance or analytics data")
                
                # Look for performance indicators
                performance_keywords = ['score', 'point', 'rating', 'percentage', 'efficiency']
                perf_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in performance_keywords)]
                if perf_cols:
                    insights.append(f"Performance metrics detected: {', '.join(perf_cols[:3])}")
            
            # Data quality insights
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                insights.append(f"Data Quality: {duplicate_rows} duplicate records found ({duplicate_rows/len(df)*100:.1f}%)")
            
            # Trend analysis for datetime columns
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols:
                insights.append(f"Time-series data detected in {len(date_cols)} columns - trend analysis possible")
            
            # Correlation insights for numeric data
            if len(numeric_cols) >= 2:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    high_corr = []
                    for i in range(len(numeric_cols)):
                        for j in range(i+1, len(numeric_cols)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:  # Strong correlation
                                high_corr.append((numeric_cols[i], numeric_cols[j], corr_val))
                    
                    if high_corr:
                        col1, col2, corr_val = high_corr[0]
                        insights.append(f"Strong correlation detected: {col1} vs {col2} (r={corr_val:.2f})")
                except Exception as e:
                    pass
            
            # Recommendations based on data characteristics
            recommendations = []
            
            if len(numeric_cols) > 0:
                recommendations.append("Consider scatter plots or correlation analysis for numeric relationships")
            
            if len(text_cols) > 0 and any('time' in col.lower() or 'date' in col.lower() for col in text_cols):
                recommendations.append("Time-series visualization could reveal temporal patterns")
            
            if len(df) > 1000:
                recommendations.append("Large dataset - consider sampling or aggregation for better visualization")
            
            if recommendations:
                insights.append("Recommendations: " + "; ".join(recommendations))
            
        except Exception as e:
            print(f"❌ Error in insight generation: {e}")
            insights.append(f"Basic analysis: {len(data_rows)} records retrieved from {table_name}")
            insights.append("Consider exploring data patterns, distributions, or relationships between variables")
        
        return " | ".join(insights) if insights else "Data retrieved successfully - consider exploring patterns and relationships."
    
    def _send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self._send_cors_headers()
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
        print(f"🔍 FAISS Search: {agent_status['faiss_index_ready']}")
    else:
        print(f"⚠️ AI Agents: Limited Mode")
    
    print(f"📊 Ready for NBA table queries with intelligent suggestions")
    print(f"🔗 Test endpoints:")
    print(f"   GET  http://localhost:{port}/health")
    print(f"   GET  http://localhost:{port}/test-nba")
    print(f"   GET  http://localhost:{port}/tables")
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
    run_server()
