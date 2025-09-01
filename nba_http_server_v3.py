#!/usr/bin/env python3
"""
Simp    def do_GET(self):
        print(f"🔍 Received GET request to: {self.path}")
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':TTP server for NBA queries using http.server
"""
import sys
from pathlib import Path
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.parse

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv()

# Initialize database
from db.engine import get_adapter
adapter = get_adapter()

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
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self._send_json_response({"status": "NBA Query Agent Ready", "database": "Snowflake"})
        
        elif parsed_path.path == '/health':
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
                
                # Process NBA query
                print(f"🔍 Processing query: {natural_language}")
                
                # Default simple query for testing
                if not natural_language.strip():
                    natural_language = "show top 10 records"
                
                # Handle general NBA queries
                if "top" in natural_language.lower() and ("5" in natural_language or "10" in natural_language):
                    limit = "5" if "5" in natural_language else "10"
                    sql = f'SELECT * FROM "Final_NBA_Output_python_20250519" LIMIT {limit}'
                    
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
                        "plotly_spec": None,
                        "message": f"Retrieved top {limit} NBA records"
                    }
                    print(f"✅ Query successful, returning {len(result.rows)} rows")
                    self._send_json_response(response)
                    return
                
                elif "Final_NBA_Output_python_20250519" in natural_language and "frequency" in natural_language.lower():
                    if "recommend" in natural_language.lower() and "message" in natural_language.lower():
                        sql = '''
                        SELECT "Recommended_Msg_Overall", COUNT(*) as frequency 
                        FROM "Final_NBA_Output_python_20250519" 
                        WHERE "Recommended_Msg_Overall" != '{}' 
                        GROUP BY "Recommended_Msg_Overall" 
                        ORDER BY frequency DESC 
                        LIMIT 10
                        '''
                        title = "Recommended Messages Frequency"
                        x_title = "Recommended Message"
                    else:
                        sql = '''
                        SELECT "Marketing_Action_Adj", COUNT(*) as frequency 
                        FROM "Final_NBA_Output_python_20250519" 
                        GROUP BY "Marketing_Action_Adj" 
                        ORDER BY frequency DESC 
                        LIMIT 10
                        '''
                        title = "Marketing Actions Frequency"
                        x_title = "Marketing Action"
                    
                    # Execute query
                    result = adapter.run(sql)
                    
                    if result.error:
                        self._send_json_response({"error": f"Query failed: {result.error}"}, 500)
                        return
                    
                    # Create visualization
                    labels = [row[0] for row in result.rows]
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
                            "title": title,
                            "xaxis": {"title": x_title, "tickangle": -45},
                            "yaxis": {"title": "Count"},
                            "margin": {"l": 100, "r": 50, "t": 100, "b": 150}
                        }
                    }
                    
                    response = {
                        "sql": sql,
                        "rows": result.rows,
                        "row_count": len(result.rows),
                        "execution_time": result.execution_time,
                        "plotly_spec": plotly_spec,
                        "success": True,
                        "job_id": job_id
                    }
                    
                    self._send_json_response(response)
                
                else:
                    self._send_json_response({"error": "Query not supported. Please use NBA table with frequency analysis."}, 400)
                    
            except Exception as e:
                self._send_json_response({"error": str(e)}, 500)
        
        else:
            self._send_json_response({"error": "Not found"}, 404)
    
    def _send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

def run_server(port=8003):
    server_address = ('', port)
    httpd = HTTPServer(server_address, NBAHandler)
    print(f"🚀 NBA Query Agent running on http://localhost:{port}")
    print(f"✅ Database: {type(adapter).__name__}")
    print(f"📊 Ready for NBA table queries with visualization")
    print(f"🔗 Test endpoints:")
    print(f"   GET  http://localhost:{port}/health")
    print(f"   GET  http://localhost:{port}/test-nba")
    print(f"   POST http://localhost:{port}/query")
    print("\n📝 Example query:")
    print('   {"natural_language": "read table Final_NBA_Output_python_20250519 and create a visualization with frequency of recommended message and provider input", "job_id": "test001"}')
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
