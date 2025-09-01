"""
LLM Agent for intelligent task handling with context and memory
Uses OpenAI GPT models for sophisticated query understanding and insights
"""
import openai
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

class LLMAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
        
        self.context_memory = []
        self.max_context_length = 10  # Keep last 10 interactions
        
        # Agent personas for different tasks
        self.personas = {
            'table_selector': {
                'role': 'You are an expert data analyst who specializes in understanding user queries and selecting the most relevant database tables.',
                'instructions': [
                    'Analyze the user query to understand what data they need',
                    'Consider NBA data context when relevant',
                    'Prioritize tables that match the query intent',
                    'Explain your reasoning for table selection'
                ]
            },
            'insight_generator': {
                'role': 'You are a Healthcare/Life-Sciences Commercial Insights Analyst specializing in commercial KPIs and market performance.',
                'instructions': [
                    'Focus on commercial metrics: NPS, TRx/NRx, Share, Channel lift, Patient outcomes, Sales performance',
                    'Analyze healthcare market trends and competitive positioning',
                    'Identify opportunities for commercial growth and optimization',
                    'Provide actionable recommendations for healthcare commercial teams',
                    'Consider regulatory compliance and patient safety in all insights'
                ]
            },
            'query_planner': {
                'role': 'You are a SQL expert who plans optimal database queries based on user requirements.',
                'instructions': [
                    'Understand the user\'s data requirements',
                    'Plan the most efficient query strategy',
                    'Consider data visualization needs',
                    'Suggest appropriate aggregations and filters'
                ]
            }
        }
    
    def add_to_memory(self, interaction: Dict[str, Any]):
        """Add interaction to context memory"""
        interaction['timestamp'] = datetime.now().isoformat()
        self.context_memory.append(interaction)
        
        # Keep only recent interactions
        if len(self.context_memory) > self.max_context_length:
            self.context_memory = self.context_memory[-self.max_context_length:]
    
    def get_context_summary(self) -> str:
        """Generate a summary of recent context"""
        if not self.context_memory:
            return "No previous context available."
            
        recent_queries = []
        for interaction in self.context_memory[-3:]:  # Last 3 interactions
            if 'query' in interaction:
                recent_queries.append(f"- {interaction['query']}")
            if 'action' in interaction:
                recent_queries.append(f"  Action: {interaction['action']}")
                
        return "Recent context:\n" + "\n".join(recent_queries)
    
    def analyze_column_requirements(self, query: str, table_name: str, available_columns: List[str]) -> Dict[str, Any]:
        """Use LLM to analyze column requirements and suggest alternatives"""
        if not self.api_key:
            return self._fallback_column_analysis(query, table_name, available_columns)
            
        try:
            prompt = f"""
You are a Healthcare/Life-Sciences Commercial Insights Analyst helping to execute data analysis tasks automatically.

GOAL: Help users accomplish their analysis goals with minimal friction. Be DECISIVE and ACTION-ORIENTED.

Table: {table_name}
Available columns:
{chr(10).join(['- ' + col for col in available_columns])}

User Query: "{query}"

Your task is to find the best way to execute the user's request immediately:

1. Understand the user's business/clinical goal
2. Find the best matching columns to accomplish this goal  
3. If there's a clear best match (>80% confidence), recommend immediate execution
4. If medium confidence (60-79%), suggest execution with alternatives shown
5. Only ask for clarification if confidence is low (<60%) 
6. Focus on healthcare/commercial outcomes (NPS, TRx, NRx, providers, patients, satisfaction, costs)

Be decisive - users want results, not endless options.

Respond in JSON format:
{{
    "requested_columns": ["columns user mentioned or needs for their goal"],
    "column_matches": [
        {{
            "requested": "what user wants to analyze",
            "matched": "best_column_to_execute_with",
            "confidence": 0.85,
            "reason": "why this executes the user's business goal",
            "auto_execute": true
        }}
    ],
    "execution_strategy": {{
        "primary_column": "column_to_use_immediately",
        "analysis_type": "frequency/trend/comparison/aggregation",
        "confidence": 0.85,
        "reasoning": "why this approach fulfills the business need"
    }},
    "suggested_alternatives": [
        {{
            "column": "alternative_column",
            "kpi_type": "NPS/TRx/Patient Outcome/Provider Performance/Market Share",
            "business_value": "specific commercial insight this provides"
        }}
    ],
    "user_goal": "the business/clinical objective to achieve",
    "recommended_action": "execute analysis with [column] / clarification needed because [specific reason]",
    "commercial_impact": "how this analysis drives commercial success"
}}
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if '```json' in content:
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            elif '```' in content:
                content = content.replace('```', '').strip()
            
            import json
            try:
                analysis = json.loads(content)
                return analysis
            except json.JSONDecodeError as e:
                print(f"🚨 JSON parsing failed for column analysis: {e}")
                return self._fallback_column_analysis(query, table_name, available_columns)
                
        except Exception as e:
            print(f"🚨 LLM column analysis failed: {e}")
            return self._fallback_column_analysis(query, table_name, available_columns)

    def _fallback_column_analysis(self, query: str, table_name: str, available_columns: List[str]) -> Dict[str, Any]:
        """Fallback column analysis without LLM"""
        import re
        
        # Extract potential column names from query
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Simple keyword matching
        potential_matches = []
        for word in words:
            for col in available_columns:
                if word in col.lower() or col.lower().find(word) != -1:
                    potential_matches.append({
                        "requested": word,
                        "matched": col,
                        "confidence": 0.7,
                        "reason": f"Keyword '{word}' found in column '{col}'"
                    })
        
        # Remove duplicates
        seen = set()
        unique_matches = []
        for match in potential_matches:
            key = (match["requested"], match["matched"])
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        return {
            "requested_columns": list(set(words)),
            "column_matches": unique_matches[:5],
            "suggested_alternatives": [
                {
                    "column": col,
                    "relevance": "Available column that might be useful",
                    "example_usage": f"Use for data analysis of {col}"
                }
                for col in available_columns[:5]
            ],
            "query_interpretation": "User wants to analyze data with specific columns",
            "recommended_approach": "Use fuzzy matching to find best column alternatives"
        }

    def analyze_query_intent(self, query: str, available_tables: List[str]) -> Dict[str, Any]:
        """Use LLM to analyze query intent and suggest best tables"""
        if not self.api_key:
            return self._fallback_intent_analysis(query, available_tables)
            
        try:
            context = self.get_context_summary()
            persona = self.personas['table_selector']
            
            prompt = f"""
{persona['role']}

{chr(10).join(persona['instructions'])}

Available tables (first 20 for reference):
{chr(10).join(['- ' + table for table in available_tables[:20]])}

User Query: "{query}"

Context from previous interactions:
{context}

Please analyze this query and provide:
1. Query intent and data requirements
2. Top 3 recommended tables with reasons
3. Query type (aggregation, filtering, visualization, etc.)
4. Suggested data analysis approach

Respond in JSON format:
{{
    "intent": "description of what user wants",
    "data_requirements": ["requirement1", "requirement2"],
    "recommended_tables": [
        {{"table": "table_name", "reason": "why this table", "confidence": "high/medium/low"}}
    ],
    "query_type": "type of query",
    "analysis_approach": "suggested approach"
}}
"""

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Get response content and validate
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                print("⚠️ Empty response from OpenAI")
                return self._fallback_intent_analysis(query, available_tables)
            
            # Try to parse JSON (handle markdown code blocks)
            try:
                # Remove markdown code blocks if present
                json_content = response_content.strip()
                if json_content.startswith('```json'):
                    json_content = json_content[7:]  # Remove ```json
                if json_content.endswith('```'):
                    json_content = json_content[:-3]  # Remove ```
                json_content = json_content.strip()
                
                result = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing failed: {e}")
                print(f"⚠️ Raw response: {response_content}")
                return self._fallback_intent_analysis(query, available_tables)
            
            # Add to memory
            self.add_to_memory({
                'query': query,
                'action': 'intent_analysis',
                'result': result
            })
            
            return result
            
        except Exception as e:
            print(f"⚠️ LLM analysis failed: {e}")
            return self._fallback_intent_analysis(query, available_tables)
    
    def generate_insights(self, query: str, data_summary: Dict[str, Any], table_name: str) -> str:
        """Generate intelligent insights from data analysis"""
        if not self.api_key:
            return self._fallback_insights(data_summary)
            
        try:
            context = self.get_context_summary()
            persona = self.personas['insight_generator']
            
            prompt = f"""
{persona['role']}

{chr(10).join(persona['instructions'])}

Original Query: "{query}"
Table Analyzed: {table_name}

Data Summary:
- Total Rows: {data_summary.get('total_rows', 'Unknown')}
- Columns: {', '.join(data_summary.get('columns', []))}
- Data Sample: {json.dumps(data_summary.get('sample_data', [])[:3], indent=2)}

Statistical Analysis:
{json.dumps(data_summary.get('statistical_analysis', {}), indent=2)}

Context:
{context}

Please provide exactly 5 concise, actionable insights in the following format:

🔍 **TOP 5 COMMERCIAL INSIGHTS**

**1. [Key Finding Title]**
• Summary: [One sentence summary]
• Impact: [Commercial impact/opportunity]
• Action: [Specific recommendation]

**2. [Key Finding Title]**
• Summary: [One sentence summary]
• Impact: [Commercial impact/opportunity]
• Action: [Specific recommendation]

**3. [Key Finding Title]**
• Summary: [One sentence summary]
• Impact: [Commercial impact/opportunity]
• Action: [Specific recommendation]

**4. [Key Finding Title]**
• Summary: [One sentence summary]
• Impact: [Commercial impact/opportunity]
• Action: [Specific recommendation]

**5. [Key Finding Title]**
• Summary: [One sentence summary]
• Impact: [Commercial impact/opportunity]
• Action: [Specific recommendation]

Focus on commercial KPIs, market performance, and actionable business insights. Keep each insight concise and focused on healthcare/life-sciences commercial value.
"""

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            # Get response content and validate
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                print("⚠️ Empty insights response from OpenAI")
                return self._fallback_insights(data_summary)
            
            insights = response_content
            
            # Add to memory
            self.add_to_memory({
                'query': query,
                'action': 'insight_generation',
                'table': table_name,
                'insights': insights[:200] + "..." if len(insights) > 200 else insights
            })
            
            return insights
            
        except Exception as e:
            print(f"⚠️ LLM insight generation failed: {e}")
            return self._fallback_insights(data_summary)
    
    def plan_query_strategy(self, query: str, table_suggestions: List[Dict]) -> Dict[str, Any]:
        """Plan optimal query execution strategy"""
        if not self.api_key:
            return self._fallback_query_planning(query, table_suggestions)
            
        try:
            persona = self.personas['query_planner']
            
            prompt = f"""
{persona['role']}

{chr(10).join(persona['instructions'])}

User Query: "{query}"

Suggested Tables:
{json.dumps(table_suggestions, indent=2)}

Plan the optimal approach:
1. Which table(s) to use and why
2. What aggregations/calculations are needed
3. Filtering requirements
4. Visualization recommendations
5. Query complexity assessment

Respond in JSON format:
{{
    "primary_table": "best_table_name",
    "secondary_tables": ["other_tables"],
    "aggregations": ["aggregation_types"],
    "filters": ["filter_requirements"],
    "visualization_type": "chart_type",
    "complexity": "simple/medium/complex",
    "execution_plan": "step_by_step_approach"
}}
"""

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2
            )
            
            # Get response content and validate
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                print("⚠️ Empty planning response from OpenAI")
                return self._fallback_query_planning(query, table_suggestions)
            
            # Try to parse JSON (handle markdown code blocks)
            try:
                # Remove markdown code blocks if present
                json_content = response_content.strip()
                if json_content.startswith('```json'):
                    json_content = json_content[7:]  # Remove ```json
                if json_content.endswith('```'):
                    json_content = json_content[:-3]  # Remove ```
                json_content = json_content.strip()
                
                result = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"⚠️ Planning JSON parsing failed: {e}")
                print(f"⚠️ Raw response: {response_content}")
                return self._fallback_query_planning(query, table_suggestions)
            
            # Add to memory
            self.add_to_memory({
                'query': query,
                'action': 'query_planning',
                'plan': result
            })
            
            return result
            
        except Exception as e:
            print(f"⚠️ LLM query planning failed: {e}")
            return self._fallback_query_planning(query, table_suggestions)
    
    def _fallback_intent_analysis(self, query: str, available_tables: List[str]) -> Dict[str, Any]:
        """Fallback intent analysis without LLM"""
        query_lower = query.lower()
        
        # Simple keyword-based analysis
        intent = "data_retrieval"
        if any(word in query_lower for word in ['frequency', 'count', 'aggregate']):
            intent = "aggregation_analysis"
        elif any(word in query_lower for word in ['visualization', 'chart', 'plot']):
            intent = "data_visualization"
        elif any(word in query_lower for word in ['top', 'bottom', 'highest', 'lowest']):
            intent = "ranking_analysis"
            
        # Find NBA tables
        nba_tables = [t for t in available_tables if 'nba' in t.lower()][:3]
        
        return {
            "intent": intent,
            "data_requirements": ["NBA data analysis"],
            "recommended_tables": [
                {"table": table, "reason": "Contains NBA data", "confidence": "medium"}
                for table in nba_tables
            ],
            "query_type": intent,
            "analysis_approach": "Basic data analysis with simple aggregations"
        }
    
    def _fallback_insights(self, data_summary: Dict[str, Any]) -> str:
        """Fallback insights without LLM"""
        total_rows = data_summary.get('total_rows', 0)
        columns = data_summary.get('columns', [])
        
        insights = f"""
📊 Data Analysis Summary:
- Dataset contains {total_rows:,} records across {len(columns)} columns
- Columns analyzed: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}

🔍 Key Observations:
- This appears to be a substantial dataset suitable for comprehensive analysis
- Multiple dimensions available for cross-analysis and correlation studies
- Data structure suggests potential for frequency analysis and trend identification

💡 Recommendations:
- Consider filtering data for specific time periods or categories
- Explore correlations between different attributes
- Use aggregation functions for summary statistics
- Implement data visualization for better pattern recognition
"""
        return insights
    
    def _fallback_query_planning(self, query: str, table_suggestions: List[Dict]) -> Dict[str, Any]:
        """Fallback query planning without LLM"""
        primary_table = table_suggestions[0]['table_name'] if table_suggestions else "Unknown"
        
        query_lower = query.lower()
        viz_type = "table"
        if any(word in query_lower for word in ['frequency', 'count']):
            viz_type = "bar_chart"
        elif any(word in query_lower for word in ['trend', 'time']):
            viz_type = "line_chart"
            
        return {
            "primary_table": primary_table,
            "secondary_tables": [],
            "aggregations": ["COUNT", "GROUP BY"],
            "filters": ["IS NOT NULL"],
            "visualization_type": viz_type,
            "complexity": "medium",
            "execution_plan": "Execute query on primary table with basic aggregations"
        }
