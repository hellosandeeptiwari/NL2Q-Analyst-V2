"""
Semantic Dictionary - Business language understanding and mapping
Handles business synonyms, domain terminology, and natural language analysis
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class SemanticMapping:
    original_term: str
    mapped_term: str
    confidence: float
    context: str
    domain: str

@dataclass
class QueryAnalysis:
    intent: str
    entities: List[str]
    filters: List[Dict[str, Any]]
    aggregations: List[str]
    time_dimension: Optional[str]
    output_format: str
    complexity_score: float
    reasoning_steps: List[str]

class SemanticDictionary:
    """
    Advanced semantic understanding for business domain queries
    Maps natural language to database concepts with domain awareness
    """
    
    def __init__(self):
        # Initialize OpenAI client for semantic matching (if API key available)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=api_key)
        else:
            self.openai_client = None
        
        # Enhanced domain-specific business glossary
        self.business_glossary = {
            # Healthcare/Pharma Core Terms
            "hcp": "healthcare provider",
            "nbrx": "new brand prescriptions", 
            "trx": "total prescriptions",
            "msl": "medical science liaison",
            "kol": "key opinion leader",
            "writers": "prescribing physicians",
            "lapsed": "discontinued patients",
            "adherence": "medication adherence",
            "persistence": "therapy persistence",
            "formulary": "insurance formulary status",
            "indication": "therapeutic indication",
            "specialists": "specialty physicians",
            "primary_care": "primary care physicians",
            "territory": "sales territory",
            "market_share": "market share percentage",
            "payer": "insurance provider",
            
            # Financial
            "revenue": "total income from sales",
            "cogs": "cost of goods sold",
            "ebitda": "earnings before interest, taxes, depreciation, amortization",
            "arr": "annual recurring revenue",
            "mrr": "monthly recurring revenue",
            "ltv": "lifetime value",
            "cac": "customer acquisition cost",
            
            # Sales/CRM
            "lead": "potential customer",
            "opportunity": "qualified sales prospect",
            "pipeline": "sales opportunities in progress",
            "quota": "sales target",
            "territory": "sales geographical area",
            
            # Marketing
            "impression": "ad view",
            "ctr": "click-through rate",
            "conversion": "desired action completion",
            "attribution": "credit assignment to marketing touchpoints",
            "cohort": "group of users with shared characteristics"
        }
        
        # Intent patterns
        self.intent_patterns = {
            "ranking": ["top", "best", "highest", "lowest", "rank", "leader"],
            "comparison": ["compare", "vs", "versus", "against", "difference"],
            "trend": ["trend", "over time", "growth", "decline", "change"],
            "distribution": ["breakdown", "split", "by", "across", "segment"],
            "aggregation": ["total", "sum", "average", "count", "mean"],
            "forecasting": ["predict", "forecast", "future", "projection"],
            "anomaly": ["unusual", "outlier", "anomaly", "spike", "drop"]
        }
        
        # Time expressions
        self.time_expressions = {
            "last week": "DATE >= CURRENT_DATE - INTERVAL '7 days'",
            "last month": "DATE >= CURRENT_DATE - INTERVAL '1 month'", 
            "last quarter": "DATE >= CURRENT_DATE - INTERVAL '3 months'",
            "last year": "DATE >= CURRENT_DATE - INTERVAL '1 year'",
            "ytd": "DATE >= DATE_TRUNC('year', CURRENT_DATE)",
            "mtd": "DATE >= DATE_TRUNC('month', CURRENT_DATE)",
            "qtd": "DATE >= DATE_TRUNC('quarter', CURRENT_DATE)",
            "this week": "DATE >= DATE_TRUNC('week', CURRENT_DATE)",
            "this month": "DATE >= DATE_TRUNC('month', CURRENT_DATE)",
            "this quarter": "DATE >= DATE_TRUNC('quarter', CURRENT_DATE)",
            "this year": "DATE >= DATE_TRUNC('year', CURRENT_DATE)"
        }
        
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive semantic analysis of natural language query
        """
        
        # Step 1: Basic intent classification
        intent = await self._classify_intent(query)
        
        # Step 2: Entity extraction
        entities = await self._extract_entities(query)
        
        # Step 3: Filter identification
        filters = await self._identify_filters(query)
        
        # Step 4: Aggregation detection
        aggregations = await self._detect_aggregations(query)
        
        # Step 5: Time dimension extraction
        time_dimension = await self._extract_time_dimension(query)
        
        # Step 6: Output format preference
        output_format = await self._determine_output_format(query)
        
        # Step 7: Complexity scoring
        complexity_score = await self._calculate_complexity(query, entities, filters, aggregations)
        
        # Step 8: Generate reasoning steps
        reasoning_steps = await self._generate_reasoning_steps(query, intent, entities, filters)
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            filters=filters,
            aggregations=aggregations,
            time_dimension=time_dimension,
            output_format=output_format,
            complexity_score=complexity_score,
            reasoning_steps=reasoning_steps
        )
    
    async def analyze_query_with_reasoning(self, query: str, model: str = "o3-mini", temperature: float = 0.1) -> QueryAnalysis:
        """
        Enhanced query analysis using reasoning models for complex planning
        """
        prompt = f"""
        You are an expert business intelligence analyst. Analyze this natural language query with deep reasoning:
        
        Query: "{query}"
        
        Provide a comprehensive analysis and return ONLY a valid JSON object with these exact keys:
        {{
            "intent": "primary business objective (e.g., ranking, aggregation, filtering, comparison, trend_analysis)",
            "entities": ["list of data entities, tables, metrics, dimensions mentioned"],
            "filters": [{{ "type": "filter_type", "value": "filter_value" }}],
            "aggregations": ["list of required aggregations like count, sum, avg, frequency"],
            "time_dimension": "time period or null if none",
            "output_format": "preferred format: table, chart, visualization, etc.",
            "complexity_score": 5.5,
            "reasoning_steps": ["step 1 reasoning", "step 2 reasoning", "step 3 reasoning"]
        }}
        
        IMPORTANT: 
        - Return ONLY valid JSON, no additional text or explanations
        - Keep reasoning_steps concise (max 3-4 steps)
        - Ensure the JSON is complete and properly closed
        - Do not exceed reasonable length for each field
        """
        
        if not self.openai_client:
            # Return basic interpretation without AI enhancement
            return {
                "domain": "pharmaceutical",
                "intent": "data_analysis", 
                "entities": [],
                "confidence": 0.6,
                "time_dimension": None,
                "output_format": "table",
                "complexity_score": 5.0,
                "reasoning_steps": ["Basic pattern matching without AI"]
            }
        
        try:
            import asyncio
            
            # Check if this is a reasoning model (o1, o3 series) that has different parameters
            if model.startswith(('o1', 'o3')):
                # Reasoning models don't support temperature or max_tokens, use max_completion_tokens with timeout
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=2000  # Reduced for faster response
                    ),
                    timeout=30.0  # 30 second timeout for reasoning models
                )
            else:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                        temperature=temperature
                    ),
                    timeout=15.0  # 15 second timeout for regular models
                )
            
            content = response.choices[0].message.content.strip()
            
            # Enhanced JSON extraction - handle various response formats
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Try to extract JSON if it's embedded in text
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]
            
            print(f"🔍 Reasoning model response: {content[:200]}...")  # Debug log
            
            try:
                analysis_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing failed: {e}")
                print(f"Raw response: {content}")
                # Extract information from the partial JSON response
                analysis_data = self._extract_from_partial_json(content)
            
            return QueryAnalysis(
                intent=analysis_data.get("intent", "unknown"),
                entities=analysis_data.get("entities", []),
                filters=analysis_data.get("filters", []),
                aggregations=analysis_data.get("aggregations", []),
                time_dimension=analysis_data.get("time_dimension"),
                output_format=analysis_data.get("output_format", "table"),
                complexity_score=analysis_data.get("complexity_score", 5.0),
                reasoning_steps=analysis_data.get("reasoning_steps", [])
            )
            
        except asyncio.TimeoutError:
            print(f"Warning: Reasoning model {model} timed out, falling back to fast analysis")
            # Fallback to standard analysis with gpt-4o-mini
            return await self.analyze_query(query)
        except Exception as e:
            print(f"Error in reasoning model analysis: {e}")
            # Fallback to standard analysis
            return await self.analyze_query(query)
    
    def _extract_from_partial_json(self, partial_content: str) -> Dict[str, Any]:
        """Extract structured information from partial JSON response"""
        import re
        
        # Extract intent
        intent_match = re.search(r'"intent":\s*"([^"]*)"', partial_content)
        intent = intent_match.group(1) if intent_match else "data_analysis"
        
        # Extract entities (handle both simple arrays and nested objects)
        entities = []
        if '"tables":' in partial_content:
            table_match = re.search(r'"tables":\s*\[(.*?)\]', partial_content)
            if table_match:
                tables = [t.strip(' "') for t in table_match.group(1).split(',')]
                entities.extend(tables)
        
        if '"columns":' in partial_content:
            col_match = re.search(r'"columns":\s*\[(.*?)\]', partial_content)
            if col_match:
                columns = [c.strip(' "') for c in col_match.group(1).split(',')]
                entities.extend(columns)
        
        # If no structured entities found, extract from entities array
        if not entities:
            entity_match = re.search(r'"entities":\s*\[(.*?)\]', partial_content)
            if entity_match:
                entities = [e.strip(' "') for e in entity_match.group(1).split(',')]
        
        # Extract filters
        filters = []
        if 'top 5' in partial_content.lower():
            filters.append({"type": "limit", "value": 5})
        
        # Extract aggregations
        aggregations = []
        if 'frequency' in partial_content.lower():
            aggregations.append("frequency")
        if 'count' in partial_content.lower():
            aggregations.append("count")
            
        return {
            "intent": intent,
            "entities": entities,
            "filters": filters,
            "aggregations": aggregations,
            "time_dimension": None,
            "output_format": "visualization",
            "complexity_score": 7.0,
            "reasoning_steps": [f"Extracted from reasoning model: {intent}"]
        }
    
    async def map_business_terms(self, terms: List[str]) -> List[SemanticMapping]:
        """
        Map business terms to database concepts
        """
        
        mappings = []
        
        for term in terms:
            term_lower = term.lower()
            
            # Direct glossary lookup
            if term_lower in self.business_glossary:
                mapping = SemanticMapping(
                    original_term=term,
                    mapped_term=self.business_glossary[term_lower],
                    confidence=0.95,
                    context="direct_glossary_match",
                    domain="business"
                )
                mappings.append(mapping)
                continue
            
            # Fuzzy matching for partial matches
            best_match = None
            best_score = 0.0
            
            for glossary_term, definition in self.business_glossary.items():
                # Simple similarity scoring
                if term_lower in glossary_term or glossary_term in term_lower:
                    score = min(len(term_lower), len(glossary_term)) / max(len(term_lower), len(glossary_term))
                    if score > best_score:
                        best_score = score
                        best_match = (glossary_term, definition)
            
            if best_match and best_score > 0.6:
                mapping = SemanticMapping(
                    original_term=term,
                    mapped_term=best_match[1],
                    confidence=best_score,
                    context="fuzzy_match",
                    domain="business"
                )
                mappings.append(mapping)
            else:
                # Use LLM for unknown terms
                llm_mapping = await self._llm_term_mapping(term)
                if llm_mapping:
                    mappings.append(llm_mapping)
        
        return mappings
    
    async def _classify_intent(self, query: str) -> str:
        """Classify the primary intent of the query"""
        
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return "general_inquiry"
        
        # Return intent with highest score
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    async def _extract_entities(self, query: str) -> List[str]:
        """Extract business entities from the query"""
        
        entities = []
        
        # Use regex patterns for common entities
        entity_patterns = {
            "customer": r"\b(customer|client|account|user)s?\b",
            "product": r"\b(product|item|sku|service)s?\b",
            "sales": r"\b(sales|revenue|income|orders?)s?\b",
            "employee": r"\b(employee|staff|rep|agent|salesperson)s?\b",
            "region": r"\b(region|territory|area|zone|geography)s?\b",
            "time": r"\b(day|week|month|quarter|year|time|date)s?\b"
        }
        
        for entity_type, pattern in entity_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                entities.append(entity_type)
        
        # Extract quoted entities (explicit mentions)
        quoted_entities = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_entities)
        
        # Extract capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        entities.extend(proper_nouns)
        
        return list(set(entities))
    
    async def _identify_filters(self, query: str) -> List[Dict[str, Any]]:
        """Identify filters and constraints in the query"""
        
        filters = []
        
        # Geographic filters
        geo_pattern = r'\bin\s+([A-Z][a-zA-Z\s]+)'
        geo_matches = re.findall(geo_pattern, query)
        for match in geo_matches:
            filters.append({
                "type": "geographic",
                "field": "location",
                "operator": "equals",
                "value": match.strip()
            })
        
        # Numeric constraints
        numeric_patterns = [
            (r'>\s*(\d+)', "greater_than"),
            (r'<\s*(\d+)', "less_than"),
            (r'=\s*(\d+)', "equals"),
            (r'top\s+(\d+)', "limit"),
            (r'first\s+(\d+)', "limit"),
            (r'bottom\s+(\d+)', "limit_asc")
        ]
        
        for pattern, operator in numeric_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                filters.append({
                    "type": "numeric",
                    "operator": operator,
                    "value": int(match)
                })
        
        # Time-based filters
        for time_expr, sql_condition in self.time_expressions.items():
            if time_expr in query.lower():
                filters.append({
                    "type": "temporal",
                    "expression": time_expr,
                    "sql_condition": sql_condition
                })
        
        return filters
    
    async def _detect_aggregations(self, query: str) -> List[str]:
        """Detect aggregation functions needed"""
        
        aggregations = []
        query_lower = query.lower()
        
        agg_patterns = {
            "count": ["count", "number of", "how many", "total number"],
            "sum": ["sum", "total", "add up"],
            "average": ["average", "avg", "mean"],
            "maximum": ["max", "maximum", "highest", "top"],
            "minimum": ["min", "minimum", "lowest", "bottom"],
            "distinct": ["unique", "distinct", "different"]
        }
        
        for agg_type, patterns in agg_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                aggregations.append(agg_type)
        
        return aggregations
    
    async def _extract_time_dimension(self, query: str) -> Optional[str]:
        """Extract time dimension for analysis"""
        
        time_dimensions = {
            "daily": ["daily", "day", "per day"],
            "weekly": ["weekly", "week", "per week"],
            "monthly": ["monthly", "month", "per month"],
            "quarterly": ["quarterly", "quarter", "per quarter"],
            "yearly": ["yearly", "year", "annual", "per year"]
        }
        
        query_lower = query.lower()
        
        for dimension, patterns in time_dimensions.items():
            if any(pattern in query_lower for pattern in patterns):
                return dimension
        
        # Check for "over time" pattern
        if "over time" in query_lower or "trend" in query_lower:
            return "time_series"
        
        return None
    
    async def _determine_output_format(self, query: str) -> str:
        """Determine preferred output format"""
        
        query_lower = query.lower()
        
        format_indicators = {
            "chart": ["chart", "graph", "plot", "visualize", "show"],
            "table": ["table", "list", "rows", "data"],
            "summary": ["summary", "overview", "brief", "highlight"],
            "report": ["report", "analysis", "detailed"]
        }
        
        for format_type, indicators in format_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return format_type
        
        # Default based on query type
        if any(word in query_lower for word in ["trend", "over time", "growth"]):
            return "chart"
        elif any(word in query_lower for word in ["top", "list", "breakdown"]):
            return "table"
        else:
            return "mixed"  # Both table and chart
    
    async def _calculate_complexity(self, query: str, entities: List[str], filters: List[Dict], aggregations: List[str]) -> float:
        """Calculate query complexity score (0-1)"""
        
        complexity = 0.0
        
        # Base complexity from query length
        complexity += min(len(query.split()) / 100, 0.2)
        
        # Entity complexity
        complexity += min(len(entities) / 10, 0.2)
        
        # Filter complexity
        complexity += min(len(filters) / 10, 0.2)
        
        # Aggregation complexity
        complexity += min(len(aggregations) / 5, 0.2)
        
        # Special complexity indicators
        if any(word in query.lower() for word in ["join", "merge", "combine"]):
            complexity += 0.1
        
        if any(word in query.lower() for word in ["forecast", "predict", "model"]):
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    async def _generate_reasoning_steps(self, query: str, intent: str, entities: List[str], filters: List[Dict]) -> List[str]:
        """Generate step-by-step reasoning for the query"""
        
        steps = []
        
        # Step 1: Understanding
        steps.append(f"Understanding the query: '{query}' - Primary intent is {intent}")
        
        # Step 2: Entity identification
        if entities:
            steps.append(f"Identified key entities: {', '.join(entities)}")
        
        # Step 3: Data requirements
        steps.append("Determining data requirements and relevant tables")
        
        # Step 4: Filters and constraints
        if filters:
            filter_desc = []
            for f in filters:
                if f["type"] == "temporal":
                    filter_desc.append(f"time filter: {f['expression']}")
                elif f["type"] == "numeric":
                    filter_desc.append(f"numeric constraint: {f['operator']} {f['value']}")
                elif f["type"] == "geographic":
                    filter_desc.append(f"location filter: {f['value']}")
            
            if filter_desc:
                steps.append(f"Applying filters: {', '.join(filter_desc)}")
        
        # Step 5: Query generation
        steps.append("Generating optimized SQL query with appropriate joins and aggregations")
        
        # Step 6: Validation
        steps.append("Validating query for safety, performance, and data governance")
        
        # Step 7: Execution
        steps.append("Executing query with timeout and row limits")
        
        # Step 8: Visualization
        if intent in ["trend", "comparison", "distribution"]:
            steps.append("Generating appropriate visualizations for the results")
        
        return steps
    
    async def _llm_term_mapping(self, term: str) -> Optional[SemanticMapping]:
        """Use LLM to map unknown business terms"""
        
        if not self.openai_client:
            # Return None if no OpenAI client available
            return None
        
        try:
            prompt = f"""
            You are a business analyst. Please provide a clear, concise definition for this business term: "{term}"
            
            If this is a common business/industry term, provide the standard definition.
            If it's not a recognized term, return "unknown".
            
            Format your response as: DEFINITION: [your definition]
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            if "DEFINITION:" in content and "unknown" not in content.lower():
                definition = content.split("DEFINITION:")[1].strip()
                return SemanticMapping(
                    original_term=term,
                    mapped_term=definition,
                    confidence=0.75,
                    context="llm_generated",
                    domain="business"
                )
        
        except Exception as e:
            print(f"Error in LLM term mapping: {e}")
        
        return None
    
    def get_business_context(self, domain: str = "general") -> Dict[str, Any]:
        """Get business context for specific domain"""
        
        context = {
            "glossary_size": len(self.business_glossary),
            "supported_intents": list(self.intent_patterns.keys()),
            "time_expressions": list(self.time_expressions.keys()),
            "domain": domain
        }
        
        # Domain-specific context
        if domain == "healthcare":
            context["key_terms"] = ["hcp", "nbrx", "trx", "msl", "kol"]
        elif domain == "financial":
            context["key_terms"] = ["revenue", "ebitda", "arr", "ltv", "cac"]
        elif domain == "sales":
            context["key_terms"] = ["lead", "opportunity", "pipeline", "quota"]
        
        return context
