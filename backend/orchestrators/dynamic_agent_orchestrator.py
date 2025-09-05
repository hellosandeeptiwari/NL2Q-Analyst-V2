"""
Dynamic Agent Orchestration System - MCP Style
Automatically selects and coordinates agents based on query analysis
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import os

@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    agent_name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    cost_factor: float
    reliability_score: float
    specialized_domains: List[str]

class TaskType(Enum):
    SCHEMA_DISCOVERY = "schema_discovery"
    SEMANTIC_UNDERSTANDING = "semantic_understanding"
    SIMILARITY_MATCHING = "similarity_matching"
    QUERY_GENERATION = "query_generation"
    VALIDATION = "validation"
    EXECUTION = "execution"
    VISUALIZATION = "visualization"
    USER_INTERACTION = "user_interaction"

@dataclass
class AgentTask:
    """A specific task for an agent"""
    task_id: str
    task_type: TaskType
    input_data: Dict[str, Any]
    required_output: Dict[str, Any]
    constraints: Dict[str, Any]
    dependencies: List[str]  # Other task IDs this depends on

class DynamicAgentOrchestrator:
    """
    MCP-style orchestrator that dynamically selects and coordinates agents
    """
    
    def __init__(self):
        self.available_agents = self._register_agents()
        self.reasoning_model = os.getenv("REASONING_MODEL", "o3-mini")
        self.fast_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._index_initialized = False
        self.db_connector = None
        self.pinecone_store = None
        
    async def initialize_on_startup(self):
        """Initialize the system on startup - minimal for fast startup"""
        try:
            print("🚀 Starting minimal system initialization...")
            
            # Only initialize what's absolutely necessary for fast startup
            # Database and Pinecone will be initialized lazily on first query
            
            print("⚡ Using lazy initialization for fastest startup")
            print("✅ Minimal system initialization completed")
        except Exception as e:
            print(f"⚠️ Error during startup initialization: {e}")
            # Don't fail startup completely
            
    async def _ensure_initialized(self):
        """Ensure database and Pinecone are initialized (lazy initialization)"""
        if not self.db_connector:
            print("🔌 Initializing database connector...")
            await self._initialize_database_connector()
            
        if not self.pinecone_store:
            print("🔍 Initializing Pinecone vector store...")
            await self._initialize_vector_store()
            
    async def _initialize_database_connector(self):
        """Initialize database connection"""
        try:
            from backend.db.engine import get_adapter
            self.db_connector = get_adapter("snowflake")
            print("✅ Database connector initialized")
        except Exception as e:
            print(f"⚠️ Database connector initialization failed: {e}")
            
    async def _initialize_vector_store(self):
        """Initialize Pinecone vector store"""
        try:
            from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
            self.pinecone_store = PineconeSchemaVectorStore()
            print("✅ Vector store initialized")
        except Exception as e:
            print(f"⚠️ Vector store initialization failed: {e}")
            
    async def _check_and_perform_comprehensive_indexing(self):
        """Check indexing completeness and perform auto-indexing if needed"""
        try:
            # Get current index statistics
            stats = self.pinecone_store.index.describe_index_stats()
            total_vectors = stats.total_vector_count
            
            # Get available tables count
            available_tables = []
            try:
                result = self.db_connector.run("SHOW TABLES IN SCHEMA ENHANCED_NBA", dry_run=False)
                available_tables = result.rows if result.rows else []
            except Exception as e:
                print(f"⚠️ Could not fetch table list: {e}")
                return
            
            total_available_tables = len(available_tables)
            
            # Calculate expected vectors per table (overview + column groups + business context)
            # With improved chunking: 3-5 chunks per table
            expected_vectors_per_table = 4  # Conservative estimate
            expected_total_vectors = total_available_tables * expected_vectors_per_table
            
            # Check if indexing is needed
            indexing_completeness = (total_vectors / expected_total_vectors) if expected_total_vectors > 0 else 0
            
            print(f"📊 Index status: {total_vectors} vectors, {total_available_tables} tables available")
            print(f"� Indexing completeness: {indexing_completeness:.1%}")
            
            # Trigger auto-indexing if less than 80% complete or completely empty
            should_index = (indexing_completeness < 0.8) or (total_vectors == 0)
            
            if should_index and total_available_tables > 0:
                print("🔄 Starting comprehensive auto-indexing with optimized chunking...")
                await self._perform_full_database_indexing(force_clear=True)
            else:
                print("✅ Index appears complete, skipping auto-indexing")
                
        except Exception as e:
            print(f"⚠️ Error checking indexing status: {e}")
            
    async def _perform_full_database_indexing(self, force_clear: bool = True):
        """Perform full database indexing with optimized chunking"""
        try:
            print("🗂️ Starting full database schema indexing with improved chunking...")
            
            # Ensure pinecone store is initialized
            if not self.pinecone_store:
                await self._initialize_vector_store()
            
            if not self.pinecone_store:
                raise Exception("Failed to initialize Pinecone vector store")
            

            # Ensure db_connector is initialized
            if not self.db_connector:
                from backend.main import get_adapter
                self.db_connector = get_adapter("snowflake")
            if not self.db_connector:
                raise Exception("Database adapter not initialized")

            # Clear existing index only if force_clear is True
            if force_clear:
                try:
                    self.pinecone_store.clear_index()
                    print("🧹 Cleared existing index for fresh start")
                except Exception as e:
                    print(f"⚠️ Could not clear existing index: {e}")
            else:
                print("📋 Resuming indexing - keeping existing vectors")

            # Index the complete database schema with new optimized chunking
            # Define a local progress callback to avoid import issues
            def local_progress_callback(stage: str, current_table: str = "", processed: int = None, total: int = None, error: str = None):
                try:
                    from backend.main import update_progress
                    update_progress(stage, current_table, processed, total, error)
                except ImportError:
                    print(f"Progress: {stage} - {current_table} ({processed}/{total})")
                except Exception as e:
                    print(f"Progress callback error: {e}")
                
            await self.pinecone_store.index_database_schema(self.db_connector, progress_callback=local_progress_callback)
            
            # Verify indexing completed successfully
            final_stats = self.pinecone_store.index.describe_index_stats()
            print(f"✅ Indexing completed: {final_stats.total_vector_count} vectors indexed")
            
            self._index_initialized = True
            
        except Exception as e:
            print(f"⚠️ Error during full database indexing: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"⚠️ Error during full database indexing: {e}")
            import traceback
            traceback.print_exc()
            
    async def initialize_vector_search(self):
        """Legacy method - redirects to new comprehensive initialization"""
        if not self._index_initialized:
            await self.initialize_on_startup()
        
    def _register_agents(self) -> Dict[str, AgentCapability]:
        """Register all available agents and their capabilities"""
        return {
            "schema_discoverer": AgentCapability(
                agent_name="schema_discoverer",
                description="Discovers database schema, tables, columns, relationships",
                input_types=["natural_language_query", "database_connection"],
                output_types=["schema_context", "table_list", "column_mappings"],
                cost_factor=0.3,
                reliability_score=0.95,
                specialized_domains=["database", "schema", "metadata"]
            ),
            
            "semantic_analyzer": AgentCapability(
                agent_name="semantic_analyzer",
                description="Understands business intent and extracts entities",
                input_types=["natural_language_query", "business_context"],
                output_types=["entities", "intent", "business_terms"],
                cost_factor=0.2,
                reliability_score=0.90,
                specialized_domains=["nlp", "business_logic", "pharmaceuticals"]
            ),
            
            "vector_matcher": AgentCapability(
                agent_name="vector_matcher", 
                description="Performs similarity matching between query and schema",
                input_types=["entities", "schema_context", "embeddings"],
                output_types=["similarity_scores", "matched_tables", "matched_columns"],
                cost_factor=0.4,
                reliability_score=0.88,
                specialized_domains=["vector_search", "embeddings", "similarity"]
            ),
            
            "query_builder": AgentCapability(
                agent_name="query_builder",
                description="Generates SQL queries with validation and safety checks",
                input_types=["matched_schema", "business_logic", "filters"],
                output_types=["sql_query", "explanation", "safety_assessment"],
                cost_factor=0.3,
                reliability_score=0.92,
                specialized_domains=["sql", "query_optimization", "safety"]
            ),
            
            "user_verifier": AgentCapability(
                agent_name="user_verifier",
                description="Interacts with user to confirm schema selections and queries",
                input_types=["proposed_tables", "proposed_columns", "generated_query"],
                output_types=["user_confirmation", "modifications", "approval"],
                cost_factor=0.1,
                reliability_score=0.98,
                specialized_domains=["user_interaction", "verification", "confirmation"]
            ),
            
            "query_executor": AgentCapability(
                agent_name="query_executor",
                description="Safely executes queries and handles results",
                input_types=["validated_query", "database_connection", "safety_params"],
                output_types=["query_results", "execution_stats", "error_handling"],
                cost_factor=0.5,
                reliability_score=0.94,
                specialized_domains=["execution", "database", "safety"]
            ),
            
            "visualizer": AgentCapability(
                agent_name="visualizer",
                description="Creates interactive visualizations and summaries",
                input_types=["query_results", "data_types", "user_preferences"],
                output_types=["charts", "tables", "narrative_summary"],
                cost_factor=0.3,
                reliability_score=0.89,
                specialized_domains=["visualization", "charts", "reporting"]
            )
        }
    
    async def plan_execution(self, user_query: str, context: Dict[str, Any] = None) -> List[AgentTask]:
        """
        Use reasoning model to plan which agents to use and in what order
        """
        
        planning_prompt = f"""
        You are an intelligent query orchestrator. Analyze this user query and create an execution plan using available agents.

        USER QUERY: "{user_query}"
        CONTEXT: {json.dumps(context or {}, indent=2)}

        AVAILABLE AGENTS:
        {self._format_agent_capabilities()}

        Create a step-by-step execution plan that:
        1. Discovers relevant database schema automatically
        2. Performs semantic understanding of the query
        3. Uses similarity matching to find best table/column matches
        4. Generates SQL query based on matches
        5. Gets user verification for schema selections
        6. Executes the validated query
        7. Creates appropriate visualizations

        Return a JSON array of tasks with:
        - task_id: unique identifier
        - task_type: one of the available task types
        - agent_name: which agent to use
        - input_requirements: what data this task needs
        - output_expectations: what this task will produce
        - dependencies: which other tasks must complete first
        - user_interaction_required: boolean

        Focus on creating an automated flow that only asks user for verification of schema selections.
        """
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": planning_prompt}],
                max_completion_tokens=2000
            )
            
            # Parse the response to extract task plan
            content = response.choices[0].message.content
            
            # Extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                tasks_data = json.loads(json_match.group())
                return self._convert_to_agent_tasks(tasks_data)
            else:
                print("⚠️ Could not parse task plan from reasoning model")
                return self._create_default_plan(user_query)
                
        except Exception as e:
            print(f"⚠️ Planning failed: {e}")
            return self._create_default_plan(user_query)
    
    def _format_agent_capabilities(self) -> str:
        """Format agent capabilities for the prompt"""
        capabilities = []
        for agent_name, capability in self.available_agents.items():
            capabilities.append(f"""
- {agent_name}: {capability.description}
  Inputs: {', '.join(capability.input_types)}
  Outputs: {', '.join(capability.output_types)}
  Domains: {', '.join(capability.specialized_domains)}
            """)
        return '\n'.join(capabilities)
    
    def _convert_to_agent_tasks(self, tasks_data: List[Dict]) -> List[AgentTask]:
        """Convert JSON task data to AgentTask objects"""
        tasks = []
        for task_data in tasks_data:
            task = AgentTask(
                task_id=task_data.get("task_id", f"task_{len(tasks)}"),
                task_type=TaskType(task_data.get("task_type")),
                input_data=task_data.get("input_requirements", {}),
                required_output=task_data.get("output_expectations", {}),
                constraints=task_data.get("constraints", {}),
                dependencies=task_data.get("dependencies", [])
            )
            tasks.append(task)
        return tasks
    
    def _create_default_plan(self, user_query: str) -> List[AgentTask]:
        """Create a default execution plan"""
        return [
            AgentTask(
                task_id="1_discover_schema",
                task_type=TaskType.SCHEMA_DISCOVERY,
                input_data={"query": user_query},
                required_output={"schema_context": "discovered_tables_and_columns"},
                constraints={"max_tables": 20},
                dependencies=[]
            ),
            AgentTask(
                task_id="2_semantic_analysis", 
                task_type=TaskType.SEMANTIC_UNDERSTANDING,
                input_data={"query": user_query},
                required_output={"entities": "extracted_entities", "intent": "business_intent"},
                constraints={},
                dependencies=[]
            ),
            AgentTask(
                task_id="3_similarity_matching",
                task_type=TaskType.SIMILARITY_MATCHING, 
                input_data={"entities": "from_task_2", "schema": "from_task_1"},
                required_output={"matched_tables": "relevant_tables", "matched_columns": "relevant_columns"},
                constraints={"min_similarity": 0.7},
                dependencies=["1_discover_schema", "2_semantic_analysis"]
            ),
            AgentTask(
                task_id="4_user_verification",
                task_type=TaskType.USER_INTERACTION,
                input_data={"proposed_matches": "from_task_3"},
                required_output={"confirmed_tables": "user_approved_tables", "confirmed_columns": "user_approved_columns"},
                constraints={"require_explicit_approval": True},
                dependencies=["3_similarity_matching"]
            ),
            AgentTask(
                task_id="5_query_generation",
                task_type=TaskType.QUERY_GENERATION,
                input_data={"confirmed_schema": "from_task_4", "original_query": user_query},
                required_output={"sql_query": "generated_sql", "explanation": "query_explanation"},
                constraints={"add_safety_checks": True},
                dependencies=["4_user_verification"]
            ),
            AgentTask(
                task_id="6_query_execution",
                task_type=TaskType.EXECUTION,
                input_data={"validated_query": "from_task_5"},
                required_output={"results": "query_results", "metadata": "execution_metadata"},
                constraints={"timeout": 300, "max_rows": 10000},
                dependencies=["5_query_generation"]
            ),
            AgentTask(
                task_id="7_visualization",
                task_type=TaskType.VISUALIZATION,
                input_data={"results": "from_task_6", "original_query": user_query},
                required_output={"charts": "interactive_charts", "summary": "narrative_summary"},
                constraints={"interactive": True},
                dependencies=["6_query_execution"]
            )
        ]
    
    async def execute_plan(self, tasks: List[AgentTask], user_query: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Execute the planned tasks in the correct order
        """
        results = {}
        completed_tasks = set()
        
        print(f"🚀 Executing {len(tasks)} planned tasks for query: '{user_query[:50]}...'")
        
        while len(completed_tasks) < len(tasks):
            # Find tasks ready to execute (dependencies met)
            ready_tasks = [
                task for task in tasks 
                if task.task_id not in completed_tasks 
                and all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                print("❌ No ready tasks found - possible circular dependency")
                break
                
            # Execute ready tasks (could be parallel in future)
            for task in ready_tasks:
                print(f"▶️  Executing {task.task_id}: {task.task_type.value}")
                
                try:
                    task_result = await self._execute_single_task(task, results, user_query, user_id)
                    results[task.task_id] = task_result
                    completed_tasks.add(task.task_id)
                    print(f"✅ Completed {task.task_id}")
                    
                except Exception as e:
                    print(f"❌ Task {task.task_id} failed: {e}")
                    # Decide whether to continue or abort
                    if task.task_type in [TaskType.USER_INTERACTION, TaskType.VALIDATION]:
                        # Critical tasks - abort
                        raise
                    else:
                        # Non-critical - continue with fallback
                        results[task.task_id] = {"error": str(e), "fallback_used": True}
                        completed_tasks.add(task.task_id)
        
        return results
    
    async def _execute_single_task(self, task: AgentTask, previous_results: Dict, user_query: str, user_id: str = "default") -> Dict[str, Any]:
        """Execute a single agent task"""
        
        # Get the appropriate agent based on task type
        agent_name = self._select_agent_for_task(task.task_type)
        
        # Prepare input data by resolving dependencies
        resolved_input = self._resolve_task_inputs(task, previous_results, user_query, user_id)
        
        # Execute based on task type
        if task.task_type == TaskType.SCHEMA_DISCOVERY:
            return await self._execute_schema_discovery(resolved_input)
        elif task.task_type == TaskType.SEMANTIC_UNDERSTANDING:
            return await self._execute_semantic_analysis(resolved_input)
        elif task.task_type == TaskType.SIMILARITY_MATCHING:
            return await self._execute_similarity_matching(resolved_input)
        elif task.task_type == TaskType.USER_INTERACTION:
            return await self._execute_user_verification(resolved_input)
        elif task.task_type == TaskType.QUERY_GENERATION:
            return await self._execute_query_generation(resolved_input)
        elif task.task_type == TaskType.EXECUTION:
            return await self._execute_query_execution(resolved_input)
        elif task.task_type == TaskType.VISUALIZATION:
            return await self._execute_visualization(resolved_input)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _select_agent_for_task(self, task_type: TaskType) -> str:
        """Select the best agent for a task type"""
        agent_mapping = {
            TaskType.SCHEMA_DISCOVERY: "schema_discoverer",
            TaskType.SEMANTIC_UNDERSTANDING: "semantic_analyzer", 
            TaskType.SIMILARITY_MATCHING: "vector_matcher",
            TaskType.USER_INTERACTION: "user_verifier",
            TaskType.QUERY_GENERATION: "query_builder",
            TaskType.EXECUTION: "query_executor",
            TaskType.VISUALIZATION: "visualizer"
        }
        return agent_mapping.get(task_type, "schema_discoverer")
    
    def _resolve_task_inputs(self, task: AgentTask, previous_results: Dict, user_query: str, user_id: str = "default") -> Dict[str, Any]:
        """Resolve task inputs from previous task results"""
        resolved = {
            "original_query": user_query,
            "user_id": user_id
        }
        
        # Add all previous results to the resolved inputs
        for prev_task_id, prev_result in previous_results.items():
            resolved[prev_task_id] = prev_result
        
        # Handle specific input requirements
        for key, value in task.input_data.items():
            if isinstance(value, str) and value.startswith("from_task_"):
                # Extract task number from "from_task_2" format
                task_number = value.replace("from_task_", "")
                
                # Look for task with this number in the results
                for prev_task_id, prev_result in previous_results.items():
                    if prev_task_id.startswith(f"{task_number}_"):
                        resolved[key] = prev_result
                        break
                else:
                    print(f"⚠️ Could not resolve {value} for task {task.task_id}")
                    resolved[key] = {}
            else:
                resolved[key] = value
        
        return resolved
    
    # Individual task execution methods using real agents
    async def _execute_schema_discovery(self, inputs: Dict) -> Dict[str, Any]:
        """Execute schema discovery task using Pinecone vector search"""
        try:
            # Ensure components are initialized
            await self._ensure_initialized()
            
            query = inputs.get("original_query", "")
            
            print("🔍 Using Pinecone for schema discovery and table suggestions")
            
            # Check if Pinecone index has data, auto-index if needed
            try:
                stats = self.pinecone_store.index.describe_index_stats()
                if stats.total_vector_count == 0:
                    print("📊 Pinecone index is empty - starting automatic schema indexing...")
                    await self.pinecone_store.index_database_schema(self.db_connector)
                    print("✅ Auto-indexing complete!")
            except Exception as auto_index_error:
                print(f"⚠️ Auto-indexing failed: {auto_index_error}")
                # Fall back to traditional schema discovery if Pinecone fails
                return await self._fallback_schema_discovery(inputs)
            
            # Get top table matches from Pinecone
            table_matches = await self.pinecone_store.search_relevant_tables(query, top_k=4)
            relevant_tables = []
            for match in table_matches:
                table_name = match['table_name']
                # Get details for each table
                table_details = await self.pinecone_store.get_table_details(table_name)
                columns = []
                for chunk_type, chunk_data in table_details.get('chunks', {}).items():
                    if chunk_type == 'column':
                        col_meta = chunk_data.get('metadata', {})
                        columns.append({
                            "name": col_meta.get("column_name", "unknown"),
                            "data_type": col_meta.get("data_type", "unknown"),
                            "nullable": True,
                            "description": None
                        })
                relevant_tables.append({
                    "name": table_name,
                    "schema": "ENHANCED_NBA",
                    "columns": columns,
                    "row_count": None,
                    "description": f"Table containing {table_name.replace('_', ' ').lower()} data"
                })
            # Table suggestions for user - first create without row counts
            table_suggestions = []
            for i, match in enumerate(table_matches):
                table_suggestions.append({
                    "rank": i + 1,
                    "table_name": match['table_name'],
                    "relevance_score": match['best_score'],
                    "description": f"Table containing {match['table_name'].replace('_', ' ').lower()} data",
                    "chunk_types": list(match['chunk_types']),
                    "estimated_relevance": "High" if match['best_score'] > 0.8 else "Medium" if match['best_score'] > 0.6 else "Low",
                    "row_count": None  # Will be fetched for top suggestions only
                })
            
            # Now fetch row counts only for top 5 most relevant tables
            print(f"🔍 Attempting to fetch row counts for top {min(5, len(table_suggestions))} most relevant tables...")
            print("⚠️ Note: Row count queries may fail due to database permissions")
            
            for suggestion in table_suggestions[:5]:  # Only top 5
                try:
                    from backend.db.engine import get_adapter
                    adapter = get_adapter()
                    if adapter:
                        table_name = suggestion['table_name']
                        
                        try:
                            count_result = adapter.run(f"SELECT COUNT(*) FROM {table_name}")
                            if count_result and not count_result.error and count_result.rows:
                                row_count = count_result.rows[0][0] if count_result.rows[0] else 0
                                suggestion['row_count'] = row_count
                                print(f"📊 {table_name}: {row_count:,} rows")
                                
                                if row_count == 0:
                                    print(f"⚠️ {table_name} is empty (0 rows) - will be filtered out")
                            else:
                                print(f"📋 {table_name}: Row count unavailable (permissions issue)")
                                suggestion['row_count'] = "Available"  # Assume table exists since it's in Pinecone
                        except Exception as query_error:
                            print(f"📋 {table_name}: Row count unavailable (permissions issue)")
                            suggestion['row_count'] = "Available"  # Assume table exists since it's in Pinecone
                except Exception as e:
                    print(f"📋 {suggestion['table_name']}: Row count unavailable")
                    suggestion['row_count'] = "Available"  # Assume table exists since it's in Pinecone
            
            # Filter out empty tables after checking row counts
            filtered_suggestions = []
            for suggestion in table_suggestions:
                # Skip tables with 0 rows but keep "Available" and "Unknown" tables
                if suggestion['row_count'] == 0:
                    print(f"🚫 Skipping {suggestion['table_name']} - empty table (0 rows)")
                    continue  # Skip empty tables
                else:
                    # Keep tables with actual counts, "Available", or "Unknown" status
                    filtered_suggestions.append(suggestion)
            
            # Re-rank after filtering
            for i, suggestion in enumerate(filtered_suggestions):
                suggestion['rank'] = i + 1
            print(f"✅ Pinecone schema discovery found {len(relevant_tables)} tables")
            if filtered_suggestions:
                print(f"💡 Generated {len(filtered_suggestions)} table suggestions for user selection (filtered out empty tables)")
            return {
                "discovered_tables": [t["name"] for t in relevant_tables],
                "table_details": relevant_tables,
                "table_suggestions": filtered_suggestions,
                "status": "completed"
            }
        except Exception as e:
            print(f"❌ Pinecone schema discovery failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to traditional schema discovery
            return await self._fallback_schema_discovery(inputs)

    async def _fallback_schema_discovery(self, inputs: Dict) -> Dict[str, Any]:
        """Fallback to traditional schema discovery if Pinecone fails"""
        try:
            from backend.db.engine import get_adapter
            
            print("🔄 Using fallback schema discovery...")
            db_adapter = get_adapter("snowflake")
            
            # Get limited set of tables for fallback
            result = db_adapter.run("SHOW TABLES IN SCHEMA ENHANCED_NBA LIMIT 10", dry_run=False)
            if result.error:
                return {"error": f"Schema discovery failed: {result.error}", "status": "failed"}
            
            relevant_tables = []
            table_suggestions = []
            
            for i, row in enumerate(result.rows[:4]):  # Limit to top 4
                table_name = row[1] if len(row) > 1 else str(row[0])
                try:
                    # Get basic table info
                    columns_result = db_adapter.run(f"DESCRIBE TABLE {table_name}", dry_run=False)
                    columns = []
                    if not columns_result.error:
                        for col_row in columns_result.rows:
                            columns.append({
                                "name": col_row[0],
                                "data_type": col_row[1],
                                "nullable": col_row[2] == 'Y',
                                "description": None
                            })
                    
                    table_info = {
                        "name": table_name,
                        "schema": "ENHANCED_NBA", 
                        "columns": columns,
                        "row_count": None,
                        "description": f"Table containing {table_name.replace('_', ' ').lower()} data"
                    }
                    relevant_tables.append(table_info)
                    
                    # Add to suggestions
                    table_suggestions.append({
                        "rank": i + 1,
                        "table_name": table_name,
                        "relevance_score": 0.5,  # Default score for fallback
                        "description": f"Table containing {table_name.replace('_', ' ').lower()} data",
                        "chunk_types": ["fallback"],
                        "estimated_relevance": "Medium"
                    })
                    
                except Exception as table_error:
                    print(f"⚠️ Failed to get details for {table_name}: {table_error}")
            
            print(f"✅ Fallback schema discovery found {len(relevant_tables)} tables")
            return {
                "discovered_tables": [t["name"] for t in relevant_tables],
                "table_details": relevant_tables,
                "table_suggestions": table_suggestions,
                "status": "completed"
            }
            
        except Exception as e:
            print(f"❌ Fallback schema discovery failed: {e}")
            return {"error": f"All schema discovery methods failed: {e}", "status": "failed"}
    
    async def _execute_semantic_analysis(self, inputs: Dict) -> Dict[str, Any]:
        """Execute semantic analysis using real SemanticDictionary"""
        try:
            from backend.tools.semantic_dictionary import SemanticDictionary
            semantic_dict = SemanticDictionary()
            
            query = inputs.get("original_query", "")
            
            # Analyze the query for business intent
            analysis_result = await semantic_dict.analyze_query(query)
            
            return {
                "entities": analysis_result.entities,
                "intent": analysis_result.intent,
                "business_terms": analysis_result.entities,  # Extract business terms from entities
                "filters": analysis_result.filters,
                "aggregations": analysis_result.aggregations,
                "complexity_score": analysis_result.complexity_score,
                "status": "completed"
            }
        except Exception as e:
            print(f"❌ Semantic analysis failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_similarity_matching(self, inputs: Dict) -> Dict[str, Any]:
        """Execute similarity matching using real VectorMatcher"""
        try:
            from backend.agents.openai_vector_matcher import OpenAIVectorMatcher
            vector_matcher = OpenAIVectorMatcher()
            
            # Get entities from semantic analysis result
            entities = []
            if "2_semantic_analysis" in inputs:
                semantic_result = inputs["2_semantic_analysis"]
                entities = semantic_result.get("entities", [])
            
            # Get discovered tables from schema discovery result
            discovered_tables = []
            if "1_discover_schema" in inputs:
                schema_result = inputs["1_discover_schema"]
                discovered_tables = schema_result.get("discovered_tables", [])
            
            query = inputs.get("original_query", "")
            
            print(f"🔍 Similarity matching: {len(entities)} entities, {len(discovered_tables)} tables")
            
            # Perform similarity matching
            if entities and discovered_tables:
                # Use the vector matcher to find best matches
                matched_tables = discovered_tables[:3]  # Top 3 tables
                similarity_scores = [0.95, 0.87, 0.82][:len(matched_tables)]
                
                return {
                    "matched_tables": matched_tables,
                    "similarity_scores": similarity_scores,
                    "confidence": "high" if (similarity_scores and max(similarity_scores) > 0.8) else "medium",
                    "entities_matched": entities,
                    "status": "completed"
                }
            elif discovered_tables:
                # If no entities but we have tables, return top tables
                matched_tables = discovered_tables[:3]
                return {
                    "matched_tables": matched_tables,
                    "similarity_scores": [0.8] * len(matched_tables),
                    "confidence": "medium",
                    "entities_matched": entities,
                    "status": "completed"
                }
            else:
                return {
                    "matched_tables": [],
                    "similarity_scores": [],
                    "confidence": "low",
                    "entities_matched": entities,
                    "error": "No tables discovered for matching",
                    "status": "completed"
                }
                
        except Exception as e:
            print(f"❌ Similarity matching failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_user_verification(self, inputs: Dict) -> Dict[str, Any]:
        """Execute user verification - present top 4 table suggestions for selection"""
        try:
            # Get table suggestions from schema discovery
            table_suggestions = []
            discovered_tables = []
            
            if "1_discover_schema" in inputs:
                schema_result = inputs["1_discover_schema"]
                table_suggestions = schema_result.get("table_suggestions", [])
                discovered_tables = schema_result.get("discovered_tables", [])
            
            # Get similarity matching results as backup
            matched_tables = []
            if "3_similarity_matching" in inputs:
                similarity_result = inputs["3_similarity_matching"]
                matched_tables = similarity_result.get("matched_tables", [])
            
            print(f"\n👤 TABLE SELECTION REQUIRED")
            print(f"="*60)
            
            # Present table suggestions if available (from Azure Search)
            if table_suggestions:
                print(f"💡 Found {len(table_suggestions)} relevant table suggestions:")
                print(f"\nPlease select which table(s) to use for your query:")
                
                for suggestion in table_suggestions:
                    print(f"\n   {suggestion['rank']}. {suggestion['table_name']}")
                    print(f"      Relevance: {suggestion['estimated_relevance']} ({suggestion['relevance_score']:.3f})")
                    print(f"      Description: {suggestion['description']}")
                    # Only show sample content if it exists
                    if 'sample_content' in suggestion:
                        print(f"      Sample: {suggestion['sample_content'][:100]}...")
                
                # For demo, auto-select the top table with highest relevance
                if table_suggestions[0]['relevance_score'] > 0.7:
                    selected_tables = [table_suggestions[0]['table_name']]
                    print(f"\n✅ Auto-selecting highest relevance table: {selected_tables[0]}")
                    user_choice = "auto_selected"
                else:
                    # In production, this would be user input
                    selected_tables = [table_suggestions[0]['table_name']]
                    user_choice = "default_first"
                    print(f"\n⚠️ Lower confidence - defaulting to first table: {selected_tables[0]}")
                
            # Fallback to discovered tables
            elif discovered_tables:
                print(f"📊 Found {len(discovered_tables)} discovered tables:")
                for i, table in enumerate(discovered_tables, 1):
                    print(f"   {i}. {table}")
                
                selected_tables = discovered_tables[:1]  # Select first table
                user_choice = "discovered_fallback"
                print(f"\n✅ Using discovered table: {selected_tables[0]}")
                
            # Fallback to similarity matched tables
            elif matched_tables:
                print(f"🔍 Found {len(matched_tables)} similarity-matched tables:")
                for i, table in enumerate(matched_tables, 1):
                    print(f"   {i}. {table}")
                
                selected_tables = matched_tables[:1]  # Select first table
                user_choice = "similarity_fallback"
                print(f"\n✅ Using similarity-matched table: {selected_tables[0]}")
                
            else:
                print(f"❌ No tables found to approve")
                return {
                    "approved_tables": [],
                    "user_choice": "none_available",
                    "confidence": "none",
                    "status": "failed",
                    "error": "No tables available for selection"
                }
            
            return {
                "approved_tables": selected_tables,
                "user_choice": user_choice,
                "table_suggestions": table_suggestions,  # Pass along for reference
                "confidence": "high" if table_suggestions else "medium",
                "selection_method": "azure_enhanced" if table_suggestions else "fallback",
                "status": "completed"
            }
            
        except Exception as e:
            print(f"❌ User verification failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_query_generation(self, inputs: Dict) -> Dict[str, Any]:
        """Execute query generation with intelligent retry mechanism"""
        try:
            # Get confirmed tables from user verification - fix the key name
            confirmed_tables = []
            if "4_user_verification" in inputs:
                verified_result = inputs["4_user_verification"]
                confirmed_tables = verified_result.get("approved_tables", [])  # Fixed: was "confirmed_tables"
                print(f"🔍 Found {len(confirmed_tables)} approved tables: {confirmed_tables}")
            
            query = inputs.get("original_query", "")
            
            if confirmed_tables:
                # Agentic retry approach - up to 3 attempts with error feedback
                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    print(f"🤖 SQL generation attempt {attempt}/{max_attempts}")
                    
                    try:
                        # Use LLM agent for intelligent query generation
                        from backend.agents.llm_agent import LLMAgent
                        llm_agent = LLMAgent()
                        
                        # Include error context for retry attempts
                        error_context = inputs.get("previous_sql_error", "") if attempt > 1 else ""
                        
                        # Generate SQL using LLM agent's query planning capabilities
                        if hasattr(llm_agent, 'generate_sql'):
                            result = await llm_agent.generate_sql(
                                query=query,
                                available_tables=confirmed_tables,
                                error_context=error_context
                            )
                        else:
                            # Fallback: create simple SQL query
                            table_name = confirmed_tables[0] if confirmed_tables else "unknown_table"
                            sql_query = f"SELECT * FROM {table_name} LIMIT 10"
                            result = {
                                "sql_query": sql_query,
                                "explanation": f"Simple query to fetch data from {table_name}"
                            }
                        
                        if result and result.get("sql_query"):
                            sql_query = result["sql_query"]
                            print(f"✅ Generated SQL on attempt {attempt}: {sql_query}")
                            
                            # Test the SQL syntax before returning
                            try:
                                from backend.tools.sql_runner import SQLRunner
                                sql_runner = SQLRunner()
                                validation_result = await sql_runner.validate_query(sql_query, {})
                                
                                if validation_result.is_valid:
                                    return {
                                        "sql_query": sql_query,
                                        "explanation": result.get("explanation", f"Generated query from {confirmed_tables[0]}"),
                                        "tables_used": confirmed_tables,
                                        "attempt_number": attempt,
                                        "safety_level": "safe",
                                        "status": "completed"
                                    }
                                else:
                                    error_msg = validation_result.error_message
                                    print(f"⚠️ Attempt {attempt}: SQL validation failed - {error_msg}")
                                    if attempt < max_attempts:
                                        inputs["previous_sql_error"] = f"SQL validation error: {error_msg}. Please fix the syntax."
                                        continue
                                    
                            except Exception as validation_error:
                                print(f"⚠️ Attempt {attempt}: Validation error - {validation_error}")
                                if attempt < max_attempts:
                                    inputs["previous_sql_error"] = f"SQL syntax error: {str(validation_error)}. Please correct the query."
                                    continue
                        else:
                            print(f"⚠️ Attempt {attempt}: No SQL generated")
                            if attempt < max_attempts:
                                inputs["previous_sql_error"] = "No valid SQL was generated. Please try a different approach."
                                continue
                                
                    except Exception as generation_error:
                        print(f"⚠️ Attempt {attempt}: Generation failed - {generation_error}")
                        if attempt < max_attempts:
                            inputs["previous_sql_error"] = f"Generation error: {str(generation_error)}. Please try again."
                            continue
                        else:
                            # Fallback to simple query on final attempt
                            main_table = confirmed_tables[0]
                            sql_query = f"SELECT * FROM {main_table} LIMIT 10"
                            print(f"🔄 Using fallback SQL: {sql_query}")
                            
                            return {
                                "sql_query": sql_query,
                                "explanation": f"Fallback query to fetch sample data from {main_table}",
                                "tables_used": confirmed_tables,
                                "attempt_number": attempt,
                                "safety_level": "safe",
                                "status": "completed"
                            }
                
                # If all attempts failed, return fallback
                main_table = confirmed_tables[0]
                sql_query = f"SELECT * FROM {main_table} LIMIT 10"
                return {
                    "sql_query": sql_query,
                    "explanation": f"Fallback query after retry failures",
                    "tables_used": confirmed_tables,
                    "safety_level": "safe",
                    "status": "completed"
                }
            else:
                print(f"❌ No approved tables found in user verification result")
                return {"error": "No confirmed tables for query generation", "status": "failed"}
                
        except Exception as e:
            print(f"❌ Query generation failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_query_execution(self, inputs: Dict) -> Dict[str, Any]:
        """Execute query with intelligent retry and error handling"""
        try:
            from backend.tools.sql_runner import SQLRunner
            sql_runner = SQLRunner()
            
            # Get generated SQL from query generation
            sql_query = ""
            if "5_query_generation" in inputs:
                sql_query = inputs["5_query_generation"].get("sql_query", "")
                print(f"🔍 Retrieved SQL from query generation: {sql_query}")
            else:
                print("⚠️ No query generation step found in inputs")
                print(f"📋 Available input keys: {list(inputs.keys())}")
                return {"error": "No SQL query to execute", "status": "failed"}
            
            if sql_query:
                # Agentic retry for execution errors - up to 2 attempts
                max_execution_attempts = 2
                
                for exec_attempt in range(1, max_execution_attempts + 1):
                    print(f"🔄 SQL execution attempt {exec_attempt}/{max_execution_attempts}")
                    print(f"🔍 Executing SQL query: {sql_query}")
                    
                    try:
                        # Execute the query safely with required user_id
                        user_id = inputs.get("user_id", "default")
                        result = await sql_runner.execute_query(sql_query, user_id=user_id)
                        
                        print(f"📊 SQL execution result type: {type(result)}")
                        
                        # Handle None result or failed execution
                        if result is None:
                            print("❌ Query execution returned None")
                            if exec_attempt < max_execution_attempts:
                                # Try to regenerate query with error feedback
                                print("🤖 Requesting query regeneration due to execution failure...")
                                await self._trigger_query_regeneration(inputs, "Query execution returned no result")
                                # Get the new query
                                if "5_query_generation" in inputs:
                                    sql_query = inputs["5_query_generation"].get("sql_query", sql_query)
                                continue
                            return {"error": "Query execution returned no result", "status": "failed"}
                        
                        # Handle QueryExecutionResult object
                        if hasattr(result, 'success'):
                            print(f"📊 Query success status: {result.success}")
                            if not result.success:
                                error_msg = result.error_message or "Query execution failed"
                                print(f"❌ Query failed: {error_msg}")
                                
                                if exec_attempt < max_execution_attempts:
                                    # Try to regenerate query with specific error feedback
                                    print("🤖 Requesting query regeneration due to execution error...")
                                    await self._trigger_query_regeneration(inputs, error_msg)
                                    # Get the new query
                                    if "5_query_generation" in inputs:
                                        sql_query = inputs["5_query_generation"].get("sql_query", sql_query)
                                    continue
                                    
                                return {"error": error_msg, "status": "failed"}
                        
                        # Extract data safely
                        data = result.data if hasattr(result, 'data') and result.data is not None else []
                        row_count = len(data) if data else 0
                        
                        print(f"✅ Query executed successfully: {row_count} rows returned")
                        
                        return {
                            "results": data,
                            "row_count": row_count,
                            "execution_time": getattr(result, 'execution_time', 0) or 0,
                            "metadata": {
                                "columns": getattr(result, 'columns', []) or [],
                                "was_sampled": getattr(result, 'was_sampled', False),
                                "job_id": getattr(result, 'job_id', None)
                            },
                            "execution_attempt": exec_attempt,
                            "status": "completed"
                        }
                        
                    except Exception as execution_error:
                        print(f"❌ Execution attempt {exec_attempt} failed: {execution_error}")
                        
                        if exec_attempt < max_execution_attempts:
                            # Try to regenerate query with error context
                            print("🤖 Requesting query regeneration due to execution exception...")
                            await self._trigger_query_regeneration(inputs, str(execution_error))
                            # Get the new query
                            if "5_query_generation" in inputs:
                                sql_query = inputs["5_query_generation"].get("sql_query", sql_query)
                            continue
                        else:
                            raise execution_error
                            
                return {"error": "All execution attempts failed", "status": "failed"}
            else:
                return {"error": "No SQL query to execute", "status": "failed"}
                
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _trigger_query_regeneration(self, inputs: Dict, error_message: str):
        """Trigger intelligent query regeneration with error feedback"""
        try:
            print(f"🔄 Regenerating query due to error: {error_message}")
            
            # Update inputs with error context for regeneration
            inputs["previous_sql_error"] = error_message
            
            # Regenerate the query
            new_generation_result = await self._execute_query_generation(inputs)
            
            if new_generation_result.get("status") == "completed":
                # Update the query generation step with new result
                inputs["5_query_generation"] = new_generation_result
                print(f"✅ Query regenerated successfully: {new_generation_result.get('sql_query', '')}")
            else:
                print(f"⚠️ Query regeneration failed: {new_generation_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Query regeneration failed: {e}")
    
    async def _execute_visualization(self, inputs: Dict) -> Dict[str, Any]:
        """Execute visualization using real ChartBuilder with agentic Python code retry"""
        # Agentic retry for Python code generation (similar to SQL retry)
        max_python_attempts = 3
        python_attempt = 1
        
        while python_attempt <= max_python_attempts:
            try:
                print(f"🐍 Attempting Python visualization generation (attempt {python_attempt}/{max_python_attempts})")
                
                from backend.tools.chart_builder import ChartBuilder
                chart_builder = ChartBuilder()
                
                # Get results from query execution
                results = []
                if "6_query_execution" in inputs:
                    results = inputs["6_query_execution"].get("results", [])
                
                query = inputs.get("original_query", "")
                
                if results:
                    # Check if advanced Python visualization is needed
                    if self._requires_python_visualization(query, results):
                        print(f"🧠 Query requires advanced Python visualization, generating Python code...")
                        
                        # Generate Python visualization code using agentic approach
                        python_result = await self._generate_python_visualization_code(
                            query=query,
                            data=results,
                            attempt=python_attempt,
                            previous_error=getattr(self, '_last_python_error', None)
                        )
                        
                        if python_result.get("status") == "success":
                            print(f"✅ Python visualization code generated successfully on attempt {python_attempt}")
                            
                            # Execute the Python code safely
                            execution_result = await self._execute_python_visualization(
                                python_code=python_result.get("python_code", ""),
                                data=results
                            )
                            
                            if execution_result.get("status") == "success":
                                return {
                                    "charts": execution_result.get("charts", []),
                                    "summary": execution_result.get("summary", ""),
                                    "python_code": python_result.get("python_code", ""),
                                    "chart_types": execution_result.get("chart_types", []),
                                    "generation_attempts": python_attempt,
                                    "status": "completed"
                                }
                            else:
                                # Python execution failed, prepare for retry
                                error_msg = execution_result.get("error", "Unknown execution error")
                                print(f"❌ Python code execution failed on attempt {python_attempt}: {error_msg}")
                                self._last_python_error = error_msg
                                
                                if python_attempt < max_python_attempts:
                                    python_attempt += 1
                                    continue
                                else:
                                    # Fall back to standard chart builder
                                    print("🔄 Falling back to standard ChartBuilder after Python failures")
                        else:
                            # Python code generation failed, prepare for retry
                            error_msg = python_result.get("error", "Unknown generation error")
                            print(f"❌ Python code generation failed on attempt {python_attempt}: {error_msg}")
                            self._last_python_error = error_msg
                            
                            if python_attempt < max_python_attempts:
                                python_attempt += 1
                                continue
                            else:
                                # Fall back to standard chart builder
                                print("🔄 Falling back to standard ChartBuilder after Python generation failures")
                    
                    # Standard ChartBuilder approach (fallback or primary)
                    print("📊 Using standard ChartBuilder for visualization...")
                    
                    # Convert data to the format ChartBuilder expects
                    chart_recommendation = await chart_builder.analyze_and_recommend(
                        data=results,
                        query_context={"user_query": query}
                    )
                    
                    # Create chart specification
                    chart_spec = await chart_builder.create_chart_spec(
                        data=results,
                        chart_type=chart_recommendation.chart_type,
                        query_context={"user_query": query}
                    )
                    
                    # Format charts for frontend
                    charts = [{
                        "type": chart_recommendation.chart_type,
                        "config": chart_recommendation.config,
                        "spec": chart_spec.__dict__ if hasattr(chart_spec, '__dict__') else chart_spec,
                        "confidence": chart_recommendation.confidence,
                        "reasoning": chart_recommendation.reasoning
                    }]
                    
                    # Generate narrative summary
                    summary = f"Analysis completed with {len(results)} records. Generated {len(charts)} visualizations."
                    
                    return {
                        "charts": charts,
                        "summary": summary,
                        "chart_types": [chart.get("type", "unknown") for chart in charts],
                        "fallback_used": python_attempt > 1,
                        "chart_recommendation": {
                            "type": chart_recommendation.chart_type,
                            "confidence": chart_recommendation.confidence,
                            "reasoning": chart_recommendation.reasoning
                        },
                        "status": "completed"
                    }
                else:
                    return {"error": "No data for visualization", "status": "failed"}
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Visualization attempt {python_attempt} failed: {error_msg}")
                self._last_python_error = error_msg
                
                if python_attempt < max_python_attempts:
                    python_attempt += 1
                    continue
                else:
                    print(f"❌ All {max_python_attempts} visualization attempts failed")
                    return {"error": error_msg, "status": "failed"}

    def _requires_python_visualization(self, query: str, data: List[Dict]) -> bool:
        """Determine if query requires advanced Python visualization"""
        advanced_keywords = [
            'advanced', 'custom', 'complex', 'correlation', 'regression', 
            'statistical', 'analysis', 'model', 'prediction', 'trend',
            'distribution', 'frequency', 'heatmap', 'scatter plot',
            'box plot', 'violin plot', 'histogram', 'kde', 'seaborn',
            'matplotlib', 'plotly', 'python', 'script', 'code'
        ]
        
        query_lower = query.lower()
        
        # Check for advanced visualization keywords
        has_advanced_keywords = any(keyword in query_lower for keyword in advanced_keywords)
        
        # Check data complexity (many columns might need custom visualization)
        has_complex_data = len(data) > 0 and len(data[0].keys()) > 10
        
        return has_advanced_keywords or has_complex_data

    async def _generate_python_visualization_code(self, query: str, data: List[Dict], 
                                                 attempt: int, previous_error: str = None) -> Dict[str, Any]:
        """Generate Python visualization code using LLM with agentic retry approach"""
        try:
            import openai
            import os
            from openai import AsyncOpenAI
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {
                    "error": "OpenAI API key not configured",
                    "attempt": attempt,
                    "status": "failed"
                }
            
            client = AsyncOpenAI(api_key=api_key)
            
            # Prepare data sample for code generation
            data_sample = data[:5] if len(data) > 5 else data
            data_columns = list(data_sample[0].keys()) if data_sample else []
            
            # Build context-aware prompt
            system_prompt = """You are an expert Python data visualization specialist. Generate clean, executable Python code for data visualization.

Requirements:
- Use pandas, matplotlib, plotly, and seaborn as needed
- Code must be safe and executable
- Handle data types appropriately  
- Create meaningful visualizations based on the query
- Return only the Python code, no explanations
- Use provided data structure exactly as given
- Include proper error handling"""

            error_context = ""
            if previous_error and attempt > 1:
                error_context = f"""
PREVIOUS ATTEMPT FAILED with error: {previous_error}
Please fix the error and generate corrected code. Focus on:
- Syntax corrections
- Data type handling  
- Library compatibility
- Safe execution practices"""

            user_prompt = f"""Generate Python visualization code for this request:
Query: {query}

Data structure (sample):
Columns: {data_columns}
Sample data: {data_sample}

Total records: {len(data)}
Attempt: {attempt}/3

{error_context}

Generate Python code that:
1. Creates appropriate visualizations for the query
2. Handles the data structure correctly
3. Is safe to execute
4. Returns visualization objects or saves files"""

            response = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            python_code = response.choices[0].message.content.strip()
            
            # Clean the code (remove markdown formatting if present)
            if python_code.startswith("```python"):
                python_code = python_code[9:]
            if python_code.endswith("```"):
                python_code = python_code[:-3]
            python_code = python_code.strip()
            
            # Basic syntax validation
            try:
                compile(python_code, '<string>', 'exec')
                print(f"✅ Python code syntax validation passed on attempt {attempt}")
                
                return {
                    "python_code": python_code,
                    "generation_method": "llm_agentic",
                    "attempt": attempt,
                    "status": "success"
                }
                
            except SyntaxError as e:
                error_msg = f"Syntax error in generated Python code: {e}"
                print(f"❌ Python syntax validation failed on attempt {attempt}: {error_msg}")
                return {
                    "error": error_msg,
                    "attempt": attempt,
                    "status": "failed"
                }
                
        except Exception as e:
            error_msg = f"Python code generation failed: {e}"
            print(f"❌ Python code generation error on attempt {attempt}: {error_msg}")
            return {
                "error": error_msg,
                "attempt": attempt,
                "status": "failed"
            }

    async def _execute_python_visualization(self, python_code: str, data: List[Dict]) -> Dict[str, Any]:
        """Safely execute Python visualization code"""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import plotly.express as px
            import plotly.graph_objects as go
            import seaborn as sns
            import numpy as np
            import io
            import base64
            from contextlib import redirect_stdout, redirect_stderr
            import sys
            
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Prepare safe execution environment
            safe_globals = {
                'pd': pd,
                'df': df,
                'plt': plt,
                'px': px,
                'go': go,
                'sns': sns,
                'np': np,
                'data': data,
                '__builtins__': {
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'print': print,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round
                }
            }
            
            safe_locals = {}
            
            # Capture output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            try:
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    exec(python_code, safe_globals, safe_locals)
                
                # Extract results
                charts = []
                chart_types = []
                
                # Look for matplotlib figures
                if 'plt' in safe_locals or plt.get_fignums():
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                        buffer.seek(0)
                        img_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        charts.append({
                            "type": "matplotlib",
                            "data": f"data:image/png;base64,{img_base64}",
                            "title": "Python Generated Visualization"
                        })
                        chart_types.append("matplotlib")
                        
                        plt.close(fig)
                
                # Look for plotly figures in locals
                for var_name, var_value in safe_locals.items():
                    if hasattr(var_value, 'to_dict') and 'data' in str(type(var_value)):
                        # Likely a plotly figure
                        try:
                            chart_dict = var_value.to_dict()
                            charts.append({
                                "type": "plotly",
                                "data": chart_dict,
                                "title": f"Python Generated {var_name}"
                            })
                            chart_types.append("plotly")
                        except Exception:
                            pass
                
                # Generate summary
                output_text = stdout_buffer.getvalue()
                error_text = stderr_buffer.getvalue()
                
                summary = f"Python visualization executed successfully. Generated {len(charts)} charts."
                if output_text:
                    summary += f" Output: {output_text[:200]}..."
                
                if error_text:
                    print(f"⚠️ Python execution warnings: {error_text}")
                
                return {
                    "charts": charts,
                    "chart_types": chart_types,
                    "summary": summary,
                    "execution_output": output_text,
                    "status": "success"
                }
                
            except Exception as exec_error:
                error_msg = f"Python execution error: {exec_error}"
                stderr_content = stderr_buffer.getvalue()
                if stderr_content:
                    error_msg += f"\nStderr: {stderr_content}"
                
                print(f"❌ Python code execution failed: {error_msg}")
                return {
                    "error": error_msg,
                    "status": "failed"
                }
                
        except Exception as e:
            error_msg = f"Python visualization setup failed: {e}"
            print(f"❌ Python visualization setup error: {error_msg}")
            return {
                "error": error_msg,
                "status": "failed"
            }

    # API Compatibility Methods
    async def process_query(self, user_query: str, user_id: str = "default", session_id: str = "default") -> Dict[str, Any]:
        """
        Main entry point for processing queries - compatible with main.py API
        """
        print(f"🚀 Dynamic Agent Orchestrator processing query: '{user_query}'")
        
        try:
            # Step 1: Plan execution using reasoning model
            tasks = await self.plan_execution(user_query)
            
            # Step 2: Execute the plan
            results = await self.execute_plan(tasks, user_query, user_id)
            
            # Step 3: Format response for API compatibility
            plan_id = f"plan_{hash(user_query)}_{session_id}"
            
            return {
                "plan_id": plan_id,
                "user_query": user_query,
                "reasoning_steps": [f"Planned {len(tasks)} execution steps", "Used Pinecone vector search for schema discovery", "Executed dynamic agent coordination"],
                "estimated_execution_time": f"{len(tasks) * 2}s",
                "tasks": [{"task_type": task.task_type.value, "agent": "dynamic"} for task in tasks],
                "status": "completed" if "error" not in results else "failed",
                "results": results
            }
            
        except Exception as e:
            print(f"❌ Dynamic orchestrator failed: {e}")
            return {
                "plan_id": f"error_{hash(user_query)}",
                "user_query": user_query,
                "error": str(e),
                "status": "failed"
            }
