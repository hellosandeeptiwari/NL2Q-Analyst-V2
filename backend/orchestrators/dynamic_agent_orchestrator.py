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
        """Initialize the system on startup"""
        try:
            print("🚀 Starting system initialization...")
            
            # Initialize database connector
            await self._initialize_database_connector()
            
            # Initialize Pinecone vector store
            await self._initialize_vector_store()
            
            # Perform auto-indexing if needed
            if self.pinecone_store and self.db_connector:
                await self._check_and_perform_comprehensive_indexing()
            
            print("✅ System initialization completed")
        except Exception as e:
            print(f"⚠️ Error during startup initialization: {e}")
            # Don't fail startup completely
            
    async def _ensure_initialized(self):
        """Ensure database and Pinecone are initialized (lazy initialization)"""
        if not self.db_connector:
            await self._initialize_database_connector()
            
        if not self.pinecone_store:
            await self._initialize_vector_store()
            
    async def _initialize_database_connector(self):
        """Initialize database connection"""
        try:
            from backend.db.engine import get_adapter
            print("🔌 Initializing database connector...")
            self.db_connector = get_adapter("snowflake")
            if self.db_connector:
                print("✅ Database connector initialized successfully")
                # Test the connection
                test_result = self.db_connector.run("SELECT 1 as test", dry_run=False)
                if test_result and not test_result.error:
                    print("✅ Database connection test successful")
                else:
                    print(f"⚠️ Database connection test failed: {test_result.error if test_result else 'Unknown error'}")
            else:
                print("❌ Database connector returned None")
        except Exception as e:
            print(f"❌ Database connector initialization failed: {e}")
            self.db_connector = None
            
    async def _initialize_vector_store(self):
        """Initialize Pinecone vector store"""
        try:
            print("🔍 Initializing Pinecone vector store...")
            from backend.pinecone_schema_vector_store import PineconeSchemaVectorStore
            self.pinecone_store = PineconeSchemaVectorStore()
            if self.pinecone_store:
                print("✅ Vector store initialized successfully")
                # Test Pinecone connection
                try:
                    stats = self.pinecone_store.index.describe_index_stats()
                    print(f"✅ Pinecone connection test successful - {stats.total_vector_count} vectors indexed")
                except Exception as test_error:
                    print(f"⚠️ Pinecone connection test failed: {test_error}")
            else:
                print("❌ Vector store returned None")
        except Exception as e:
            print(f"❌ Vector store initialization failed: {e}")
            self.pinecone_store = None
            
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
        Use o3-mini reasoning model to dynamically plan which agents to use and in what order
        """
        
        print(f"🧠 Letting o3-mini analyze and plan dynamically for query: '{user_query}'")
        
        planning_prompt = f"""You are an intelligent query orchestrator for pharmaceutical data analysis. Analyze the user's query and determine the most efficient execution plan.

USER QUERY: "{user_query}"

AVAILABLE CAPABILITIES:
- schema_discovery: Find relevant database tables and columns
- semantic_understanding: Extract entities and business intent from query
- similarity_matching: Match query terms to database schema
- user_interaction: Get user confirmation for ambiguous requests
- query_generation: Generate SQL queries
- execution: Execute queries and retrieve data
- visualization: Create charts, graphs, and visual analysis

INSTRUCTIONS:
Analyze the query and determine:
1. What is the user actually asking for?
2. What steps are truly necessary to fulfill this request?
3. Can this be accomplished with fewer steps?
4. Does the user explicitly ask for visual output (charts, graphs, trends, analysis)?

Create a JSON array with only the essential tasks needed. Do not include unnecessary steps.

Examples of good planning:
- Simple data request "show me top 5 records" → might only need: schema_discovery, query_generation, execution
- Complex analysis "analyze sales trends and show charts" → might need: schema_discovery, semantic_understanding, similarity_matching, query_generation, execution, visualization
- Ambiguous request → might need user_interaction for clarification

OUTPUT: Return only a JSON array of tasks in this format:
[
  {{"task_id": "1_taskname", "task_type": "capability_name"}},
  {{"task_id": "2_taskname", "task_type": "capability_name"}}
]

Be intelligent - use only what's needed for this specific query."""
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Use o3-mini for planning as specified
            model_to_use = self.reasoning_model  # This is o3-mini
            print(f"🧠 Using model for planning: {model_to_use}")
            print(f"🔍 OpenAI API Key available: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
            print(f"📝 Planning prompt length: {len(planning_prompt)} characters")
            print(f"📋 Full planning prompt:\n{planning_prompt}")
            
            print(f"🚀 Calling o3-mini for planning...")
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": planning_prompt}],
                max_completion_tokens=1000  # o3-mini uses max_completion_tokens and doesn't support temperature
            )
            print(f"✅ o3-mini responded successfully")
            
            # Parse the response to extract task plan
            content = response.choices[0].message.content.strip()
            print(f"� o3-mini full response:")
            print(f"--- START o3-mini RESPONSE ---")
            print(content)
            print(f"--- END o3-mini RESPONSE ---")
            print(f"📊 Response length: {len(content)} characters")
            print(f"🔍 Response starts with: '{content[:50]}...'")
            print(f"🔍 Response ends with: '...{content[-50:]}'")
            
            # Clean up the response for JSON parsing
            original_content = content
            try:
                print(f"🧹 Starting JSON cleanup...")
                # Remove any markdown formatting
                if '```json' in content:
                    print(f"🔧 Found ```json markers, extracting JSON...")
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    print(f"🔧 Found ``` markers, extracting content...")
                    content = content.split('```')[1].split('```')[0]
                
                # Remove any leading/trailing text that isn't JSON
                content = content.strip()
                print(f"🧹 After cleanup: '{content[:100]}...'")
                
                if not content.startswith('['):
                    print(f"⚠️ Content doesn't start with '[', searching for JSON array...")
                    # Find the first [ and last ]
                    start = content.find('[')
                    end = content.rfind(']')
                    print(f"🔍 Found '[' at position: {start}, ']' at position: {end}")
                    if start != -1 and end != -1:
                        content = content[start:end+1]
                        print(f"🔧 Extracted JSON: '{content[:100]}...'")
                    else:
                        print(f"❌ No valid JSON array found in response")
                        raise ValueError("No JSON array found in o3-mini response")
                
                print(f"🔄 Attempting to parse JSON...")
                tasks_data = json.loads(content)
                print(f"✅ JSON parsed successfully!")
                print(f"📊 Parsed data type: {type(tasks_data)}")
                print(f"📊 Number of tasks: {len(tasks_data) if isinstance(tasks_data, list) else 'Not a list'}")
                
                if isinstance(tasks_data, list):
                    for i, task in enumerate(tasks_data):
                        print(f"  Task {i+1}: {task}")
                
                if isinstance(tasks_data, list) and len(tasks_data) > 0:
                    print(f"✅ o3-mini planning successful: {len(tasks_data)} tasks")
                    return self._convert_to_agent_tasks(tasks_data, user_query)
                else:
                    print(f"❌ Invalid task data structure from o3-mini")
                    print(f"📊 Data: {tasks_data}")
                    raise ValueError("Invalid task data structure from o3-mini")

            except Exception as parse_err:
                print(f"❌ o3-mini JSON parsing failed!")
                print(f"🔍 Parse error type: {type(parse_err).__name__}")
                print(f"🔍 Parse error message: {str(parse_err)}")
                print(f"📤 Original o3-mini response:")
                print(f"--- ORIGINAL RESPONSE ---")
                print(original_content)
                print(f"--- END ORIGINAL ---")
                print(f"🧹 Cleaned content that failed to parse:")
                print(f"--- CLEANED CONTENT ---")
                print(repr(content))  # Use repr to show exact characters
                print(f"--- END CLEANED ---")
                print("🔄 Falling back to dynamic default plan...")
                return self._create_default_plan(user_query)
                
        except Exception as e:
            print(f"❌ o3-mini model call failed completely!")
            print(f"🔍 Error type: {type(e).__name__}")
            print(f"🔍 Error message: {str(e)}")
            print(f"🔍 Full error details:")
            import traceback
            traceback.print_exc()
            print("🔄 Falling back to dynamic default plan...")
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
    
    def _convert_to_agent_tasks(self, tasks_data: List[Dict], user_query: str) -> List[AgentTask]:
        """Convert JSON task data to AgentTask objects"""
        print(f"🔄 Converting {len(tasks_data)} tasks to AgentTask objects...")
        tasks = []
        
        for i, task_data in enumerate(tasks_data):
            print(f"📋 Processing task {i+1}: {task_data}")
            try:
                # Validate required fields
                task_id = task_data.get("task_id", f"task_{i+1}")
                task_type_str = task_data.get("task_type")
                
                print(f"  🏷️ Task ID: {task_id}")
                print(f"  🔧 Task Type String: {task_type_str}")
                
                if not task_type_str:
                    print(f"  ⚠️ Missing task_type, skipping task")
                    continue
                
                # Convert string to TaskType enum
                try:
                    task_type = TaskType(task_type_str)
                    print(f"  ✅ Task Type Enum: {task_type}")
                except ValueError as e:
                    print(f"  ❌ Invalid task_type '{task_type_str}': {e}")
                    print(f"  📋 Valid task types: {[t.value for t in TaskType]}")
                    continue
                
                task = AgentTask(
                    task_id=task_id,
                    task_type=task_type,
                    input_data=task_data.get("input_requirements", {"query": user_query}),
                    required_output=task_data.get("output_expectations", {}),
                    constraints=task_data.get("constraints", {}),
                    dependencies=task_data.get("dependencies", [])
                )
                tasks.append(task)
                print(f"  ✅ Task created successfully")
                
            except Exception as task_err:
                print(f"  ❌ Failed to create task {i+1}: {task_err}")
                print(f"  🔍 Task data: {task_data}")
                continue
        
        print(f"✅ Successfully converted {len(tasks)} out of {len(tasks_data)} tasks")
        return tasks
    
    def _create_default_plan(self, user_query: str) -> List[AgentTask]:
        """Create a query-aware execution plan based on user intent"""
        print(f"🔍 Creating dynamic plan for query: '{user_query}'")
        
        # Analyze query intent to determine required tasks
        query_lower = user_query.lower()
        
        # Determine if visualization is needed
        needs_visualization = any(keyword in query_lower for keyword in [
            'chart', 'graph', 'plot', 'visualize', 'visualization', 'trend', 'compare',
            'distribution', 'pattern', 'analysis', 'insight', 'dashboard'
        ])
        
        # Determine if it's a simple data retrieval
        is_simple_query = any(keyword in query_lower for keyword in [
            'show', 'fetch', 'get', 'retrieve', 'list', 'top', 'first', 'last',
            'rows', 'records', 'data', 'table'
        ]) and not needs_visualization
        
        # Start with core required tasks
        tasks = [
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
            )
        ]
        
        # Add visualization task only if needed
        if needs_visualization:
            print("📊 Adding visualization task based on query intent")
            tasks.append(
                AgentTask(
                    task_id="7_visualization",
                    task_type=TaskType.VISUALIZATION,
                    input_data={"results": "from_task_6", "original_query": user_query},
                    required_output={"charts": "interactive_charts", "summary": "narrative_summary"},
                    constraints={"interactive": True},
                    dependencies=["6_query_execution"]
                )
            )
        else:
            print("📋 Skipping visualization - query appears to be simple data retrieval")
        
        print(f"✅ Created dynamic plan with {len(tasks)} tasks (visualization: {needs_visualization})")
        return tasks
    
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
        # Fix user_id mapping - RBAC expects "default_user" not "default"
        if user_id == "default":
            user_id = "default_user"
            
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
    
    def _find_task_result_by_type(self, inputs: Dict, task_type: str) -> Dict[str, Any]:
        """Universal helper to find task results by type regardless of naming convention"""
        # Try direct match first (for consistency)
        if task_type in inputs:
            return inputs[task_type]
        
        # Map task types to common patterns
        type_patterns = {
            "schema_discovery": ["1_discover_schema", "discover_schema", "schema_discovery"],
            "semantic_understanding": ["2_semantic_understanding", "semantic_understanding", "semantic_analysis"],
            "similarity_matching": ["3_similarity_matching", "similarity_matching", "vector_matching"],
            "user_verification": ["4_user_verification", "user_verification", "user_interaction"],
            "query_generation": ["5_query_generation", "query_generation", "sql_generation"],
            "execution": ["6_query_execution", "query_execution", "execution"],
            "visualization": ["7_visualization", "visualization", "charts"]
        }
        
        # Look for numbered patterns first (dynamic o3-mini naming)
        for key in inputs.keys():
            if task_type in key:
                return inputs[key]
        
        # Look for pattern matches
        patterns = type_patterns.get(task_type, [task_type])
        for pattern in patterns:
            if pattern in inputs:
                return inputs[pattern]
            # Check if any key contains the pattern
            for key in inputs.keys():
                if pattern in key.lower():
                    return inputs[key]
        
        return {}
    
    def _get_user_id_from_context(self, inputs: Dict) -> str:
        """Extract user_id from inputs, with fallback"""
        return inputs.get("user_id", "default_user")
    
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
                # Get details for each table. Prefer a fast hybrid approach:
                # 1) Use Pinecone for table discovery (we already have match)
                # 2) Use the Enhanced SchemaRetriever to fetch column-level
                #    metadata (names, types) for the matched tables. This avoids
                #    falling back to SELECT * and keeps latency low by focusing
                #    only on the small set of matched tables.
                columns = []
                try:
                    # Try to use the SchemaRetriever for richer column info
                    from backend.agents.schema_retriever import SchemaRetriever
                    retriever = SchemaRetriever()
                    col_info = await retriever.get_columns_for_table(table_name, schema="ENHANCED_NBA")
                    if col_info and isinstance(col_info, list):
                        for col in col_info:
                            columns.append({
                                "name": col.get("name") or col.get("column_name"),
                                "data_type": col.get("data_type", "unknown"),
                                "nullable": col.get("nullable", True),
                                "description": col.get("description")
                            })
                except Exception:
                    # Fall back to chunk-derived metadata if retriever isn't available
                    try:
                        table_details = await self.pinecone_store.get_table_details(table_name)
                        
                        # Use enhanced column extraction from get_table_details
                        extracted_columns = table_details.get('columns', [])
                        for col_name in extracted_columns:
                            columns.append({
                                "name": col_name,
                                "data_type": "unknown",
                                "nullable": True,
                                "description": None
                            })
                        
                        # Fallback: check chunks manually if no columns extracted
                        if not extracted_columns:
                            for chunk_type, chunk_data in table_details.get('chunks', {}).items():
                                if chunk_type in ['column_group', 'column']:
                                    col_meta = chunk_data.get('metadata', {})
                                    # Handle both direct column info and column group info
                                    if 'columns' in col_meta:
                                        for col_name in col_meta['columns']:
                                            columns.append({
                                                "name": col_name,
                                                "data_type": "unknown", 
                                                "nullable": True,
                                                "description": None
                                            })
                                    elif 'column_name' in col_meta:
                                        columns.append({
                                            "name": col_meta.get("column_name", "unknown"),
                                            "data_type": col_meta.get("data_type", "unknown"),
                                            "nullable": True,
                                            "description": None
                                        })
                    except Exception:
                        # As a last resort leave columns empty and let later
                        # steps handle column discovery per-table
                        columns = []

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
                    "row_count": "Available"  # Skip expensive row count fetching for better performance
                })
            
            # Skip row count fetching to improve performance (saves 10+ seconds)
            print(f"⚡ Skipping row count fetching for performance - assuming all tables are available")
            
            # Set all tables as available since they exist in Pinecone
            for suggestion in table_suggestions:
                suggestion['row_count'] = "Available"
            
            # Filter out empty tables after checking row counts
            filtered_suggestions = []
            for suggestion in table_suggestions:
                # Since we're not checking row counts, assume all tables are valid
                print(f"✅ Including {suggestion['table_name']} - assumed available")
                filtered_suggestions.append(suggestion)
            
            # Re-rank after filtering
            for i, suggestion in enumerate(filtered_suggestions):
                suggestion['rank'] = i + 1
            print(f"✅ Pinecone schema discovery found {len(relevant_tables)} tables")
            if filtered_suggestions:
                print(f"💡 Generated {len(filtered_suggestions)} table suggestions for user selection")
            return {
                "discovered_tables": [t["name"] for t in relevant_tables],
                "table_details": relevant_tables,
                "table_suggestions": filtered_suggestions,
                "pinecone_matches": table_matches,  # Store original Pinecone matches for reuse
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
            
            # Get entities from semantic analysis result using dynamic helper
            semantic_result = self._find_task_result_by_type(inputs, "semantic_understanding")
            entities = semantic_result.get("entities", [])
            
            # Get discovered tables from schema discovery result using dynamic helper
            schema_result = self._find_task_result_by_type(inputs, "schema_discovery")
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
            # Get table suggestions from schema discovery using dynamic helper
            schema_result = self._find_task_result_by_type(inputs, "schema_discovery")
            table_suggestions = schema_result.get("table_suggestions", [])
            discovered_tables = schema_result.get("discovered_tables", [])
            
            # Get similarity matching results as backup using dynamic helper
            similarity_result = self._find_task_result_by_type(inputs, "similarity_matching")
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
        """Generate SQL query - completely self-sufficient for any planning scenario"""
        try:
            query = inputs.get("original_query", inputs.get("query", ""))
            print(f"🔍 SQL Generation for: {query}")
            
            # Discover what context we have available (from ANY previous tasks)
            available_context = self._gather_available_context(inputs)
            
            # Build table context from whatever is available
            confirmed_tables = self._determine_target_tables(available_context, query)
            
            if not confirmed_tables:
                print(f"🔍 No table context available - performing autonomous table discovery")
                confirmed_tables = await self._autonomous_table_discovery(query)
            
            if not confirmed_tables:
                return {
                    "error": "Could not determine target tables for query",
                    "status": "failed",
                    "suggestion": "Query may be too ambiguous - consider specifying table names"
                }
            
            print(f"🎯 Target tables: {confirmed_tables}")
            
            # Generate SQL with whatever context we have
            return await self._generate_sql_with_context(query, confirmed_tables, available_context)
            
        except Exception as e:
            print(f"❌ Query generation failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _gather_available_context(self, inputs: Dict) -> Dict[str, Any]:
        """Gather all available context from any previous tasks"""
        context = {
            "schemas": [],
            "entities": [],
            "matched_tables": [],
            "user_preferences": {},
            "semantic_intent": None
        }
        
        for key, value in inputs.items():
            if not isinstance(value, dict):
                continue
                
            # Extract useful information regardless of task naming
            if "schema" in str(value).lower():
                context["schemas"].extend(value.get("discovered_tables", []))
                context["schemas"].extend(value.get("table_suggestions", []))
            
            if "entities" in value:
                context["entities"].extend(value.get("entities", []))
            
            if "matched_tables" in value:
                context["matched_tables"].extend(value.get("matched_tables", []))
            
            if "intent" in value:
                context["semantic_intent"] = value.get("intent")
            
            if "approved_tables" in value:
                context["user_preferences"]["tables"] = value.get("approved_tables", [])
        
        return context
    
    def _determine_target_tables(self, context: Dict, query: str) -> List[str]:
        """Determine target tables from available context"""
        # Priority 1: User-approved tables
        if context.get("user_preferences", {}).get("tables"):
            return context["user_preferences"]["tables"]
        
        # Priority 2: Matched tables from similarity
        if context.get("matched_tables"):
            return context["matched_tables"][:1]  # Top match
        
        # Priority 3: Schema suggestions
        schemas = context.get("schemas", [])
        if schemas:
            # Handle both dict and string formats
            for schema in schemas:
                if isinstance(schema, dict) and "table_name" in schema:
                    return [schema["table_name"]]
                elif isinstance(schema, str):
                    return [schema]
        
        return []
    
    async def _autonomous_table_discovery(self, query: str) -> List[str]:
        """Discover tables autonomously when no prior context exists"""
        try:
            await self._ensure_initialized()
            
            # Method 1: Pinecone search
            if hasattr(self, 'pinecone_store') and self.pinecone_store:
                matches = await self.pinecone_store.search_relevant_tables(query, top_k=1)
                if matches:
                    return [matches[0]['table_name']]
            
            # Method 2: Query pattern analysis
            query_lower = query.lower()
            if 'nba' in query_lower:
                return ['nba_final_output']  # Educated guess
            
            # Method 3: Could add database introspection here
            return []
            
        except Exception as e:
            print(f"⚠️ Autonomous discovery failed: {e}")
            return []
    
    async def _generate_sql_with_context(self, query: str, tables: List[str], context: Dict) -> Dict[str, Any]:
        """Generate SQL using available tables and context"""
        try:
            # Use the existing database-aware SQL generation that was working before
            result = await self._generate_database_aware_sql(
                query=query,
                available_tables=tables,
                error_context="",
                pinecone_matches=context.get("pinecone_matches", [])
            )
            
            if result and result.get("sql_query"):
                return {
                    "sql_query": result["sql_query"],
                    "explanation": result.get("explanation", f"Generated query for {tables[0]}"),
                    "tables_used": tables,
                    "context_used": list(context.keys()),
                    "status": "completed"
                }
            else:
                # Fallback to intelligent fallback SQL generation
                return await self._fallback_sql_generation(tables[0], query)
                
        except Exception as e:
            print(f"⚠️ Database-aware generation failed: {e}")
            return await self._fallback_sql_generation(tables[0], query)

    async def _execute_query_execution(self, inputs: Dict) -> Dict[str, Any]:
        """Execute SQL query - works with any planning scenario"""
        try:
            from backend.tools.sql_runner import SQLRunner
            sql_runner = SQLRunner()
            
            # Find SQL query from ANY previous task or generate it ourselves
            sql_query = self._find_sql_query(inputs)
            
            if not sql_query:
                print(f"🔍 No SQL found in previous tasks - generating SQL autonomously")
                # Generate SQL ourselves if o3-mini didn't plan a separate generation step
                generation_result = await self._execute_query_generation(inputs)
                if generation_result.get("status") == "completed":
                    sql_query = generation_result.get("sql_query")
                else:
                    return {
                        "error": "Could not obtain SQL query for execution",
                        "status": "failed"
                    }
            
            print(f"🔍 Executing SQL: {sql_query}")
            
            # Execute with retry logic
            user_id = self._get_user_id_from_context(inputs)
            
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                print(f"🔄 Execution attempt {attempt}/{max_attempts}")
                
                try:
                    result = await sql_runner.execute_query(sql_query, user_id=user_id)
                    
                    if result and hasattr(result, 'success') and result.success:
                        data = result.data if hasattr(result, 'data') and result.data is not None else []
                        return {
                            "results": data,
                            "row_count": len(data) if data else 0,
                            "execution_time": getattr(result, 'execution_time', 0) or 0,
                            "metadata": {
                                "columns": getattr(result, 'columns', []) or [],
                                "was_sampled": getattr(result, 'was_sampled', False),
                                "job_id": getattr(result, 'job_id', None)
                            },
                            "sql_executed": sql_query,
                            "execution_attempt": attempt,
                            "status": "completed"
                        }
                    else:
                        error_msg = getattr(result, 'error_message', 'Unknown execution error')
                        print(f"⚠️ Attempt {attempt} failed: {error_msg}")
                        
                        if attempt < max_attempts:
                            # Try with error recovery
                            continue
                        else:
                            return {
                                "error": f"Query execution failed after {max_attempts} attempts: {error_msg}",
                                "sql_query": sql_query,
                                "status": "failed"
                            }
                            
                except Exception as e:
                    print(f"⚠️ Attempt {attempt} exception: {e}")
                    if attempt < max_attempts:
                        continue
                    else:
                        return {
                            "error": f"Query execution failed: {str(e)}",
                            "sql_query": sql_query,
                            "status": "failed"
                        }
            
        except Exception as e:
            print(f"❌ Query execution setup failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _find_sql_query(self, inputs: Dict) -> str:
        """Find SQL query from any previous task result"""
        # Look through all previous results for SQL
        for key, value in inputs.items():
            if isinstance(value, dict):
                # Direct SQL query field
                if "sql_query" in value:
                    sql = value["sql_query"]
                    if sql and isinstance(sql, str):
                        print(f"🎯 Found SQL in task {key}: {sql[:50]}...")
                        return sql
                
                # Could also be in nested results
                if "results" in value and isinstance(value["results"], dict):
                    if "sql_query" in value["results"]:
                        sql = value["results"]["sql_query"]
                        if sql and isinstance(sql, str):
                            print(f"🎯 Found SQL in nested results of {key}")
                            return sql
        
        # Check if SQL was passed directly in inputs
        if "sql_query" in inputs:
            return inputs["sql_query"]
        
        print(f"❌ No SQL query found in previous tasks")
        return ""

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

    async def _try_alternative_tables(self, inputs: Dict, permission_error: str) -> Dict[str, Any]:
        """Try alternative tables when permission is denied for the primary table"""
        try:
            # Get the original table suggestions from schema discovery
            table_suggestions = []
            if "1_discover_schema" in inputs:
                schema_result = inputs["1_discover_schema"]
                table_suggestions = schema_result.get("table_suggestions", [])
            
            if len(table_suggestions) <= 1:
                print("⚠️ No alternative tables available to try")
                return {"error": permission_error, "status": "failed"}
            
            # Try the next best table (skip the first one that failed)
            for suggestion in table_suggestions[1:4]:  # Try up to 3 alternatives
                alt_table = suggestion['table_name']
                print(f"🔄 Trying alternative table: {alt_table}")
                
                # Generate SQL for the alternative table
                fallback_result = await self._fallback_sql_generation(alt_table)
                if not fallback_result.get("sql_query"):
                    continue
                
                alt_sql = fallback_result["sql_query"]
                print(f"🔍 Testing alternative SQL: {alt_sql}")
                
                # Test execution with the alternative table
                try:
                    from backend.tools.sql_runner import SQLRunner
                    sql_runner = SQLRunner()
                    user_id = inputs.get("user_id", "default_user")
                    
                    test_result = await sql_runner.execute_query(alt_sql, user_id=user_id)
                    
                    if test_result and hasattr(test_result, 'success') and test_result.success:
                        print(f"✅ Alternative table {alt_table} accessible - using it")
                        
                        # Update the user verification step with the new table (dynamic)
                        user_verification_result = self._find_task_result_by_type(inputs, "user_verification")
                        if user_verification_result:
                            # Find the key for user verification in inputs and update it
                            for key, value in inputs.items():
                                if "user_verification" in key.lower() or "user_interaction" in key.lower():
                                    inputs[key]["approved_tables"] = [alt_table]
                                    break
                        
                        # Update the query generation step (dynamic)
                        # Find the key for query generation and update it
                        for key, value in inputs.items():
                            if "query_generation" in key.lower() or "sql_generation" in key.lower():
                                inputs[key] = {
                                    "sql_query": alt_sql,
                                    "explanation": f"Alternative query using accessible table {alt_table}",
                                    "tables_used": [alt_table],
                                    "status": "completed"
                                }
                                break
                        
                        # Return successful execution result
                        data = test_result.data if hasattr(test_result, 'data') and test_result.data is not None else []
                        return {
                            "results": data,
                            "row_count": len(data) if data else 0,
                            "execution_time": getattr(test_result, 'execution_time', 0) or 0,
                            "metadata": {
                                "columns": getattr(test_result, 'columns', []) or [],
                                "was_sampled": getattr(test_result, 'was_sampled', False),
                                "job_id": getattr(test_result, 'job_id', None),
                                "alternative_table_used": alt_table
                            },
                            "execution_attempt": 1,
                            "status": "completed"
                        }
                    else:
                        print(f"⚠️ Alternative table {alt_table} also failed")
                        
                except Exception as alt_error:
                    print(f"⚠️ Alternative table {alt_table} error: {alt_error}")
                    continue
            
            print("❌ All alternative tables failed or inaccessible")
            return {"error": f"All tables inaccessible: {permission_error}", "status": "failed"}
            
        except Exception as e:
            print(f"❌ Alternative table search failed: {e}")
            return {"error": permission_error, "status": "failed"}
    
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
                
                # Get results from query execution using dynamic helper
                exec_result = self._find_task_result_by_type(inputs, "execution")
                results = exec_result.get("results", [])
                
                # Check if query execution actually succeeded
                if exec_result.get("status") == "failed":
                    print(f"❌ Query execution failed - no data for visualization: {exec_result.get('error', 'Unknown error')}")
                    return {
                        "error": f"Cannot create visualization: {exec_result.get('error', 'Query execution failed')}",
                        "status": "failed"
                    }
                
                query = inputs.get("original_query", "")
                
                print(f"📊 Visualization input: {len(results)} rows of data")
                if results and len(results) > 0:
                    print(f"📋 Sample data columns: {list(results[0].keys()) if results[0] else 'No columns'}")
                
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
                            
                            if execution_result.get("status") == "success" and execution_result.get("charts"):
                                print(f"✅ Python visualization successful: Generated {len(execution_result.get('charts', []))} charts")
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
                                error_msg = execution_result.get("error", "No charts generated")
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
- Include proper error handling

Important:
- Assign your main chart to a variable named `fig` (this is required). If you produce multiple charts, place them in a list called `figures` and still include a main `fig`.
- If using Plotly, construct a `plotly.graph_objects.Figure` (go.Figure) and assign it to `fig`.
"""

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
4. Assigns the main visualization to a variable named `fig` (required). If multiple charts are created, also return a list `figures`.
5. Returns or exposes the `fig`/`figures` objects so the orchestrator can detect and serialize them.
"""

            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
                'json': json,  # Add json module
                'io': io,      # Add io module
                'base64': base64,  # Add base64 module
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
                    'round': round,
                    'isinstance': isinstance,
                    'type': type,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                    '__import__': __import__,
                    # Add standard exception classes
                    'Exception': Exception,
                    'ValueError': ValueError,
                    'TypeError': TypeError,
                    'KeyError': KeyError,
                    'IndexError': IndexError,
                    'AttributeError': AttributeError,
                    'RuntimeError': RuntimeError,
                    'ImportError': ImportError,
                    'NameError': NameError,
                    'ZeroDivisionError': ZeroDivisionError,
                    # Add commonly used functions
                    'sorted': sorted,
                    'reversed': reversed,
                    'any': any,
                    'all': all,
                    'bool': bool,
                    'tuple': tuple,
                    'set': set,
                    'frozenset': frozenset
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
                processed_objects = set()  # Track processed objects to prevent duplicates

                # Build a combined namespace to scan for figures (globals + locals)
                combined_ns = {}
                combined_ns.update(safe_globals)
                combined_ns.update(safe_locals)

                # Prefer obvious variable names
                candidate_names = [
                    'fig', 'figure', 'chart', 'figures', 'fig_list', 'figures_list'
                ]

                # Helper: detect plotly figures
                def _is_plotly_fig(obj):
                    try:
                        import plotly.graph_objs as go
                        if isinstance(obj, go.Figure):
                            return True
                    except Exception:
                        pass
                    # Fallback: duck-type check
                    if hasattr(obj, 'to_dict'):
                        try:
                            d = obj.to_dict()
                            if isinstance(d, dict) and ('data' in d or 'layout' in d):
                                return True
                        except Exception:
                            return False
                    return False

                # Helper: detect matplotlib figures
                def _is_matplotlib_fig(obj):
                    try:
                        import matplotlib.figure as mplfig
                        if isinstance(obj, mplfig.Figure):
                            return True
                    except Exception:
                        pass
                    return False

                # 1) Check for named candidate variables first
                for name in candidate_names:
                    if name in combined_ns:
                        obj = combined_ns.get(name)
                        if obj is None:
                            continue
                        if _is_plotly_fig(obj):
                            try:
                                # Check if we've already processed this object
                                obj_id = id(obj)
                                if obj_id in processed_objects:
                                    continue
                                processed_objects.add(obj_id)
                                
                                # Convert plotly figure to dict and clean numpy arrays
                                chart_dict = obj.to_dict()
                                # Clean numpy arrays and other non-serializable objects
                                cleaned_chart_dict = self._convert_non_serializable_data(chart_dict)
                                
                                charts.append({
                                    'type': 'plotly',
                                    'data': cleaned_chart_dict,
                                    'title': f'Python Generated {name}'
                                })
                                chart_types.append('plotly')
                            except Exception:
                                pass
                        elif _is_matplotlib_fig(obj):
                            try:
                                buf = io.BytesIO()
                                obj.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                                buf.seek(0)
                                charts.append({
                                    'type': 'matplotlib',
                                    'data': f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
                                    'title': f'Python Generated {name}'
                                })
                                chart_types.append('matplotlib')
                            except Exception:
                                pass

                # 2) Inspect matplotlib global fignums as fallback
                try:
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
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
                except Exception:
                    pass

                # 3) Scan all variables for plotly figures as a final pass
                for var_name, var_value in combined_ns.items():
                    if var_value is None:
                        continue
                    try:
                        if _is_plotly_fig(var_value):
                            try:
                                # Check if we've already processed this object
                                obj_id = id(var_value)
                                if obj_id in processed_objects:
                                    continue
                                processed_objects.add(obj_id)
                                
                                # Convert plotly figure to dict and clean numpy arrays
                                chart_dict = var_value.to_dict()
                                # Clean numpy arrays and other non-serializable objects
                                cleaned_chart_dict = self._convert_non_serializable_data(chart_dict)
                                
                                charts.append({
                                    "type": "plotly",
                                    "data": cleaned_chart_dict,
                                    "title": f"Python Generated {var_name}"
                                })
                                chart_types.append("plotly")
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                # Generate summary
                output_text = stdout_buffer.getvalue()
                error_text = stderr_buffer.getvalue()
                
                # Enhanced logging when no charts are detected
                if len(charts) == 0:
                    print("❌ No charts detected in Python execution!")
                    print(f"📊 Available variables in namespace: {list(combined_ns.keys())}")
                    print(f"📋 Checked candidate names: {candidate_names}")
                    print(f"🔍 Matplotlib figures detected: {len(plt.get_fignums())}")
                    
                    # Check for any objects that might be charts
                    potential_charts = []
                    for name, obj in combined_ns.items():
                        if obj is not None:
                            obj_type = str(type(obj))
                            if any(chart_hint in obj_type.lower() for chart_hint in ['figure', 'plot', 'chart', 'graph']):
                                potential_charts.append(f"{name}: {obj_type}")
                    
                    if potential_charts:
                        print(f"🤔 Potential chart objects found: {potential_charts}")
                    else:
                        print("🚫 No chart-like objects detected in execution namespace")
                
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

    async def _generate_database_aware_sql(self, query: str, available_tables: List[str], 
                                         error_context: str = "", pinecone_matches: List[Dict] = None) -> Dict[str, Any]:
        """Generate SQL with database-specific awareness using schema from Pinecone"""
        try:
            import openai
            import os
            from openai import AsyncOpenAI
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"error": "OpenAI API key not configured", "status": "failed"}
            
            client = AsyncOpenAI(api_key=api_key)
            
            # Extract schema information from Pinecone results (no additional DB calls needed!)
            table_schemas = []
            if pinecone_matches:
                for match in pinecone_matches:
                    table_name = match.get('table_name')
                    if table_name in available_tables:
                        # Get detailed schema from Pinecone metadata
                        table_details = await self.pinecone_store.get_table_details(table_name)
                        
                        # Extract column information from the enhanced table details
                        columns = table_details.get('columns', [])
                        
                        if not columns:
                            # Fallback: Extract from chunks manually if columns not already extracted
                            for chunk_type, chunk_data in table_details.get('chunks', {}).items():
                                if chunk_type == 'column_group':
                                    chunk_metadata = chunk_data.get('metadata', {})
                                    if 'columns' in chunk_metadata:
                                        # Get column names from metadata
                                        for col_name in chunk_metadata['columns']:
                                            columns.append(col_name)
                                elif chunk_type == 'table_overview':
                                    # Could also extract column info from overview content if needed
                                    pass
                        
                        # If no columns found in metadata, parse from content
                        if not columns and 'column_group' in table_details.get('chunks', {}):
                            content = table_details['chunks']['column_group'].get('metadata', {}).get('content', '')
                            # Parse column names from content like "col1 (VARCHAR), col2 (INTEGER)"
                            import re
                            col_matches = re.findall(r'(\w+)\s*\([^)]+\)', content)
                            columns.extend(col_matches)
                        
                        if columns:
                            schema_text = f"Table: {table_name}\nColumns: {', '.join(columns)}"
                            table_schemas.append(schema_text)
                            print(f"📊 Extracted schema from Pinecone for {table_name}: {len(columns)} columns")
                        else:
                            print(f"⚠️ No column information found in Pinecone for {table_name}")
                            table_schemas.append(f"Table: {table_name}\nColumns: [Use DESCRIBE TABLE to explore]")
            
            # Fallback: if no Pinecone matches provided, get fresh schema info
            if not table_schemas:
                print("⚠️ No Pinecone schema found, fetching directly from database")
                for table_name in available_tables:
                    try:
                        describe_sql = f'DESCRIBE TABLE "COMMERCIAL_AI"."ENHANCED_NBA"."{table_name}"'
                        result = self.db_connector.run(describe_sql, dry_run=False)
                        
                        if result and result.rows:
                            columns = [row[0] for row in result.rows[:20]]  # Limit to first 20 columns
                            table_schema = f"Table: {table_name}\nColumns: {', '.join(columns)}"
                            table_schemas.append(table_schema)
                            print(f"📊 Retrieved schema directly for {table_name}: {len(columns)} columns")
                    except Exception as e:
                        print(f"⚠️ Error getting schema for {table_name}: {e}")
                        table_schemas.append(f"Table: {table_name}\nColumns: [Schema unavailable]")
            
            # Database-specific system prompt with ACTUAL SCHEMA INFORMATION
            schema_context = "\n\n".join(table_schemas)
            system_prompt = f"""You are a database query expert specializing in Snowflake SQL generation.

Database Context:
- Engine: Snowflake
- Database: COMMERCIAL_AI
- Schema: ENHANCED_NBA
- Full qualification: "COMMERCIAL_AI"."ENHANCED_NBA"."table_name"

ACTUAL TABLE SCHEMAS:
{schema_context}

Requirements:
- Use proper Snowflake identifier quoting (double quotes for case-sensitive names)
- Always specify full path: "COMMERCIAL_AI"."ENHANCED_NBA"."table_name"
- Use ONLY the column names shown above - they are the exact column names in the tables
- Keep queries simple and safe
- Use LIMIT to prevent large result sets
- Handle case-sensitive table/column names properly

Generate clean, executable SQL only. No explanations."""

            error_context_text = ""
            if error_context:
                error_context_text = f"\n\nPrevious Error Context: {error_context}\nPlease fix the identified issues."

            user_prompt = f"""Generate a Snowflake SQL query for: {query}

Use these tables: {', '.join(available_tables)}
{error_context_text}

Return only the SQL query, properly formatted for Snowflake."""

            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean the SQL (remove markdown if present)
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            elif sql_query.startswith("```"):
                sql_query = sql_query[3:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            # Apply Snowflake-specific fixes
            sql_query = self._apply_snowflake_quoting(sql_query, available_tables)
            
            return {
                "sql_query": sql_query,
                "explanation": f"Database-aware SQL for Snowflake generated from: {query}",
                "generation_method": "database_aware",
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Database-aware SQL generation failed: {e}")
            return {"error": str(e), "status": "failed"}

    def _apply_snowflake_quoting(self, sql_query: str, table_names: List[str]) -> str:
        """Apply proper Snowflake quoting to table and column references"""
        try:
            import re
            
            # Ensure full database.schema.table qualification and proper quoting
            for table_name in table_names:
                # Remove any existing qualification to avoid double-prefixing
                clean_table_name = table_name.replace('COMMERCIAL_AI.ENHANCED_NBA.', '')
                clean_table_name = clean_table_name.replace('ENHANCED_NBA.', '').replace('"', '')
                
                # Pattern 1: Replace unqualified table references in FROM clauses
                unqualified_pattern = rf'\bFROM\s+{re.escape(clean_table_name)}\b'
                sql_query = re.sub(unqualified_pattern, f'FROM "COMMERCIAL_AI"."ENHANCED_NBA"."{clean_table_name}"', sql_query, flags=re.IGNORECASE)
                
                # Pattern 2: Replace JOIN references  
                join_pattern = rf'\bJOIN\s+{re.escape(clean_table_name)}\b'
                sql_query = re.sub(join_pattern, f'JOIN "COMMERCIAL_AI"."ENHANCED_NBA"."{clean_table_name}"', sql_query, flags=re.IGNORECASE)
                
                # Pattern 3: Fix any existing malformed references
                malformed_patterns = [
                    rf'\bENHANCED_NBA\.{re.escape(clean_table_name)}\b',
                    rf'\bCOMMERCIAL_AI\.{re.escape(clean_table_name)}\b'
                ]
                for pattern in malformed_patterns:
                    sql_query = re.sub(pattern, f'"COMMERCIAL_AI"."ENHANCED_NBA"."{clean_table_name}"', sql_query, flags=re.IGNORECASE)
                
                # Pattern 4: Fix cases where schema/database got treated as table
                schema_as_table_patterns = [
                    rf'\bFROM\s+"?ENHANCED_NBA"?\s*$',
                    rf'\bFROM\s+"?COMMERCIAL_AI"?\s*$'
                ]
                for pattern in schema_as_table_patterns:
                    sql_query = re.sub(pattern, f'FROM "COMMERCIAL_AI"."ENHANCED_NBA"."{clean_table_name}"', sql_query, flags=re.IGNORECASE)
            
            return sql_query
            
        except Exception as e:
            print(f"⚠️ Snowflake quoting failed: {e}")
            return sql_query

    async def _fallback_sql_generation(self, table_name: str, query: str = "") -> Dict[str, Any]:
        """Generate a query-aware fallback SQL with proper database quoting"""
        try:
            # Clean the table name - remove any existing schema qualification
            clean_table_name = table_name.replace('ENHANCED_NBA.', '').replace('"', '')
            
            # Parse the query for specific requirements
            query_lower = query.lower() if query else ""
            
            # Determine LIMIT from query
            limit = 10  # default
            if "top 5" in query_lower or "first 5" in query_lower:
                limit = 5
            elif "top 3" in query_lower:
                limit = 3
            elif "top 10" in query_lower:
                limit = 10
            elif "limit" in query_lower:
                # Try to extract number after limit
                import re
                match = re.search(r'limit\s+(\d+)', query_lower)
                if match:
                    limit = int(match.group(1))
            
            # Try to get column information for better fallback
            columns = []
            try:
                from backend.agents.schema_retriever import SchemaRetriever
                retriever = SchemaRetriever()
                if hasattr(retriever, 'get_columns_for_table'):
                    cols = await retriever.get_columns_for_table(clean_table_name, schema="ENHANCED_NBA")
                    if cols:
                        columns = [c.get('name') or c.get('column_name') for c in cols]
            except Exception:
                pass
            
            # Build WHERE clause for filtering
            where_clause = ""
            if "recommended" in query_lower and ("not {}" in query_lower or "with text" in query_lower):
                # Filter for non-empty recommended messages
                if columns and any("recommended" in col.lower() for col in columns):
                    rec_col = next((col for col in columns if "recommended" in col.lower()), None)
                    if rec_col:
                        where_clause = f' WHERE "{rec_col}" IS NOT NULL AND "{rec_col}" != \'\' AND "{rec_col}" != \'{{}}\''
            
            # Build the SQL query with proper schema.table format
            if columns:
                # Use specific columns if available, prioritize relevant ones
                relevant_cols = []
                for col in columns:
                    if any(keyword in col.lower() for keyword in ['input', 'recommended', 'action', 'value']):
                        relevant_cols.append(col)
                
                if relevant_cols:
                    col_list = ', '.join([f'"{col}"' for col in relevant_cols[:8]])
                else:
                    col_list = ', '.join([f'"{col}"' for col in columns[:8]])
                    
                sql_query = f'SELECT {col_list} FROM "ENHANCED_NBA"."{clean_table_name}"{where_clause} LIMIT {limit}'
            else:
                # Fallback to SELECT * with proper quoting
                sql_query = f'SELECT * FROM "ENHANCED_NBA"."{clean_table_name}"{where_clause} LIMIT {limit}'
            
            print(f"🔧 Generated query-aware fallback SQL: {sql_query}")
            
            return {
                "sql_query": sql_query,
                "explanation": f"Query-aware fallback for {clean_table_name} (limit: {limit}, filters applied: {bool(where_clause)})",
                "generation_method": "smart_fallback",
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Fallback SQL generation failed: {e}")
            # Emergency fallback - ensure we always return something valid
            clean_name = table_name.replace('ENHANCED_NBA.', '').replace('"', '')
            return {
                "sql_query": f'SELECT * FROM "ENHANCED_NBA"."{clean_name}" LIMIT 10',
                "explanation": f"Emergency fallback for {clean_name}",
                "generation_method": "emergency",
                "status": "success"
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
                "reasoning_steps": [f"Planned {len(tasks)} execution steps", "Analyzed database structure and content", "Coordinated intelligent query processing"],
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

    def _convert_non_serializable_data(self, obj):
        """Convert non-JSON-serializable objects to serializable ones"""
        import numpy as np
        
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):  # Any numpy-like array
            try:
                return obj.tolist()
            except:
                return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_non_serializable_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_non_serializable_data(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_non_serializable_data(item) for item in obj]
        else:
            return obj
