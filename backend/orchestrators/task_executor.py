"""
Task Execution Module for Dynamic Agent Orchestrator
Handles the execution of individual agent tasks
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import os

from .agent_types import TaskType, AgentTask


class TaskExecutor:
    """Handles execution of individual agent tasks"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        
    async def execute_single_task(self, task: AgentTask, previous_results: Dict, user_query: str, user_id: str = "default", conversation_context: Dict = None) -> Dict[str, Any]:
        """Execute a single agent task"""
        
        # Get the appropriate agent based on task type
        agent_name = self._select_agent_for_task(task.task_type)
        
        # Prepare input data by resolving dependencies
        resolved_input = self._resolve_task_inputs(task, previous_results, user_query, user_id, conversation_context)
        
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
        elif task.task_type == TaskType.PYTHON_GENERATION:
            return await self._execute_python_generation(resolved_input)
        elif task.task_type == TaskType.VISUALIZATION_BUILDER:
            return await self._execute_visualization_builder(resolved_input)
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
            TaskType.PYTHON_GENERATION: "python_generator",
            TaskType.VISUALIZATION_BUILDER: "visualization_builder"
        }
        return agent_mapping.get(task_type, "schema_discoverer")
    
    def _resolve_task_inputs(self, task: AgentTask, previous_results: Dict, user_query: str, user_id: str = "default", conversation_context: Dict = None) -> Dict[str, Any]:
        """Resolve task inputs from previous task results"""
        # Fix user_id mapping - RBAC expects "default_user" not "default"
        if user_id == "default":
            user_id = "default_user"
            
        resolved = {
            "original_query": user_query,
            "user_id": user_id
        }
        
        # Add conversation context if available
        if conversation_context:
            resolved["conversation_context"] = conversation_context
        
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
        # Debug logging
        if task_type == "execution":
            print(f"🔍 Looking for execution results in inputs...")
            print(f"   Available keys: {list(inputs.keys())}")
            
            # Check results structure specifically
            results = inputs.get('results', {})
            if isinstance(results, dict):
                print(f"   Results keys: {list(results.keys())}")
                for key, value in results.items():
                    if 'execution' in key.lower():
                        print(f"   Found execution key: {key}")
                        if isinstance(value, dict) and 'results' in value:
                            print(f"   Has results data: {len(value['results']) if value['results'] else 0} rows")
        
        # Try direct match first (for consistency)
        if task_type in inputs:
            return inputs[task_type]
        
        # Check inputs.results structure (common pattern)
        results = inputs.get('results', {})
        if isinstance(results, dict):
            # Look for exact key match in results
            if task_type in results:
                return results[task_type]
            
            # Look for numbered/prefixed patterns in results
            for key, value in results.items():
                if task_type in key.lower():
                    if task_type == "execution":
                        print(f"✅ Found execution result under key: {key}")
                    return value
        
        # Map task types to common patterns
        type_patterns = {
            "schema_discovery": ["1_schema_discovery", "discover_schema", "schema_discovery", "1_discover_schema"],
            "semantic_understanding": ["2_semantic_understanding", "semantic_understanding", "semantic_analysis"],
            "similarity_matching": ["3_similarity_matching", "similarity_matching", "vector_matching"],
            "user_verification": ["4_user_verification", "user_verification", "user_interaction"],
            "query_generation": ["5_query_generation", "query_generation", "sql_generation"],
            "execution": ["4_execution", "6_execution", "6_query_execution", "query_execution", "execution"],
            "python_generation": ["python_generation", "7_python_generation"],
            "visualization": ["7_visualization", "visualization", "charts"]
        }
        
        # Look for numbered patterns first (dynamic o3-mini naming)
        patterns = type_patterns.get(task_type, [task_type])
        
        # Search in both main inputs and results
        search_spaces = [inputs, results] if isinstance(results, dict) else [inputs]
        
        for search_space in search_spaces:
            # Direct pattern matches
            for pattern in patterns:
                if pattern in search_space:
                    if task_type == "execution":
                        print(f"✅ Found execution via pattern {pattern}")
                    return search_space[pattern]
            
            # Partial matches (key contains pattern)
            for key in search_space.keys():
                for pattern in patterns:
                    if pattern.lower() in key.lower():
                        if task_type == "execution":
                            print(f"✅ Found execution via partial match: {key} contains {pattern}")
                        return search_space[key]
        
        if task_type == "execution":
            print(f"❌ No execution results found in any search space")
        
        return {}
    
    def _get_user_id_from_context(self, inputs: Dict) -> str:
        """Extract user_id from inputs, with fallback"""
        return inputs.get("user_id", "default_user")

    # Task execution methods would be moved here from the main orchestrator
    async def _execute_schema_discovery(self, inputs: Dict) -> Dict[str, Any]:
        """Execute schema discovery task using orchestrator's methods"""
        return await self.orchestrator._execute_schema_discovery(inputs)
    
    async def _execute_semantic_analysis(self, inputs: Dict) -> Dict[str, Any]:
        """Execute semantic analysis using orchestrator's methods"""
        return await self.orchestrator._execute_semantic_analysis(inputs)
    
    async def _execute_similarity_matching(self, inputs: Dict) -> Dict[str, Any]:
        """Execute similarity matching using orchestrator's methods"""
        return await self.orchestrator._execute_similarity_matching(inputs)
    
    async def _execute_user_verification(self, inputs: Dict) -> Dict[str, Any]:
        """Execute user verification using orchestrator's methods"""
        return await self.orchestrator._execute_user_verification(inputs)
    
    async def _execute_query_generation(self, inputs: Dict) -> Dict[str, Any]:
        """Execute query generation using orchestrator's methods"""
        return await self.orchestrator._execute_query_generation(inputs)
    
    async def _execute_query_execution(self, inputs: Dict) -> Dict[str, Any]:
        """Execute query execution using orchestrator's methods"""
        return await self.orchestrator._execute_query_execution(inputs)
    
    async def _execute_python_generation(self, inputs: Dict) -> Dict[str, Any]:
        """Execute python generation using orchestrator's methods"""
        return await self.orchestrator._execute_python_generation(inputs)
    
    async def _execute_visualization_builder(self, inputs: Dict) -> Dict[str, Any]:
        """Execute visualization builder using orchestrator's methods"""
        return await self.orchestrator._execute_visualization_builder(inputs)