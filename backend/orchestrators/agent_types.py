"""
Agent Types and Data Classes
Defines the core types used by the Dynamic Agent Orchestrator
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


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
    PYTHON_GENERATION = "python_generation"
    VISUALIZATION_BUILDER = "visualization_builder"
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