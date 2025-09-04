"""
RBAC Manager - Role-Based Access Control and Authorization
Implements enterprise-grade security with fine-grained permissions
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import re

class Permission(Enum):
    """Granular permission types"""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    READ_SCHEMA = "read_schema"
    MODIFY_SCHEMA = "modify_schema"
    EXECUTE_QUERY = "execute_query"
    VIEW_PII = "view_pii"
    EXPORT_DATA = "export_data"
    ADMIN_FUNCTIONS = "admin_functions"
    COST_APPROVAL = "cost_approval"
    AUDIT_ACCESS = "audit_access"

class AccessLevel(Enum):
    """Data access levels"""
    NONE = "none"
    READ_ONLY = "read_only"
    FULL_ACCESS = "full_access"
    ADMIN = "admin"

@dataclass
class User:
    user_id: str
    email: str
    roles: List[str]
    department: str
    cost_limit: float
    data_classification_access: List[str]
    attributes: Dict[str, Any]
    active: bool = True
    created_at: datetime = None
    last_login: datetime = None

@dataclass
class Role:
    role_id: str
    name: str
    description: str
    permissions: List[Permission]
    resource_patterns: List[str]  # Regex patterns for accessible resources
    cost_limit: float
    time_restrictions: Optional[Dict[str, Any]] = None
    approval_required: bool = False

@dataclass
class Policy:
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    target_resources: List[str]
    target_users: List[str]
    target_roles: List[str]
    effect: str  # "allow" or "deny"
    conditions: Dict[str, Any]
    priority: int = 100

@dataclass
class AccessRequest:
    request_id: str
    user_id: str
    resource: str
    action: str
    context: Dict[str, Any]
    status: str  # "pending", "approved", "denied"
    requested_at: datetime
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    approval_reason: Optional[str] = None

class RBACManager:
    """
    Comprehensive Role-Based Access Control manager
    """
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.policies: Dict[str, Policy] = {}
        self.access_requests: Dict[str, AccessRequest] = {}
        
        # Initialize default roles and policies
        self._initialize_default_rbac()
    
    def _initialize_default_rbac(self):
        """Initialize default roles and policies"""
        
        # Default roles
        self.roles["analyst"] = Role(
            role_id="analyst",
            name="Data Analyst",
            description="Can query and analyze data",
            permissions=[
                Permission.READ_DATA,
                Permission.READ_SCHEMA,
                Permission.EXECUTE_QUERY,
                Permission.EXPORT_DATA
            ],
            resource_patterns=["public.*", "analytics.*"],
            cost_limit=50.0
        )
        
        self.roles["business_user"] = Role(
            role_id="business_user",
            name="Business User",
            description="Limited data access for business insights",
            permissions=[
                Permission.READ_DATA,
                Permission.READ_SCHEMA,
                Permission.EXECUTE_QUERY
            ],
            resource_patterns=["public.*", "business.*"],
            cost_limit=10.0,
            approval_required=True
        )
        
        self.roles["data_scientist"] = Role(
            role_id="data_scientist",
            name="Data Scientist",
            description="Advanced analytics and modeling",
            permissions=[
                Permission.READ_DATA,
                Permission.READ_SCHEMA,
                Permission.EXECUTE_QUERY,
                Permission.VIEW_PII,
                Permission.EXPORT_DATA
            ],
            resource_patterns=[".*"],  # Access to all schemas
            cost_limit=200.0
        )
        
        self.roles["admin"] = Role(
            role_id="admin",
            name="Administrator",
            description="Full system access",
            permissions=[p for p in Permission],
            resource_patterns=[".*"],
            cost_limit=1000.0
        )
        
        # Default policies
        self.policies["pii_protection"] = Policy(
            policy_id="pii_protection",
            name="PII Protection Policy",
            description="Restricts access to PII data",
            rules=[
                {
                    "condition": "column_contains_pii",
                    "action": "mask_unless_authorized"
                }
            ],
            target_resources=[".*\\.pii_.*", ".*\\..*_pii", ".*\\.email", ".*\\.ssn"],
            target_users=["*"],
            target_roles=["*"],
            effect="deny",
            conditions={"requires_permission": "VIEW_PII"},
            priority=1
        )
        
        self.policies["cost_control"] = Policy(
            policy_id="cost_control",
            name="Cost Control Policy",
            description="Enforces cost limits",
            rules=[
                {
                    "condition": "estimated_cost > user_limit",
                    "action": "require_approval"
                }
            ],
            target_resources=["*"],
            target_users=["*"],
            target_roles=["*"],
            effect="allow",
            conditions={"cost_threshold": 100.0},
            priority=50
        )
    
    async def check_query_permissions(self, user_id: str, sql: str, estimated_cost: float = 0.0) -> Dict[str, Any]:
        """
        Check if user has permission to execute a query
        """
        
        user = self.users.get(user_id)
        if not user or not user.active:
            return {
                "allowed": False,
                "error": "User not found or inactive",
                "reason": "authentication_failure"
            }
        
        # Extract referenced resources from SQL
        referenced_resources = await self._extract_resources_from_sql(sql)
        
        # Check permissions for each resource
        for resource in referenced_resources:
            resource_check = await self._check_resource_access(user, resource, "read")
            if not resource_check["allowed"]:
                return resource_check
        
        # Check cost limits
        cost_check = await self._check_cost_limits(user, estimated_cost)
        if not cost_check["allowed"]:
            return cost_check
        
        # Apply policies
        policy_check = await self._apply_policies(user, sql, referenced_resources, estimated_cost)
        if not policy_check["allowed"]:
            return policy_check
        
        return {
            "allowed": True,
            "resources": referenced_resources,
            "applied_policies": policy_check.get("applied_policies", [])
        }
    
    async def check_data_access(self, user_id: str, table: str, columns: List[str]) -> Dict[str, Any]:
        """
        Check data access permissions for specific table and columns
        """
        
        user = self.users.get(user_id)
        if not user:
            return {"allowed": False, "error": "User not found"}
        
        # Check table access
        table_access = await self._check_resource_access(user, table, "read")
        if not table_access["allowed"]:
            return table_access
        
        # Check column-level access
        accessible_columns = []
        restricted_columns = []
        
        for column in columns:
            column_resource = f"{table}.{column}"
            column_access = await self._check_resource_access(user, column_resource, "read")
            
            if column_access["allowed"]:
                accessible_columns.append(column)
            else:
                restricted_columns.append({
                    "column": column,
                    "reason": column_access.get("reason", "access_denied")
                })
        
        return {
            "allowed": len(accessible_columns) > 0,
            "accessible_columns": accessible_columns,
            "restricted_columns": restricted_columns,
            "requires_masking": await self._check_masking_requirements(user, table, accessible_columns)
        }
    
    async def can_access_pii(self, user_id: str) -> bool:
        """Check if user can access PII data"""
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Check if user has VIEW_PII permission through any role
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role and Permission.VIEW_PII in role.permissions:
                return True
        
        return False
    
    async def request_elevated_access(
        self, 
        user_id: str, 
        resource: str, 
        action: str, 
        justification: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Request elevated access that requires approval
        """
        
        request_id = f"req_{int(datetime.now().timestamp())}_{user_id[:8]}"
        
        request = AccessRequest(
            request_id=request_id,
            user_id=user_id,
            resource=resource,
            action=action,
            context={
                **context,
                "justification": justification,
                "requested_permissions": action
            },
            status="pending",
            requested_at=datetime.now()
        )
        
        self.access_requests[request_id] = request
        
        # Notify administrators (in real implementation)
        await self._notify_administrators(request)
        
        return request_id
    
    async def approve_access_request(
        self, 
        request_id: str, 
        approver_id: str, 
        approved: bool, 
        reason: str = ""
    ) -> bool:
        """
        Approve or deny an access request
        """
        
        request = self.access_requests.get(request_id)
        if not request:
            return False
        
        # Check if approver has admin permissions
        approver = self.users.get(approver_id)
        if not approver or not await self._has_permission(approver, Permission.COST_APPROVAL):
            return False
        
        request.status = "approved" if approved else "denied"
        request.reviewed_by = approver_id
        request.reviewed_at = datetime.now()
        request.approval_reason = reason
        
        # If approved, grant temporary elevated access
        if approved:
            await self._grant_temporary_access(request)
        
        return True
    
    async def add_user(self, user: User) -> bool:
        """Add a new user"""
        
        if user.user_id in self.users:
            return False
        
        user.created_at = datetime.now()
        self.users[user.user_id] = user
        return True
    
    async def update_user_roles(self, user_id: str, roles: List[str]) -> bool:
        """Update user's roles"""
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Validate roles exist
        for role_id in roles:
            if role_id not in self.roles:
                return False
        
        user.roles = roles
        return True
    
    async def create_role(self, role: Role) -> bool:
        """Create a new role"""
        
        if role.role_id in self.roles:
            return False
        
        self.roles[role.role_id] = role
        return True
    
    async def create_policy(self, policy: Policy) -> bool:
        """Create a new policy"""
        
        if policy.policy_id in self.policies:
            return False
        
        self.policies[policy.policy_id] = policy
        return True
    
    async def _check_resource_access(self, user: User, resource: str, action: str) -> Dict[str, Any]:
        """Check if user can access a specific resource"""
        
        # Check through user's roles
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if not role:
                continue
            
            # Check if role has required permission
            required_permission = self._get_required_permission(action)
            if required_permission not in role.permissions:
                continue
            
            # Check resource patterns
            for pattern in role.resource_patterns:
                if re.match(pattern, resource, re.IGNORECASE):
                    return {"allowed": True, "role": role_id}
        
        return {
            "allowed": False,
            "error": f"Access denied to resource: {resource}",
            "reason": "insufficient_permissions"
        }
    
    async def _check_cost_limits(self, user: User, estimated_cost: float) -> Dict[str, Any]:
        """Check cost limits for user"""
        
        # Check user-specific limit
        if estimated_cost > user.cost_limit:
            return {
                "allowed": False,
                "error": f"Query cost (${estimated_cost:.2f}) exceeds user limit (${user.cost_limit:.2f})",
                "reason": "cost_limit_exceeded",
                "requires_approval": True
            }
        
        # Check role-based limits
        max_role_limit = 0
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role:
                max_role_limit = max(max_role_limit, role.cost_limit)
        
        if estimated_cost > max_role_limit:
            return {
                "allowed": False,
                "error": f"Query cost (${estimated_cost:.2f}) exceeds role limit (${max_role_limit:.2f})",
                "reason": "role_cost_limit_exceeded",
                "requires_approval": True
            }
        
        return {"allowed": True}
    
    async def _apply_policies(
        self, 
        user: User, 
        sql: str, 
        resources: List[str], 
        estimated_cost: float
    ) -> Dict[str, Any]:
        """Apply security policies"""
        
        applied_policies = []
        
        # Sort policies by priority
        sorted_policies = sorted(self.policies.values(), key=lambda p: p.priority)
        
        for policy in sorted_policies:
            # Check if policy applies to this user/role
            if not await self._policy_applies_to_user(policy, user):
                continue
            
            # Check if policy applies to these resources
            if not await self._policy_applies_to_resources(policy, resources):
                continue
            
            # Evaluate policy conditions
            policy_result = await self._evaluate_policy(policy, user, sql, resources, estimated_cost)
            
            if policy_result["triggered"]:
                applied_policies.append({
                    "policy_id": policy.policy_id,
                    "effect": policy.effect,
                    "action": policy_result.get("action"),
                    "reason": policy_result.get("reason")
                })
                
                # If deny policy is triggered, reject immediately
                if policy.effect == "deny" and not policy_result.get("override_allowed", False):
                    return {
                        "allowed": False,
                        "error": policy_result.get("reason", "Policy violation"),
                        "reason": "policy_violation",
                        "applied_policies": applied_policies
                    }
        
        return {
            "allowed": True,
            "applied_policies": applied_policies
        }
    
    async def _extract_resources_from_sql(self, sql: str) -> List[str]:
        """Extract table/resource references from SQL"""
        
        resources = []
        
        # Extract FROM clauses
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        from_matches = re.findall(from_pattern, sql, re.IGNORECASE)
        resources.extend(from_matches)
        
        # Extract JOIN clauses
        join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
        resources.extend(join_matches)
        
        return list(set(resources))  # Remove duplicates
    
    def _get_required_permission(self, action: str) -> Permission:
        """Map action to required permission"""
        
        action_permission_map = {
            "read": Permission.READ_DATA,
            "write": Permission.WRITE_DATA,
            "execute": Permission.EXECUTE_QUERY,
            "export": Permission.EXPORT_DATA,
            "admin": Permission.ADMIN_FUNCTIONS
        }
        
        return action_permission_map.get(action, Permission.READ_DATA)
    
    async def _has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role and permission in role.permissions:
                return True
        
        return False
    
    async def _check_masking_requirements(self, user: User, table: str, columns: List[str]) -> List[str]:
        """Check which columns require masking for this user"""
        
        masking_required = []
        
        if not await self.can_access_pii(user.user_id):
            # Check for PII columns
            pii_patterns = ["email", "phone", "ssn", "credit_card", "pii"]
            
            for column in columns:
                for pattern in pii_patterns:
                    if pattern in column.lower():
                        masking_required.append(column)
                        break
        
        return masking_required
    
    async def _policy_applies_to_user(self, policy: Policy, user: User) -> bool:
        """Check if policy applies to user"""
        
        # Check user-specific targeting
        if "*" not in policy.target_users and user.user_id not in policy.target_users:
            return False
        
        # Check role-based targeting
        if "*" not in policy.target_roles:
            user_roles = set(user.roles)
            target_roles = set(policy.target_roles)
            if not user_roles.intersection(target_roles):
                return False
        
        return True
    
    async def _policy_applies_to_resources(self, policy: Policy, resources: List[str]) -> bool:
        """Check if policy applies to resources"""
        
        if "*" in policy.target_resources:
            return True
        
        for resource in resources:
            for pattern in policy.target_resources:
                if re.match(pattern, resource, re.IGNORECASE):
                    return True
        
        return False
    
    async def _evaluate_policy(
        self, 
        policy: Policy, 
        user: User, 
        sql: str, 
        resources: List[str], 
        estimated_cost: float
    ) -> Dict[str, Any]:
        """Evaluate if policy conditions are met"""
        
        # Simple rule evaluation - would be more sophisticated in practice
        for rule in policy.rules:
            condition = rule.get("condition", "")
            
            if condition == "column_contains_pii":
                # Check if query accesses PII columns
                pii_accessed = any("pii" in resource.lower() or "email" in resource.lower() 
                                 for resource in resources)
                if pii_accessed and not await self.can_access_pii(user.user_id):
                    return {
                        "triggered": True,
                        "action": rule.get("action", "deny"),
                        "reason": "PII access requires special permission"
                    }
            
            elif condition == "estimated_cost > user_limit":
                if estimated_cost > user.cost_limit:
                    return {
                        "triggered": True,
                        "action": rule.get("action", "require_approval"),
                        "reason": f"Cost ${estimated_cost:.2f} exceeds limit ${user.cost_limit:.2f}"
                    }
        
        return {"triggered": False}
    
    async def _grant_temporary_access(self, request: AccessRequest):
        """Grant temporary elevated access after approval"""
        
        # In practice, this would create temporary permission overrides
        # For now, we'll just log the approval
        pass
    
    async def _notify_administrators(self, request: AccessRequest):
        """Notify administrators of access request"""
        
        # In practice, this would send notifications via email/Slack
        pass
