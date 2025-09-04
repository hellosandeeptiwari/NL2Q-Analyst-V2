"""
Enhanced User Profile Management for Pharma NL2Q System
Supports role-based access, pharma-specific contexts, and user preferences
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid
import os
from enum import Enum

class UserRole(Enum):
    ANALYST = "analyst"
    DATA_SCIENTIST = "data_scientist"
    MEDICAL_AFFAIRS = "medical_affairs"
    COMMERCIAL = "commercial"
    REGULATORY = "regulatory"
    EXECUTIVE = "executive"
    ADMIN = "admin"

class AccessLevel(Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    RESTRICTED = "restricted"
    FULL = "full"

@dataclass
class UserPreferences:
    """User interface and workflow preferences"""
    theme: str = "dark"  # dark, light, pharma-blue
    default_visualization: str = "bar"  # bar, line, pie, table
    auto_execute_high_confidence: bool = True
    show_sql_queries: bool = False
    default_data_limit: int = 100
    preferred_therapeutic_areas: List[str] = None
    notification_settings: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.preferred_therapeutic_areas is None:
            self.preferred_therapeutic_areas = []
        if self.notification_settings is None:
            self.notification_settings = {
                "query_completion": True,
                "data_updates": True,
                "system_alerts": True
            }

@dataclass
class UserProfile:
    """Comprehensive user profile for pharma analytics platform"""
    user_id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    access_level: AccessLevel
    department: str
    therapeutic_areas: List[str]
    data_access_permissions: List[str]  # table/schema permissions
    preferences: UserPreferences
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    session_timeout: int = 3600  # seconds
    total_queries: int = 0
    favorite_queries: List[Dict[str, Any]] = None
    recent_tables: List[str] = None
    
    def __post_init__(self):
        if self.favorite_queries is None:
            self.favorite_queries = []
        if self.recent_tables is None:
            self.recent_tables = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for API responses"""
        profile_dict = asdict(self)
        profile_dict['role'] = self.role.value
        profile_dict['access_level'] = self.access_level.value
        profile_dict['created_at'] = self.created_at.isoformat()
        profile_dict['last_login'] = self.last_login.isoformat() if self.last_login else None
        return profile_dict
    
    def can_access_table(self, table_name: str) -> bool:
        """Check if user can access specific table"""
        if self.access_level == AccessLevel.FULL:
            return True
        
        # Check specific table permissions
        for permission in self.data_access_permissions:
            if permission == "*" or table_name.startswith(permission):
                return True
        
        return False
    
    def add_favorite_query(self, query: str, description: str = None):
        """Add query to favorites"""
        favorite = {
            "id": str(uuid.uuid4()),
            "query": query,
            "description": description or query[:50] + "...",
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }
        self.favorite_queries.append(favorite)
        
        # Keep only latest 20 favorites
        if len(self.favorite_queries) > 20:
            self.favorite_queries = self.favorite_queries[-20:]
    
    def add_recent_table(self, table_name: str):
        """Track recently used tables"""
        if table_name in self.recent_tables:
            self.recent_tables.remove(table_name)
        
        self.recent_tables.insert(0, table_name)
        
        # Keep only last 10 tables
        if len(self.recent_tables) > 10:
            self.recent_tables = self.recent_tables[:10]

class UserProfileManager:
    """Manages user profiles with file-based storage"""
    
    def __init__(self, storage_path: str = "backend/auth/user_profiles.json"):
        self.storage_path = storage_path
        self.profiles: Dict[str, UserProfile] = {}
        self.load_profiles()
    
    def load_profiles(self):
        """Load user profiles from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for user_id, profile_data in data.items():
                    profile_data['role'] = UserRole(profile_data['role'])
                    profile_data['access_level'] = AccessLevel(profile_data['access_level'])
                    profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                    if profile_data['last_login']:
                        profile_data['last_login'] = datetime.fromisoformat(profile_data['last_login'])
                    
                    # Reconstruct preferences
                    pref_data = profile_data.get('preferences', {})
                    profile_data['preferences'] = UserPreferences(**pref_data)
                    
                    self.profiles[user_id] = UserProfile(**profile_data)
        except Exception as e:
            print(f"Error loading user profiles: {e}")
            self.profiles = {}
    
    def save_profiles(self):
        """Save user profiles to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {}
            for user_id, profile in self.profiles.items():
                data[user_id] = profile.to_dict()
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving user profiles: {e}")
    
    def create_user(self, username: str, email: str, full_name: str, 
                   role: UserRole, department: str = "Analytics",
                   therapeutic_areas: List[str] = None) -> UserProfile:
        """Create new user profile"""
        user_id = str(uuid.uuid4())
        
        # Set default access level based on role
        access_level = AccessLevel.BASIC
        if role in [UserRole.DATA_SCIENTIST, UserRole.ANALYST]:
            access_level = AccessLevel.ADVANCED
        elif role == UserRole.ADMIN:
            access_level = AccessLevel.FULL
        
        # Set default data permissions
        data_permissions = ["ENHANCED_NBA.*"]  # Default pharma tables
        if role == UserRole.REGULATORY:
            data_permissions.extend(["REGULATORY.*", "COMPLIANCE.*"])
        elif role == UserRole.COMMERCIAL:
            data_permissions.extend(["SALES.*", "MARKETING.*"])
        elif role == UserRole.MEDICAL_AFFAIRS:
            data_permissions.extend(["CLINICAL.*", "MEDICAL.*"])
        elif role == UserRole.ADMIN:
            data_permissions = ["*"]  # Full access
        
        profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            access_level=access_level,
            department=department,
            therapeutic_areas=therapeutic_areas or [],
            data_access_permissions=data_permissions,
            preferences=UserPreferences(),
            created_at=datetime.now()
        )
        
        self.profiles[user_id] = profile
        self.save_profiles()
        return profile
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.profiles.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get user profile by username"""
        for profile in self.profiles.values():
            if profile.username == username:
                return profile
        return None
    
    def update_user_login(self, user_id: str):
        """Update user's last login time"""
        if user_id in self.profiles:
            self.profiles[user_id].last_login = datetime.now()
            self.save_profiles()
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        if user_id in self.profiles:
            profile = self.profiles[user_id]
            for key, value in preferences.items():
                if hasattr(profile.preferences, key):
                    setattr(profile.preferences, key, value)
            self.save_profiles()
    
    def increment_query_count(self, user_id: str):
        """Increment user's query count"""
        if user_id in self.profiles:
            self.profiles[user_id].total_queries += 1
            self.save_profiles()

# Global instance
profile_manager = UserProfileManager()

def get_user_profile(user_id: str) -> Optional[UserProfile]:
    """Get user profile - convenience function"""
    return profile_manager.get_user(user_id)

def create_demo_users():
    """Create demo users for testing"""
    demo_users = [
        {
            "username": "analyst1",
            "email": "analyst1@pharma.com",
            "full_name": "Sarah Chen",
            "role": UserRole.ANALYST,
            "department": "Commercial Analytics",
            "therapeutic_areas": ["Oncology", "Diabetes"]
        },
        {
            "username": "medaffairs1",
            "email": "medaffairs1@pharma.com", 
            "full_name": "Dr. Michael Roberts",
            "role": UserRole.MEDICAL_AFFAIRS,
            "department": "Medical Affairs",
            "therapeutic_areas": ["Oncology", "Immunology"]
        },
        {
            "username": "datascientist1",
            "email": "ds1@pharma.com",
            "full_name": "Alex Kumar",
            "role": UserRole.DATA_SCIENTIST,
            "department": "Data Science",
            "therapeutic_areas": ["All"]
        }
    ]
    
    for user_data in demo_users:
        if not profile_manager.get_user_by_username(user_data["username"]):
            profile_manager.create_user(**user_data)

# Initialize demo users
create_demo_users()
