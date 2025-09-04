"""
Enhanced FastAPI endpoints for user profiles and chat history
Supports modern agentic approach with pharma-specific features
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from backend.auth.user_profile import (
    UserProfile, UserRole, profile_manager, 
    get_user_profile, create_demo_users
)
from backend.history.enhanced_chat_history import (
    ChatHistoryManager, MessageType, MessageStatus,
    get_chat_history_manager, ChatMessage, Conversation
)
from backend.agents.agentic_orchestrator import AgenticOrchestrator

# Pydantic Models
class CreateConversationRequest(BaseModel):
    user_id: str
    title: Optional[str] = None
    therapeutic_area: Optional[str] = None

class SendMessageRequest(BaseModel):
    conversation_id: str
    user_id: str
    content: str
    message_type: str = "user_query"

class AgentQueryRequest(BaseModel):
    query: str
    user_id: str
    session_id: str
    conversation_id: Optional[str] = None
    therapeutic_area: Optional[str] = None

class UpdatePreferencesRequest(BaseModel):
    preferences: Dict[str, Any]

# Enhanced Main Application
def create_enhanced_app() -> FastAPI:
    app = FastAPI(title="Pharma NL2Q Analytics Platform", version="2.0.0")
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize services
    orchestrator = AgenticOrchestrator()
    chat_manager = get_chat_history_manager()
    
    # Initialize demo data
    create_demo_users()
    
    # ==================== USER PROFILE ENDPOINTS ====================
    
    @app.get("/api/user/profile/{user_id}")
    async def get_user_profile_endpoint(user_id: str):
        """Get user profile by ID"""
        try:
            profile = get_user_profile(user_id)
            if not profile:
                raise HTTPException(status_code=404, detail="User not found")
            return profile.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/user/profile/username/{username}")
    async def get_user_by_username(username: str):
        """Get user profile by username"""
        try:
            profile = profile_manager.get_user_by_username(username)
            if not profile:
                raise HTTPException(status_code=404, detail="User not found")
            return profile.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/api/user/profile/{user_id}/preferences")
    async def update_user_preferences(user_id: str, request: UpdatePreferencesRequest):
        """Update user preferences"""
        try:
            profile_manager.update_user_preferences(user_id, request.preferences)
            return {"status": "success", "message": "Preferences updated"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/user/profile/{user_id}/favorite-query")
    async def add_favorite_query(user_id: str, query: str, description: str = None):
        """Add query to user favorites"""
        try:
            profile = get_user_profile(user_id)
            if not profile:
                raise HTTPException(status_code=404, detail="User not found")
            
            profile.add_favorite_query(query, description)
            profile_manager.save_profiles()
            return {"status": "success", "message": "Query added to favorites"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/user/profile/{user_id}/analytics")
    async def get_user_analytics(user_id: str, days: int = 30):
        """Get user usage analytics"""
        try:
            analytics = chat_manager.get_usage_analytics(user_id, days)
            return analytics
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== CHAT HISTORY ENDPOINTS ====================
    
    @app.get("/api/chat/conversations/{user_id}")
    async def get_user_conversations(user_id: str, limit: int = 50, include_archived: bool = False):
        """Get user's conversations"""
        try:
            conversations = chat_manager.get_user_conversations(user_id, limit, include_archived)
            return [conv.to_dict() for conv in conversations]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/chat/conversation/{conversation_id}")
    async def get_conversation(conversation_id: str):
        """Get specific conversation with messages"""
        try:
            conversation = chat_manager.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            return conversation.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/chat/conversation")
    async def create_conversation(request: CreateConversationRequest):
        """Create new conversation"""
        try:
            conversation = chat_manager.create_conversation(
                user_id=request.user_id,
                title=request.title,
                therapeutic_area=request.therapeutic_area
            )
            return conversation.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/chat/message")
    async def add_message(request: SendMessageRequest):
        """Add message to conversation"""
        try:
            message = chat_manager.add_message(
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                message_type=MessageType(request.message_type),
                content=request.content
            )
            return message.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/chat/search/{user_id}")
    async def search_conversations(user_id: str, q: str, limit: int = 20):
        """Search user's conversations"""
        try:
            conversations = chat_manager.search_conversations(user_id, q, limit)
            return [conv.to_dict() for conv in conversations]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/api/chat/conversation/{conversation_id}/title")
    async def update_conversation_title(conversation_id: str, title: str):
        """Update conversation title"""
        try:
            chat_manager.update_conversation_title(conversation_id, title)
            return {"status": "success", "message": "Title updated"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/api/chat/conversation/{conversation_id}/favorite")
    async def toggle_conversation_favorite(conversation_id: str):
        """Toggle conversation favorite status"""
        try:
            chat_manager.toggle_favorite(conversation_id)
            return {"status": "success", "message": "Favorite status updated"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/api/chat/conversation/{conversation_id}/archive")
    async def archive_conversation(conversation_id: str):
        """Archive conversation"""
        try:
            chat_manager.archive_conversation(conversation_id)
            return {"status": "success", "message": "Conversation archived"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== ENHANCED AGENT ENDPOINTS ====================
    
    @app.post("/api/agent/query")
    async def enhanced_agent_query(request: AgentQueryRequest):
        """
        Enhanced agent query with chat history integration
        """
        try:
            start_time = datetime.now()
            
            # Get or create conversation
            if request.conversation_id:
                conversation = chat_manager.get_conversation(request.conversation_id)
                if not conversation:
                    raise HTTPException(status_code=404, detail="Conversation not found")
            else:
                # Create new conversation
                conversation = chat_manager.create_conversation(
                    user_id=request.user_id,
                    title=request.query[:50] + "..." if len(request.query) > 50 else request.query,
                    therapeutic_area=request.therapeutic_area
                )
            
            # Add user message to chat history
            user_message = chat_manager.add_message(
                conversation_id=conversation.conversation_id,
                user_id=request.user_id,
                message_type=MessageType.USER_QUERY,
                content=request.query,
                status=MessageStatus.COMPLETED
            )
            
            # Get user profile for context
            user_profile = get_user_profile(request.user_id)
            
            # Enhanced context for agent
            enhanced_context = {
                "user_id": request.user_id,
                "conversation_id": conversation.conversation_id,
                "user_role": user_profile.role.value if user_profile else "analyst",
                "therapeutic_areas": user_profile.therapeutic_areas if user_profile else [],
                "conversation_context": conversation.get_recent_context(5),
                "user_permissions": user_profile.data_access_permissions if user_profile else []
            }
            
            # Create plan with enhanced orchestrator
            plan = await orchestrator.create_plan(
                user_query=request.query,
                user_id=request.user_id,
                session_id=conversation.conversation_id,
                context=enhanced_context
            )
            
            # Update user query count
            if user_profile:
                profile_manager.increment_query_count(request.user_id)
            
            # Add plan creation message
            plan_message = chat_manager.add_message(
                conversation_id=conversation.conversation_id,
                user_id="system",
                message_type=MessageType.PLAN_UPDATE,
                content="AI Agent has created an execution plan for your query.",
                metadata={
                    "plan_id": plan.plan_id,
                    "reasoning_steps": plan.reasoning_steps,
                    "estimated_cost": plan.estimated_cost
                },
                status=MessageStatus.COMPLETED,
                response_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
            return {
                "plan_id": plan.plan_id,
                "status": plan.status.value,
                "conversation_id": conversation.conversation_id,
                "message_id": user_message.message_id,
                "plan_message_id": plan_message.message_id
            }
            
        except Exception as e:
            print(f"Error in enhanced agent query: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/agent/plan/{plan_id}/status")
    async def get_plan_status(plan_id: str):
        """Get plan execution status"""
        try:
            status = await orchestrator.get_plan_status(plan_id)
            return status.to_dict() if status else {"error": "Plan not found"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/agent/plan/{plan_id}/approve")
    async def approve_plan(plan_id: str, approver_id: str):
        """Approve plan for execution"""
        try:
            result = await orchestrator.approve_plan(plan_id, approver_id)
            return {"status": "success", "plan_id": plan_id, "approved_by": approver_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== PHARMA-SPECIFIC ENDPOINTS ====================
    
    @app.get("/api/pharma/therapeutic-areas")
    async def get_therapeutic_areas():
        """Get available therapeutic areas"""
        return {
            "therapeutic_areas": [
                "Oncology", "Diabetes", "Cardiovascular", "Immunology", 
                "Neurology", "Infectious Disease", "Respiratory", "Dermatology",
                "Gastroenterology", "Ophthalmology", "Rheumatology", "Psychiatry"
            ]
        }
    
    @app.get("/api/pharma/data-sources")
    async def get_available_data_sources():
        """Get available pharma data sources"""
        return {
            "data_sources": [
                {
                    "name": "ENHANCED_NBA",
                    "description": "Enhanced Next Best Action commercial data",
                    "tables": ["rx_facts", "hcp_master", "product_master", "patient_master"]
                },
                {
                    "name": "CLINICAL_TRIALS",
                    "description": "Clinical trial data and outcomes",
                    "tables": ["trial_data", "patient_outcomes", "adverse_events"]
                },
                {
                    "name": "MARKET_RESEARCH",
                    "description": "Market research and competitive intelligence", 
                    "tables": ["market_data", "competitor_analysis", "market_share"]
                }
            ]
        }
    
    @app.get("/api/pharma/compliance-templates")
    async def get_compliance_templates():
        """Get pharma compliance query templates"""
        return {
            "templates": [
                {
                    "category": "Adverse Events",
                    "queries": [
                        "Show adverse events by therapeutic area in the last quarter",
                        "Analyze safety signals for specific products",
                        "Compare adverse event rates across products"
                    ]
                },
                {
                    "category": "Clinical Performance",
                    "queries": [
                        "Analyze patient outcomes by treatment protocol",
                        "Compare efficacy metrics across studies",
                        "Review treatment adherence rates"
                    ]
                },
                {
                    "category": "Commercial Analytics",
                    "queries": [
                        "Show prescriber behavior analysis",
                        "Analyze market share trends",
                        "Review promotional campaign effectiveness"
                    ]
                }
            ]
        }
    
    # ==================== HEALTH CHECK ====================
    
    @app.get("/health")
    async def health_check():
        """Enhanced health check with service status"""
        try:
            # Check database connections
            db_status = "healthy"  # Add actual DB health check
            
            # Check chat history database
            chat_db_status = "healthy"
            try:
                chat_manager.get_user_conversations("test", limit=1)
            except:
                chat_db_status = "degraded"
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "database": db_status,
                    "chat_history": chat_db_status,
                    "user_profiles": "healthy",
                    "agentic_orchestrator": "healthy"
                },
                "version": "2.0.0"
            }
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    return app

# Create the enhanced app instance
enhanced_app = create_enhanced_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(enhanced_app, host="0.0.0.0", port=8000)
