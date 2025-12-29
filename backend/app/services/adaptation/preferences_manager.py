"""
User Preferences Manager - handles storage and retrieval of user preferences.
"""
from typing import Optional, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel, Field
import logging

from .config import config

logger = logging.getLogger(__name__)


class UserPreferences(BaseModel):
    """User preferences schema."""
    
    language: str = Field(default="en", description="Preferred language code")
    tone: str = Field(default="friendly", description="Response tone: friendly, professional, casual")
    name: Optional[str] = Field(default=None, description="User's preferred name")
    topics_of_interest: list[str] = Field(default_factory=list, description="Topics user is interested in")
    response_length: str = Field(default="medium", description="Preferred response length: short, medium, detailed")
    enable_memory: bool = Field(default=True, description="Enable conversation memory")
    custom_instructions: Optional[str] = Field(default=None, description="Custom instructions for the assistant")
    
    class Config:
        json_schema_extra = {
            "example": {
                "language": "en-IN",
                "tone": "friendly",
                "name": "Alex",
                "topics_of_interest": ["technology", "science"],
                "response_length": "medium",
                "enable_memory": True,
                "custom_instructions": "Always provide code examples when discussing programming"
            }
        }


class UserPreferencesDocument(BaseModel):
    """Complete user preferences document for database."""
    
    user_id: str
    preferences: UserPreferences
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)


class PreferencesManager:
    """Manages user preferences with caching."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize preferences manager.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db.user_preferences
        self._cache: Dict[str, UserPreferencesDocument] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    async def get_preferences(self, user_id: str) -> UserPreferences:
        """
        Get user preferences with caching.
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences object
        """
        # Check cache first
        if config.enable_preference_cache and user_id in self._cache:
            cache_age = (datetime.utcnow() - self._cache_timestamps[user_id]).total_seconds()
            if cache_age < config.cache_ttl_seconds:
                logger.debug(f"Cache hit for user {user_id}")
                return self._cache[user_id].preferences
        
        # Fetch from database
        doc = await self.collection.find_one({"user_id": user_id})
        
        if doc:
            pref_doc = UserPreferencesDocument(**doc)
            
            # Update cache
            if config.enable_preference_cache:
                self._cache[user_id] = pref_doc
                self._cache_timestamps[user_id] = datetime.utcnow()
            
            logger.info(f"Loaded preferences for user {user_id}")
            return pref_doc.preferences
        else:
            # Return default preferences
            logger.info(f"No preferences found for user {user_id}, using defaults")
            return UserPreferences()
    
    async def update_preferences(
        self, 
        user_id: str, 
        preferences: UserPreferences
    ) -> UserPreferencesDocument:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            preferences: New preferences
            
        Returns:
            Updated preferences document
        """
        now = datetime.utcnow()
        
        # Check if preferences exist
        existing = await self.collection.find_one({"user_id": user_id})
        
        if existing:
            # Update existing
            update_data = {
                "preferences": preferences.model_dump(),
                "updated_at": now,
                "version": existing.get("version", 1) + 1
            }
            
            await self.collection.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            
            logger.info(f"Updated preferences for user {user_id}")
        else:
            # Create new
            pref_doc = UserPreferencesDocument(
                user_id=user_id,
                preferences=preferences,
                created_at=now,
                updated_at=now
            )
            
            await self.collection.insert_one(pref_doc.model_dump())
            logger.info(f"Created preferences for user {user_id}")
        
        # Invalidate cache
        if user_id in self._cache:
            del self._cache[user_id]
            del self._cache_timestamps[user_id]
        
        # Fetch and return updated document
        updated_doc = await self.collection.find_one({"user_id": user_id})
        return UserPreferencesDocument(**updated_doc)
    
    async def delete_preferences(self, user_id: str) -> bool:
        """
        Delete user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.collection.delete_one({"user_id": user_id})
        
        # Invalidate cache
        if user_id in self._cache:
            del self._cache[user_id]
            del self._cache_timestamps[user_id]
        
        if result.deleted_count > 0:
            logger.info(f"Deleted preferences for user {user_id}")
            return True
        else:
            logger.warning(f"No preferences found to delete for user {user_id}")
            return False
    
    async def get_all_preferences(
        self, 
        skip: int = 0, 
        limit: int = 100
    ) -> list[UserPreferencesDocument]:
        """
        Get all user preferences (admin function).
        
        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List of preferences documents
        """
        cursor = self.collection.find().skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [UserPreferencesDocument(**doc) for doc in docs]
    
    async def get_preferences_stats(self) -> Dict[str, Any]:
        """
        Get statistics about user preferences.
        
        Returns:
            Dictionary with statistics
        """
        total_users = await self.collection.count_documents({})
        
        # Aggregate by tone
        tone_pipeline = [
            {"$group": {"_id": "$preferences.tone", "count": {"$sum": 1}}}
        ]
        tone_stats = await self.collection.aggregate(tone_pipeline).to_list(None)
        
        # Aggregate by language
        lang_pipeline = [
            {"$group": {"_id": "$preferences.language", "count": {"$sum": 1}}}
        ]
        lang_stats = await self.collection.aggregate(lang_pipeline).to_list(None)
        
        return {
            "total_users": total_users,
            "tone_distribution": {item["_id"]: item["count"] for item in tone_stats},
            "language_distribution": {item["_id"]: item["count"] for item in lang_stats}
        }
    
    def clear_cache(self):
        """Clear the preferences cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Preferences cache cleared")
    
    async def ensure_indexes(self):
        """Create necessary database indexes."""
        await self.collection.create_index("user_id", unique=True)
        await self.collection.create_index("updated_at")
        logger.info("Preferences indexes created")
