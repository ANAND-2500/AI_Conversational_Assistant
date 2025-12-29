"""
Memory Module - manages conversation memory and context retention.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field
import numpy as np
import logging

from .config import config
from .utils import embedding_generator, text_preprocessor

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    """Single conversation turn."""
    
    role: str = Field(description="Role: user or assistant")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KeyFact(BaseModel):
    """Extracted key fact from conversation."""
    
    fact: str = Field(description="The extracted fact")
    source_message: str = Field(description="Original message")
    confidence: float = Field(description="Confidence score 0-1")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = Field(default=None, description="Fact embedding")


class SessionMemory(BaseModel):
    """Memory for a conversation session."""
    
    session_id: str
    user_id: Optional[str] = None
    turns: List[ConversationTurn] = Field(default_factory=list)
    key_facts: List[KeyFact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    summary: Optional[str] = None


class MemoryModule:
    """Manages conversation memory with short-term and long-term storage."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize memory module.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.sessions_collection = db.conversation_sessions
        self.facts_collection = db.key_facts
    
    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add a conversation turn to memory.
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            user_id: Optional user identifier
            metadata: Optional metadata
            
        Returns:
            Created conversation turn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Update session in database
        await self.sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"turns": turn.model_dump()},
                "$set": {
                    "updated_at": datetime.utcnow(),
                    "user_id": user_id
                },
                "$setOnInsert": {
                    "session_id": session_id,
                    "created_at": datetime.utcnow(),
                    "key_facts": [],
                    "summary": None
                }
            },
            upsert=True
        )
        
        logger.debug(f"Added {role} turn to session {session_id}")
        
        # Extract key facts if it's a user message
        if role == "user" and config.enable_long_term_memory:
            await self._extract_key_facts(session_id, content)
        
        return turn
    
    async def get_recent_context(
        self,
        session_id: str,
        window_size: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Get recent conversation turns for context.
        
        Args:
            session_id: Session identifier
            window_size: Number of recent turns (default from config)
            
        Returns:
            List of recent conversation turns
        """
        if window_size is None:
            window_size = config.memory_window_size
        
        session = await self.sessions_collection.find_one({"session_id": session_id})
        
        if not session or not session.get("turns"):
            return []
        
        # Get last N turns
        turns_data = session["turns"][-window_size:]
        return [ConversationTurn(**turn) for turn in turns_data]
    
    async def get_key_facts(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[KeyFact]:
        """
        Get key facts extracted from conversation.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of facts to return
            
        Returns:
            List of key facts
        """
        session = await self.sessions_collection.find_one({"session_id": session_id})
        
        if not session or not session.get("key_facts"):
            return []
        
        facts_data = session["key_facts"][-limit:]
        return [KeyFact(**fact) for fact in facts_data]
    
    async def search_relevant_memories(
        self,
        session_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant past conversations using semantic similarity.
        
        Args:
            session_id: Session identifier
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant conversation turns with similarity scores
        """
        session = await self.sessions_collection.find_one({"session_id": session_id})
        
        if not session or not session.get("turns"):
            return []
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query)
        
        # Get embeddings for all turns (or use cached ones)
        relevant_turns = []
        
        for turn_data in session["turns"]:
            turn = ConversationTurn(**turn_data)
            
            # Generate embedding for turn content
            turn_embedding = embedding_generator.generate_embedding(turn.content)
            
            # Calculate similarity
            similarity = embedding_generator.cosine_similarity(query_embedding, turn_embedding)
            
            if similarity > config.similarity_threshold:
                relevant_turns.append({
                    "turn": turn,
                    "similarity": similarity
                })
        
        # Sort by similarity and return top K
        relevant_turns.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_turns[:top_k]
    
    async def get_session_summary(self, session_id: str) -> Optional[str]:
        """
        Get or generate session summary.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary or None
        """
        session = await self.sessions_collection.find_one({"session_id": session_id})
        
        if not session:
            return None
        
        # Return cached summary if available
        if session.get("summary"):
            return session["summary"]
        
        # Generate summary from turns
        turns = session.get("turns", [])
        if not turns:
            return None
        
        # Simple summary: concatenate recent user messages
        user_messages = [
            turn["content"] for turn in turns[-10:]
            if turn["role"] == "user"
        ]
        
        if user_messages:
            summary = "User discussed: " + "; ".join(user_messages[:3])
            
            # Cache the summary
            await self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"summary": summary}}
            )
            
            return summary
        
        return None
    
    async def _extract_key_facts(self, session_id: str, message: str):
        """
        Extract key facts from a user message.
        
        Args:
            session_id: Session identifier
            message: User message
        """
        # Simple fact extraction based on patterns
        # In production, use NER or more sophisticated methods
        
        fact_patterns = [
            r"my name is (\w+)",
            r"i am (\w+)",
            r"i like (.+)",
            r"i work (?:at|for) (.+)",
            r"i live in (.+)",
        ]
        
        import re
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, message.lower())
            for match in matches:
                fact_text = f"User mentioned: {match}"
                
                # Generate embedding
                embedding = embedding_generator.generate_embedding(fact_text)
                
                fact = KeyFact(
                    fact=fact_text,
                    source_message=message,
                    confidence=0.8,  # Simple confidence score
                    timestamp=datetime.utcnow(),
                    embedding=embedding.tolist()
                )
                
                # Add to session
                await self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$push": {"key_facts": fact.model_dump()}}
                )
                
                logger.info(f"Extracted key fact from session {session_id}: {fact_text}")
    
    async def get_full_session(self, session_id: str) -> Optional[SessionMemory]:
        """
        Get complete session memory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session memory object or None
        """
        session = await self.sessions_collection.find_one({"session_id": session_id})
        
        if not session:
            return None
        
        return SessionMemory(**session)
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.sessions_collection.delete_one({"session_id": session_id})
        
        if result.deleted_count > 0:
            logger.info(f"Deleted session {session_id}")
            return True
        else:
            logger.warning(f"Session {session_id} not found")
            return False
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[SessionMemory]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions
            
        Returns:
            List of session memories
        """
        cursor = self.sessions_collection.find(
            {"user_id": user_id}
        ).sort("updated_at", -1).limit(limit)
        
        sessions = await cursor.to_list(length=limit)
        return [SessionMemory(**session) for session in sessions]
    
    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Delete sessions older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted sessions
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.sessions_collection.delete_many({
            "updated_at": {"$lt": cutoff_date}
        })
        
        logger.info(f"Deleted {result.deleted_count} old sessions")
        return result.deleted_count
    
    async def ensure_indexes(self):
        """Create necessary database indexes."""
        await self.sessions_collection.create_index("session_id", unique=True)
        await self.sessions_collection.create_index("user_id")
        await self.sessions_collection.create_index("updated_at")
        logger.info("Memory module indexes created")
