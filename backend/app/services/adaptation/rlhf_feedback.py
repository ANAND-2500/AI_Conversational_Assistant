"""
RLHF Feedback Module - collects and processes user feedback for model improvement.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    DETAILED = "detailed"


class FeedbackDocument(BaseModel):
    """Feedback document schema."""
    
    feedback_id: str
    session_id: str
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    feedback_type: FeedbackType
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5")
    comments: Optional[str] = None
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RLHFFeedback:
    """Manages RLHF feedback collection and analysis."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize RLHF feedback module.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db.feedback
    
    async def submit_feedback(
        self,
        session_id: str,
        feedback_type: FeedbackType,
        rating: Optional[int] = None,
        comments: Optional[str] = None,
        message_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_message: Optional[str] = None,
        assistant_response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackDocument:
        """
        Submit user feedback.
        
        Args:
            session_id: Session identifier
            feedback_type: Type of feedback
            rating: Optional rating (1-5)
            comments: Optional text comments
            message_id: Optional message identifier
            user_id: Optional user identifier
            user_message: Optional user message content
            assistant_response: Optional assistant response content
            metadata: Optional metadata
            
        Returns:
            Created feedback document
        """
        import uuid
        
        feedback = FeedbackDocument(
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            message_id=message_id,
            user_id=user_id,
            feedback_type=feedback_type,
            rating=rating,
            comments=comments,
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        await self.collection.insert_one(feedback.model_dump())
        
        logger.info(f"Feedback submitted: {feedback.feedback_id} for session {session_id}")
        
        return feedback
    
    async def get_feedback_by_session(
        self,
        session_id: str
    ) -> List[FeedbackDocument]:
        """
        Get all feedback for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of feedback documents
        """
        cursor = self.collection.find({"session_id": session_id})
        docs = await cursor.to_list(length=None)
        return [FeedbackDocument(**doc) for doc in docs]
    
    async def get_feedback_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary with statistics
        """
        # Build query
        query = {}
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        # Total feedback count
        total_count = await self.collection.count_documents(query)
        
        # Feedback type distribution
        type_pipeline = [
            {"$match": query},
            {"$group": {"_id": "$feedback_type", "count": {"$sum": 1}}}
        ]
        type_stats = await self.collection.aggregate(type_pipeline).to_list(None)
        
        # Average rating
        rating_pipeline = [
            {"$match": {**query, "rating": {"$ne": None}}},
            {"$group": {
                "_id": None,
                "avg_rating": {"$avg": "$rating"},
                "count": {"$sum": 1}
            }}
        ]
        rating_stats = await self.collection.aggregate(rating_pipeline).to_list(None)
        
        avg_rating = rating_stats[0]["avg_rating"] if rating_stats else None
        rating_count = rating_stats[0]["count"] if rating_stats else 0
        
        # Positive vs negative feedback
        positive_count = await self.collection.count_documents({
            **query,
            "$or": [
                {"feedback_type": "thumbs_up"},
                {"rating": {"$gte": 4}}
            ]
        })
        
        negative_count = await self.collection.count_documents({
            **query,
            "$or": [
                {"feedback_type": "thumbs_down"},
                {"rating": {"$lte": 2}}
            ]
        })
        
        return {
            "total_feedback": total_count,
            "feedback_by_type": {item["_id"]: item["count"] for item in type_stats},
            "average_rating": round(avg_rating, 2) if avg_rating else None,
            "total_ratings": rating_count,
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "satisfaction_rate": round(positive_count / total_count * 100, 2) if total_count > 0 else 0
        }
    
    async def get_high_quality_conversations(
        self,
        min_rating: int = 4,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get conversations with high feedback ratings for training.
        
        Args:
            min_rating: Minimum rating threshold
            limit: Maximum number of conversations
            
        Returns:
            List of high-quality conversation data
        """
        pipeline = [
            {
                "$match": {
                    "rating": {"$gte": min_rating},
                    "user_message": {"$ne": None},
                    "assistant_response": {"$ne": None}
                }
            },
            {
                "$sort": {"rating": -1, "timestamp": -1}
            },
            {
                "$limit": limit
            },
            {
                "$project": {
                    "session_id": 1,
                    "user_message": 1,
                    "assistant_response": 1,
                    "rating": 1,
                    "timestamp": 1
                }
            }
        ]
        
        results = await self.collection.aggregate(pipeline).to_list(None)
        
        logger.info(f"Retrieved {len(results)} high-quality conversations")
        
        return results
    
    async def get_low_quality_conversations(
        self,
        max_rating: int = 2,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get conversations with low feedback ratings for analysis.
        
        Args:
            max_rating: Maximum rating threshold
            limit: Maximum number of conversations
            
        Returns:
            List of low-quality conversation data
        """
        pipeline = [
            {
                "$match": {
                    "$or": [
                        {"rating": {"$lte": max_rating}},
                        {"feedback_type": "thumbs_down"}
                    ],
                    "user_message": {"$ne": None},
                    "assistant_response": {"$ne": None}
                }
            },
            {
                "$sort": {"rating": 1, "timestamp": -1}
            },
            {
                "$limit": limit
            },
            {
                "$project": {
                    "session_id": 1,
                    "user_message": 1,
                    "assistant_response": 1,
                    "rating": 1,
                    "comments": 1,
                    "timestamp": 1
                }
            }
        ]
        
        results = await self.collection.aggregate(pipeline).to_list(None)
        
        logger.info(f"Retrieved {len(results)} low-quality conversations for analysis")
        
        return results
    
    async def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in feedback data.
        
        Returns:
            Analysis results
        """
        # Get feedback from last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        # Common issues from comments
        negative_feedback = await self.collection.find({
            "$or": [
                {"feedback_type": "thumbs_down"},
                {"rating": {"$lte": 2}}
            ],
            "comments": {"$ne": None},
            "timestamp": {"$gte": week_ago}
        }).to_list(length=100)
        
        # Extract common keywords from negative feedback
        from collections import Counter
        import re
        
        all_words = []
        for feedback in negative_feedback:
            if feedback.get("comments"):
                words = re.findall(r'\b\w+\b', feedback["comments"].lower())
                all_words.extend(words)
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "was", "are", "were", "to", "of", "in", "for", "on", "with"}
        filtered_words = [w for w in all_words if w not in stop_words and len(w) > 3]
        
        common_issues = Counter(filtered_words).most_common(10)
        
        return {
            "negative_feedback_count": len(negative_feedback),
            "common_issue_keywords": [{"word": word, "count": count} for word, count in common_issues],
            "analysis_period": "last_7_days"
        }
    
    async def ensure_indexes(self):
        """Create necessary database indexes."""
        await self.collection.create_index("session_id")
        await self.collection.create_index("user_id")
        await self.collection.create_index("timestamp")
        await self.collection.create_index("rating")
        await self.collection.create_index("feedback_type")
        logger.info("Feedback indexes created")
