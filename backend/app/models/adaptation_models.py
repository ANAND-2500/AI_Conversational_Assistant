"""
Database models for the adaptation module.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class UserPreferencesModel(BaseModel):
    """User preferences database model."""
    user_id: str
    language: str = "en"
    tone: str = "friendly"
    name: Optional[str] = None
    topics_of_interest: List[str] = Field(default_factory=list)
    response_length: str = "medium"
    enable_memory: bool = True
    custom_instructions: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackModel(BaseModel):
    """Feedback database model."""
    feedback_id: str
    session_id: str
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    feedback_type: str
    rating: Optional[int] = None
    comments: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvaluationReportModel(BaseModel):
    """Evaluation report database model."""
    report_id: str
    created_at: datetime
    total_evaluations: int
    aggregate_metrics: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]
