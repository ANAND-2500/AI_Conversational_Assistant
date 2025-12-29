"""
API routes for personalization and adaptation features.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.adaptation.preferences_manager import (
    PreferencesManager,
    UserPreferences,
    UserPreferencesDocument
)
from ..services.adaptation.rlhf_feedback import RLHFFeedback, FeedbackType
from ..services.adaptation.evaluator import Evaluator
from ..services.adaptation.fine_tuning_prep import FineTuningPrep
from ..services.adaptation.report_generator import report_generator

# Request/Response models
class PreferencesUpdateRequest(BaseModel):
    """Request model for updating preferences."""
    language: Optional[str] = None
    tone: Optional[str] = None
    name: Optional[str] = None
    topics_of_interest: Optional[list[str]] = None
    response_length: Optional[str] = None
    enable_memory: Optional[bool] = None
    custom_instructions: Optional[str] = None


class FeedbackSubmitRequest(BaseModel):
    """Request model for submitting feedback."""
    session_id: str
    feedback_type: FeedbackType
    rating: Optional[int] = Field(None, ge=1, le=5)
    comments: Optional[str] = None
    message_id: Optional[str] = None
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None


class EvaluationRunRequest(BaseModel):
    """Request model for running evaluation."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_samples: int = 100


class FineTuningPrepRequest(BaseModel):
    """Request model for fine-tuning preparation."""
    output_dir: str
    min_rating: int = 4
    max_conversations: int = 1000
    train_split: float = 0.8
    format_type: str = "openai"
    anonymize: bool = True


# Create router
router = APIRouter(prefix="/api/v1", tags=["adaptation"])


# Dependency to get database (placeholder - implement based on your setup)
async def get_db():
    """Get database connection."""
    # TODO: Implement database connection
    # from motor.motor_asyncio import AsyncIOMotorClient
    # client = AsyncIOMotorClient(settings.MONGO_URI)
    # return client.ai_assistant
    raise HTTPException(
        status_code=501,
        detail="Database connection not implemented"
    )


# Preferences endpoints
@router.get("/preferences")
async def get_preferences(
    user_id: str,
    db = Depends(get_db)
):
    """
    Get user preferences.
    
    Args:
        user_id: User identifier
        
    Returns:
        User preferences
    """
    preferences_manager = PreferencesManager(db)
    preferences = await preferences_manager.get_preferences(user_id)
    return {"preferences": preferences.model_dump()}


@router.put("/preferences")
async def update_preferences(
    user_id: str,
    request: PreferencesUpdateRequest,
    db = Depends(get_db)
):
    """
    Update user preferences.
    
    Args:
        user_id: User identifier
        request: Preferences update request
        
    Returns:
        Updated preferences
    """
    preferences_manager = PreferencesManager(db)
    
    # Get current preferences
    current_prefs = await preferences_manager.get_preferences(user_id)
    
    # Update with new values
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(current_prefs, key, value)
    
    # Save
    updated_doc = await preferences_manager.update_preferences(user_id, current_prefs)
    
    return {
        "message": "Preferences updated successfully",
        "preferences": updated_doc.preferences.model_dump()
    }


@router.delete("/preferences")
async def delete_preferences(
    user_id: str,
    db = Depends(get_db)
):
    """
    Delete user preferences.
    
    Args:
        user_id: User identifier
        
    Returns:
        Success message
    """
    preferences_manager = PreferencesManager(db)
    deleted = await preferences_manager.delete_preferences(user_id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Preferences not found"
        )
    
    return {"message": "Preferences deleted successfully"}


# Feedback endpoints
@router.post("/feedback")
async def submit_feedback(
    request: FeedbackSubmitRequest,
    user_id: Optional[str] = None,
    db = Depends(get_db)
):
    """
    Submit user feedback.
    
    Args:
        request: Feedback submission request
        user_id: Optional user identifier
        
    Returns:
        Feedback confirmation
    """
    rlhf_feedback = RLHFFeedback(db)
    
    feedback = await rlhf_feedback.submit_feedback(
        session_id=request.session_id,
        feedback_type=request.feedback_type,
        rating=request.rating,
        comments=request.comments,
        message_id=request.message_id,
        user_id=user_id,
        user_message=request.user_message,
        assistant_response=request.assistant_response
    )
    
    return {
        "message": "Feedback submitted successfully",
        "feedback_id": feedback.feedback_id
    }


@router.get("/feedback/stats")
async def get_feedback_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db = Depends(get_db)
):
    """
    Get feedback statistics.
    
    Args:
        start_date: Optional start date
        end_date: Optional end date
        
    Returns:
        Feedback statistics
    """
    rlhf_feedback = RLHFFeedback(db)
    stats = await rlhf_feedback.get_feedback_stats(start_date, end_date)
    return stats


# Admin evaluation endpoints
@router.post("/admin/evaluation/run")
async def run_evaluation(
    request: EvaluationRunRequest,
    db = Depends(get_db)
):
    """
    Run model evaluation (admin only).
    
    Args:
        request: Evaluation run request
        
    Returns:
        Evaluation report
    """
    evaluator = Evaluator(db)
    
    report = await evaluator.run_evaluation(
        start_date=request.start_date,
        end_date=request.end_date,
        max_samples=request.max_samples
    )
    
    # Generate HTML report
    html_path = report_generator.generate_html_report(report.model_dump())
    
    return {
        "message": "Evaluation completed",
        "report": report.model_dump(),
        "html_report_path": html_path
    }


@router.get("/admin/evaluation/reports")
async def get_evaluation_reports(
    limit: int = 10,
    db = Depends(get_db)
):
    """
    Get evaluation reports (admin only).
    
    Args:
        limit: Maximum number of reports
        
    Returns:
        List of evaluation reports
    """
    evaluator = Evaluator(db)
    reports = await evaluator.get_evaluation_reports(limit)
    
    return {
        "reports": [report.model_dump() for report in reports]
    }


@router.get("/admin/evaluation/compare")
async def compare_evaluations(
    report_id_1: str,
    report_id_2: str,
    db = Depends(get_db)
):
    """
    Compare two evaluation reports (admin only).
    
    Args:
        report_id_1: First report ID
        report_id_2: Second report ID
        
    Returns:
        Comparison results
    """
    evaluator = Evaluator(db)
    comparison = await evaluator.compare_evaluations(report_id_1, report_id_2)
    
    # Generate comparison chart
    chart_path = report_generator.create_comparison_chart(
        comparison,
        f"comparison_{report_id_1[:8]}_{report_id_2[:8]}.png"
    )
    
    return {
        "comparison": comparison,
        "chart_path": chart_path
    }


@router.post("/admin/finetuning/prepare")
async def prepare_finetuning_dataset(
    request: FineTuningPrepRequest,
    db = Depends(get_db)
):
    """
    Prepare fine-tuning dataset (admin only).
    
    Args:
        request: Fine-tuning preparation request
        
    Returns:
        Dataset metadata
    """
    fine_tuning_prep = FineTuningPrep(db)
    
    metadata = await fine_tuning_prep.prepare_dataset(
        output_dir=request.output_dir,
        min_rating=request.min_rating,
        max_conversations=request.max_conversations,
        train_split=request.train_split,
        format_type=request.format_type,
        anonymize=request.anonymize
    )
    
    return {
        "message": "Fine-tuning dataset prepared",
        "metadata": metadata
    }


@router.get("/admin/preferences/stats")
async def get_preferences_stats(
    db = Depends(get_db)
):
    """
    Get preferences statistics (admin only).
    
    Returns:
        Preferences statistics
    """
    preferences_manager = PreferencesManager(db)
    stats = await preferences_manager.get_preferences_stats()
    return stats
