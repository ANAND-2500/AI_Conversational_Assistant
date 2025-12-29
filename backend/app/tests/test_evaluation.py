"""
Tests for evaluation features.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from app.services.adaptation.metrics_calculator import MetricsCalculator
from app.services.adaptation.evaluator import Evaluator
from app.services.adaptation.rlhf_feedback import RLHFFeedback, FeedbackType


@pytest.fixture
def mock_db():
    """Create mock database."""
    db = MagicMock()
    db.feedback = AsyncMock()
    db.conversation_sessions = AsyncMock()
    db.evaluations = AsyncMock()
    return db


def test_metrics_calculator_exact_match():
    """Test exact match metric."""
    calc = MetricsCalculator()
    
    score = calc.calculate_exact_match("hello world", "hello world")
    assert score == 1.0
    
    score = calc.calculate_exact_match("hello world", "goodbye world")
    assert score == 0.0
    
    # Case insensitive
    score = calc.calculate_exact_match("Hello World", "hello world", case_sensitive=False)
    assert score == 1.0


def test_metrics_calculator_f1_score():
    """Test F1 score calculation."""
    calc = MetricsCalculator()
    
    score = calc.calculate_f1_score(
        "the quick brown fox",
        "the quick brown fox"
    )
    assert score == 1.0
    
    score = calc.calculate_f1_score(
        "the quick brown fox",
        "the lazy dog"
    )
    assert 0.0 < score < 1.0


def test_metrics_calculator_response_length_score():
    """Test response length scoring."""
    calc = MetricsCalculator()
    
    # Perfect medium length
    score = calc.calculate_response_length_score(
        " ".join(["word"] * 100),
        preferred_length="medium"
    )
    assert score == 1.0
    
    # Too short for medium
    score = calc.calculate_response_length_score(
        "short",
        preferred_length="medium"
    )
    assert score < 1.0


def test_metrics_calculator_toxicity_score():
    """Test toxicity detection."""
    calc = MetricsCalculator()
    
    # Clean text
    score = calc.calculate_toxicity_score("This is a nice message")
    assert score == 0.0
    
    # Toxic text
    score = calc.calculate_toxicity_score("You are stupid and dumb")
    assert score > 0.0


@pytest.mark.asyncio
async def test_rlhf_feedback_submit(mock_db):
    """Test feedback submission."""
    mock_db.feedback.insert_one = AsyncMock()
    
    rlhf = RLHFFeedback(mock_db)
    feedback = await rlhf.submit_feedback(
        session_id="session123",
        feedback_type=FeedbackType.RATING,
        rating=5,
        comments="Great response!",
        user_id="user123"
    )
    
    assert feedback.session_id == "session123"
    assert feedback.rating == 5
    assert feedback.feedback_type == FeedbackType.RATING


@pytest.mark.asyncio
async def test_rlhf_feedback_stats(mock_db):
    """Test feedback statistics."""
    # Mock aggregation results
    mock_db.feedback.count_documents = AsyncMock(return_value=100)
    mock_db.feedback.aggregate = AsyncMock()
    mock_db.feedback.aggregate.return_value.to_list = AsyncMock(return_value=[
        {"_id": "thumbs_up", "count": 60},
        {"_id": "thumbs_down", "count": 40}
    ])
    
    # Mock rating stats
    async def mock_aggregate_rating(pipeline):
        mock_cursor = AsyncMock()
        if any("avg_rating" in str(stage) for stage in pipeline):
            mock_cursor.to_list = AsyncMock(return_value=[
                {"_id": None, "avg_rating": 4.2, "count": 80}
            ])
        else:
            mock_cursor.to_list = AsyncMock(return_value=[])
        return mock_cursor
    
    mock_db.feedback.aggregate = mock_aggregate_rating
    
    rlhf = RLHFFeedback(mock_db)
    stats = await rlhf.get_feedback_stats()
    
    assert "total_feedback" in stats
    assert "average_rating" in stats


@pytest.mark.asyncio
async def test_evaluator_run_evaluation(mock_db):
    """Test running evaluation."""
    # Mock feedback data
    mock_db.feedback.find = MagicMock()
    mock_cursor = AsyncMock()
    mock_cursor.limit = MagicMock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=[
        {
            "session_id": "session123",
            "user_message": "Hello",
            "assistant_response": "Hi there!",
            "rating": 5,
            "timestamp": datetime.utcnow()
        }
    ])
    mock_db.feedback.find.return_value = mock_cursor
    
    # Mock session data
    mock_db.conversation_sessions.find_one = AsyncMock(return_value={
        "session_id": "session123",
        "turns": [
            {"role": "user", "content": "Hello", "timestamp": datetime.utcnow()}
        ]
    })
    
    # Mock evaluation insert
    mock_db.evaluations.insert_one = AsyncMock()
    
    evaluator = Evaluator(mock_db)
    report = await evaluator.run_evaluation(max_samples=10)
    
    assert report.total_evaluations >= 0
    assert "aggregate_metrics" in report.model_dump()
    assert len(report.recommendations) > 0


def test_metrics_all_metrics():
    """Test calculating all metrics at once."""
    calc = MetricsCalculator()
    
    metrics = calc.calculate_all_metrics(
        reference="The quick brown fox jumps over the lazy dog",
        candidate="The quick brown fox jumps over the lazy dog"
    )
    
    assert "exact_match" in metrics
    assert "f1_score" in metrics
    assert "bleu_score" in metrics
    assert "cosine_similarity" in metrics
    assert metrics["exact_match"] == 1.0
    assert metrics["f1_score"] == 1.0
