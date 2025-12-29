"""
Tests for personalization features.
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from app.services.adaptation.preferences_manager import (
    PreferencesManager,
    UserPreferences,
    UserPreferencesDocument
)
from app.services.adaptation.memory_module import (
    MemoryModule,
    ConversationTurn,
    SessionMemory
)
from app.services.adaptation.prompt_personalizer import PromptPersonalizer


@pytest.fixture
def mock_db():
    """Create mock database."""
    db = MagicMock()
    db.user_preferences = AsyncMock()
    db.conversation_sessions = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_preferences_manager_get_default(mock_db):
    """Test getting default preferences for new user."""
    mock_db.user_preferences.find_one = AsyncMock(return_value=None)
    
    manager = PreferencesManager(mock_db)
    prefs = await manager.get_preferences("user123")
    
    assert prefs.language == "en"
    assert prefs.tone == "friendly"
    assert prefs.response_length == "medium"


@pytest.mark.asyncio
async def test_preferences_manager_update(mock_db):
    """Test updating user preferences."""
    mock_db.user_preferences.find_one = AsyncMock(return_value=None)
    mock_db.user_preferences.insert_one = AsyncMock()
    mock_db.user_preferences.find_one = AsyncMock(return_value={
        "user_id": "user123",
        "preferences": {
            "language": "en-IN",
            "tone": "professional",
            "name": "Test User",
            "topics_of_interest": ["tech"],
            "response_length": "detailed",
            "enable_memory": True,
            "custom_instructions": None
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "version": 1
    })
    
    manager = PreferencesManager(mock_db)
    new_prefs = UserPreferences(
        language="en-IN",
        tone="professional",
        name="Test User",
        response_length="detailed"
    )
    
    result = await manager.update_preferences("user123", new_prefs)
    
    assert result.user_id == "user123"
    assert result.preferences.tone == "professional"


@pytest.mark.asyncio
async def test_memory_module_add_turn(mock_db):
    """Test adding conversation turn."""
    mock_db.conversation_sessions.update_one = AsyncMock()
    
    memory = MemoryModule(mock_db)
    turn = await memory.add_turn(
        session_id="session123",
        role="user",
        content="Hello, how are you?",
        user_id="user123"
    )
    
    assert turn.role == "user"
    assert turn.content == "Hello, how are you?"


@pytest.mark.asyncio
async def test_memory_module_get_recent_context(mock_db):
    """Test getting recent conversation context."""
    mock_db.conversation_sessions.find_one = AsyncMock(return_value={
        "session_id": "session123",
        "turns": [
            {"role": "user", "content": "Hello", "timestamp": datetime.utcnow(), "metadata": {}},
            {"role": "assistant", "content": "Hi there!", "timestamp": datetime.utcnow(), "metadata": {}},
            {"role": "user", "content": "How are you?", "timestamp": datetime.utcnow(), "metadata": {}}
        ]
    })
    
    memory = MemoryModule(mock_db)
    context = await memory.get_recent_context("session123", window_size=2)
    
    assert len(context) == 2
    assert context[0].role == "assistant"
    assert context[1].role == "user"


@pytest.mark.asyncio
async def test_prompt_personalizer_build_prompt(mock_db):
    """Test building personalized prompt."""
    # Mock preferences manager
    mock_db.user_preferences.find_one = AsyncMock(return_value={
        "user_id": "user123",
        "preferences": {
            "language": "en",
            "tone": "friendly",
            "name": "Alice",
            "topics_of_interest": [],
            "response_length": "medium",
            "enable_memory": True,
            "custom_instructions": None
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "version": 1
    })
    
    # Mock memory module
    mock_db.conversation_sessions.find_one = AsyncMock(return_value={
        "session_id": "session123",
        "turns": [
            {"role": "user", "content": "Hello", "timestamp": datetime.utcnow(), "metadata": {}}
        ],
        "key_facts": []
    })
    
    prefs_manager = PreferencesManager(mock_db)
    memory_module = MemoryModule(mock_db)
    personalizer = PromptPersonalizer(prefs_manager, memory_module)
    
    prompt = await personalizer.build_personalized_prompt(
        user_message="What's the weather like?",
        session_id="session123",
        user_id="user123"
    )
    
    assert "system" in prompt
    assert "user" in prompt
    assert "Alice" in prompt["system"] or "there" in prompt["system"]


def test_preferences_validation():
    """Test preferences validation."""
    prefs = UserPreferences(
        language="en",
        tone="friendly",
        response_length="medium"
    )
    
    assert prefs.language == "en"
    assert prefs.tone == "friendly"
    assert prefs.enable_memory == True  # default value


def test_conversation_turn_creation():
    """Test conversation turn creation."""
    turn = ConversationTurn(
        role="user",
        content="Test message"
    )
    
    assert turn.role == "user"
    assert turn.content == "Test message"
    assert isinstance(turn.timestamp, datetime)
