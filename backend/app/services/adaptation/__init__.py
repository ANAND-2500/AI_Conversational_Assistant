"""
Learning & Adaptation Module for AI Conversational Assistant.

This module provides personalization, model improvement, and evaluation capabilities.
"""

from .config import config, AdaptationConfig
from .preferences_manager import PreferencesManager
from .memory_module import MemoryModule
from .prompt_personalizer import PromptPersonalizer

__all__ = [
    "config",
    "AdaptationConfig",
    "PreferencesManager",
    "MemoryModule",
    "PromptPersonalizer",
]

__version__ = "1.0.0"
