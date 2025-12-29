"""
Prompt Personalizer - customizes prompts based on user preferences and memory.
"""
from typing import Dict, Any, Optional, List
import logging

from .preferences_manager import UserPreferences, PreferencesManager
from .memory_module import MemoryModule, ConversationTurn
from .prompt_templates import template_library, TemplateCategory
from .config import config

logger = logging.getLogger(__name__)


class PromptPersonalizer:
    """Personalizes prompts based on user context."""
    
    def __init__(
        self,
        preferences_manager: PreferencesManager,
        memory_module: MemoryModule
    ):
        """
        Initialize prompt personalizer.
        
        Args:
            preferences_manager: Preferences manager instance
            memory_module: Memory module instance
        """
        self.preferences_manager = preferences_manager
        self.memory_module = memory_module
    
    async def build_personalized_prompt(
        self,
        user_message: str,
        session_id: str,
        user_id: Optional[str] = None,
        template_category: Optional[TemplateCategory] = None
    ) -> Dict[str, str]:
        """
        Build a personalized prompt for the LLM.
        
        Args:
            user_message: User's message
            session_id: Session identifier
            user_id: Optional user identifier
            template_category: Optional template category to use
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Get user preferences
        if user_id:
            preferences = await self.preferences_manager.get_preferences(user_id)
        else:
            preferences = UserPreferences()  # Default preferences
        
        # Get conversation memory
        recent_context = await self.memory_module.get_recent_context(session_id)
        key_facts = await self.memory_module.get_key_facts(session_id)
        
        # Determine template to use
        if template_category:
            template = template_library.get_best_template(template_category)
            if not template:
                template = template_library.get_template("fallback")
        else:
            # Use personalized template if user has preferences
            if user_id and preferences.enable_memory:
                template = template_library.get_template("personalized_general")
            else:
                # Classify intent and select template
                template_id = self._classify_intent(user_message)
                template = template_library.get_template(template_id)
        
        if not template:
            template = template_library.get_template("fallback")
        
        # Build context strings
        context_str = self._format_recent_context(recent_context)
        facts_str = self._format_key_facts(key_facts)
        
        # Prepare template variables
        variables = {
            "user_message": user_message,
            "user_name": preferences.name or "there",
            "tone": preferences.tone,
            "response_length": preferences.response_length,
            "language": preferences.language,
            "custom_instructions": preferences.custom_instructions or "",
            "recent_context": context_str,
            "key_facts": facts_str
        }
        
        # Fill in the template
        try:
            system_prompt = template.system_prompt.format(**variables)
            user_prompt = template.user_prompt_template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in template: {e}, using fallback")
            # Fallback to simple prompt
            system_prompt = f"You are a helpful AI assistant. Respond in a {preferences.tone} tone."
            user_prompt = user_message
        
        # Apply response length guidance
        system_prompt = self._apply_length_preference(system_prompt, preferences.response_length)
        
        logger.info(f"Built personalized prompt for session {session_id} using template {template.template_id}")
        
        return {
            "system": system_prompt,
            "user": user_prompt,
            "template_id": template.template_id
        }
    
    def _classify_intent(self, message: str) -> str:
        """
        Classify user intent to select appropriate template.
        
        Args:
            message: User message
            
        Returns:
            Template ID
        """
        message_lower = message.lower()
        
        # Simple keyword-based classification
        # In production, use a proper intent classifier
        
        if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting_friendly"
        
        if any(word in message_lower for word in ["code", "program", "function", "debug", "error", "api"]):
            return "technical_support"
        
        if any(word in message_lower for word in ["write", "story", "poem", "creative", "imagine"]):
            return "creative_writing"
        
        # Default to Q&A
        return "qa_general"
    
    def _format_recent_context(self, turns: List[ConversationTurn]) -> str:
        """
        Format recent conversation turns for context.
        
        Args:
            turns: List of conversation turns
            
        Returns:
            Formatted context string
        """
        if not turns:
            return "No previous context."
        
        context_lines = []
        for turn in turns[-config.memory_window_size:]:
            role_label = "User" if turn.role == "user" else "Assistant"
            # Truncate long messages
            content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            context_lines.append(f"{role_label}: {content}")
        
        return "\n".join(context_lines)
    
    def _format_key_facts(self, facts: List[Any]) -> str:
        """
        Format key facts for context.
        
        Args:
            facts: List of key facts
            
        Returns:
            Formatted facts string
        """
        if not facts:
            return "No key facts extracted yet."
        
        fact_lines = [f"- {fact.fact}" for fact in facts[:5]]
        return "\n".join(fact_lines)
    
    def _apply_length_preference(self, system_prompt: str, length_pref: str) -> str:
        """
        Add length guidance to system prompt.
        
        Args:
            system_prompt: Original system prompt
            length_pref: Length preference (short/medium/detailed)
            
        Returns:
            Modified system prompt
        """
        length_instructions = {
            "short": "\n\nKeep your responses brief and concise (1-2 sentences when possible).",
            "medium": "\n\nProvide balanced responses with appropriate detail.",
            "detailed": "\n\nProvide comprehensive, detailed responses with examples and explanations."
        }
        
        instruction = length_instructions.get(length_pref, "")
        return system_prompt + instruction
    
    async def build_simple_prompt(
        self,
        user_message: str,
        system_instruction: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Build a simple non-personalized prompt.
        
        Args:
            user_message: User's message
            system_instruction: Optional system instruction
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        system_prompt = system_instruction or "You are a helpful AI assistant."
        
        return {
            "system": system_prompt,
            "user": user_message,
            "template_id": "simple"
        }
    
    async def get_prompt_for_continuation(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get a prompt for continuing a conversation.
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            
        Returns:
            Dictionary with prompt components
        """
        # Get recent context
        recent_context = await self.memory_module.get_recent_context(session_id)
        
        if not recent_context:
            return await self.build_simple_prompt(
                "Continue the conversation",
                "You are a helpful AI assistant."
            )
        
        # Build context-aware continuation prompt
        context_str = self._format_recent_context(recent_context)
        
        system_prompt = f"""You are a helpful AI assistant.
        
Previous conversation:
{context_str}

Continue the conversation naturally based on the context above."""
        
        return {
            "system": system_prompt,
            "user": "Continue the conversation",
            "template_id": "continuation"
        }
