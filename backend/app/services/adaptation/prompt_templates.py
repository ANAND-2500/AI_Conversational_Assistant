"""
Prompt Template Library - manages prompt templates for different use cases.
"""
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class TemplateCategory(str, Enum):
    """Template categories."""
    GREETING = "greeting"
    QA = "qa"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    FALLBACK = "fallback"
    PERSONALIZED = "personalized"


class PromptTemplate(BaseModel):
    """Prompt template model."""
    
    template_id: str
    category: TemplateCategory
    name: str
    system_prompt: str
    user_prompt_template: str
    variables: List[str]
    version: str = "1.0"
    active: bool = True
    performance_score: float = 0.0
    usage_count: int = 0


class PromptTemplateLibrary:
    """Library of prompt templates."""
    
    def __init__(self):
        """Initialize template library with default templates."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        
        # Greeting template
        self.templates["greeting_friendly"] = PromptTemplate(
            template_id="greeting_friendly",
            category=TemplateCategory.GREETING,
            name="Friendly Greeting",
            system_prompt="""You are a friendly and helpful AI assistant. 
Greet the user warmly and ask how you can help them today.""",
            user_prompt_template="User has just started a conversation.",
            variables=[],
            version="1.0"
        )
        
        # Q&A template
        self.templates["qa_general"] = PromptTemplate(
            template_id="qa_general",
            category=TemplateCategory.QA,
            name="General Q&A",
            system_prompt="""You are a knowledgeable AI assistant.
Answer questions accurately and concisely.
If you don't know something, admit it rather than making up information.""",
            user_prompt_template="{user_message}",
            variables=["user_message"],
            version="1.0"
        )
        
        # Creative template
        self.templates["creative_writing"] = PromptTemplate(
            template_id="creative_writing",
            category=TemplateCategory.CREATIVE,
            name="Creative Writing",
            system_prompt="""You are a creative AI assistant skilled in writing, storytelling, and imagination.
Help users with creative tasks like writing stories, poems, or brainstorming ideas.
Be imaginative and engaging.""",
            user_prompt_template="{user_message}",
            variables=["user_message"],
            version="1.0"
        )
        
        # Technical template
        self.templates["technical_support"] = PromptTemplate(
            template_id="technical_support",
            category=TemplateCategory.TECHNICAL,
            name="Technical Support",
            system_prompt="""You are a technical AI assistant with expertise in programming, software, and technology.
Provide clear, accurate technical explanations.
Include code examples when relevant.
Use proper formatting for code blocks.""",
            user_prompt_template="{user_message}",
            variables=["user_message"],
            version="1.0"
        )
        
        # Personalized template
        self.templates["personalized_general"] = PromptTemplate(
            template_id="personalized_general",
            category=TemplateCategory.PERSONALIZED,
            name="Personalized Assistant",
            system_prompt="""You are a personalized AI assistant.

User Information:
- Name: {user_name}
- Preferred tone: {tone}
- Response length preference: {response_length}
- Language: {language}

{custom_instructions}

Recent context:
{recent_context}

Key facts about the user:
{key_facts}

Adapt your responses based on the user's preferences and conversation history.""",
            user_prompt_template="{user_message}",
            variables=["user_name", "tone", "response_length", "language", "custom_instructions", "recent_context", "key_facts", "user_message"],
            version="1.0"
        )
        
        # Fallback template
        self.templates["fallback"] = PromptTemplate(
            template_id="fallback",
            category=TemplateCategory.FALLBACK,
            name="Fallback",
            system_prompt="""You are a helpful AI assistant.
Respond to the user's message in a helpful and appropriate manner.""",
            user_prompt_template="{user_message}",
            variables=["user_message"],
            version="1.0"
        )
        
        logger.info(f"Loaded {len(self.templates)} default templates")
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a template by ID.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template or None if not found
        """
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: TemplateCategory) -> List[PromptTemplate]:
        """
        Get all templates in a category.
        
        Args:
            category: Template category
            
        Returns:
            List of templates
        """
        return [
            template for template in self.templates.values()
            if template.category == category and template.active
        ]
    
    def add_template(self, template: PromptTemplate):
        """
        Add a new template to the library.
        
        Args:
            template: Template to add
        """
        self.templates[template.template_id] = template
        logger.info(f"Added template: {template.template_id}")
    
    def update_template_performance(self, template_id: str, score: float):
        """
        Update template performance score.
        
        Args:
            template_id: Template identifier
            score: Performance score (0-1)
        """
        if template_id in self.templates:
            template = self.templates[template_id]
            # Running average
            total_score = template.performance_score * template.usage_count + score
            template.usage_count += 1
            template.performance_score = total_score / template.usage_count
            
            logger.debug(f"Updated template {template_id} performance: {template.performance_score:.3f}")
    
    def get_best_template(self, category: TemplateCategory) -> Optional[PromptTemplate]:
        """
        Get the best performing template in a category.
        
        Args:
            category: Template category
            
        Returns:
            Best template or None
        """
        category_templates = self.get_templates_by_category(category)
        
        if not category_templates:
            return None
        
        # Sort by performance score
        category_templates.sort(key=lambda t: t.performance_score, reverse=True)
        return category_templates[0]
    
    def list_all_templates(self) -> List[PromptTemplate]:
        """
        List all templates.
        
        Returns:
            List of all templates
        """
        return list(self.templates.values())


# Global template library instance
template_library = PromptTemplateLibrary()
