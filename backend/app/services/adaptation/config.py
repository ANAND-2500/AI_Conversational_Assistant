"""
Configuration module for Learning & Adaptation system.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class AdaptationConfig(BaseSettings):
    """Configuration for the Learning & Adaptation Module."""
    
    # Memory Settings
    memory_window_size: int = 8
    enable_long_term_memory: bool = True
    max_memory_tokens: int = 2000
    memory_decay_factor: float = 0.95
    
    # Embedding Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7
    
    # Personalization Settings
    default_tone: str = "friendly"
    default_response_length: str = "medium"
    default_language: str = "en"
    
    # Prompt Template Settings
    template_version: str = "v1.0"
    enable_ab_testing: bool = False
    
    # RLHF Settings
    min_feedback_rating_for_training: int = 4
    feedback_aggregation_window_days: int = 7
    
    # Fine-tuning Settings
    train_val_split_ratio: float = 0.8
    min_conversations_for_finetuning: int = 100
    data_augmentation_enabled: bool = True
    
    # Evaluation Settings
    evaluation_batch_size: int = 100
    enable_bleu_score: bool = True
    enable_rouge_score: bool = True
    enable_bert_score: bool = True
    bert_score_model: str = "microsoft/deberta-xlarge-mnli"
    
    # Cache Settings
    cache_ttl_seconds: int = 3600
    enable_preference_cache: bool = True
    
    # Performance Settings
    max_concurrent_evaluations: int = 5
    async_memory_writes: bool = True
    
    class Config:
        env_prefix = "ADAPTATION_"
        case_sensitive = False


# Global config instance
config = AdaptationConfig()
