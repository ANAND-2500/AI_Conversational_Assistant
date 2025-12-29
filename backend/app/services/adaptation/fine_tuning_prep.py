"""
Fine-tuning Data Preparation - prepares conversation data for model fine-tuning.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel
import json
import logging
from pathlib import Path

from .config import config
from .utils import data_anonymizer, text_preprocessor

logger = logging.getLogger(__name__)


class FineTuningExample(BaseModel):
    """Single fine-tuning example."""
    
    messages: List[Dict[str, str]]
    metadata: Optional[Dict[str, Any]] = None


class FineTuningPrep:
    """Prepares datasets for model fine-tuning."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize fine-tuning preparation module.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.sessions_collection = db.conversation_sessions
        self.feedback_collection = db.feedback
    
    async def prepare_dataset(
        self,
        output_dir: str,
        min_rating: int = 4,
        max_conversations: int = 1000,
        train_split: float = 0.8,
        format_type: str = "openai",
        anonymize: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare fine-tuning dataset from conversation logs.
        
        Args:
            output_dir: Directory to save dataset files
            min_rating: Minimum feedback rating to include
            max_conversations: Maximum number of conversations
            train_split: Train/validation split ratio
            format_type: Output format (openai, huggingface, custom)
            anonymize: Whether to anonymize PII
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Starting dataset preparation...")
        
        # Get high-quality conversations based on feedback
        high_quality_feedback = await self.feedback_collection.find({
            "rating": {"$gte": min_rating},
            "user_message": {"$ne": None},
            "assistant_response": {"$ne": None}
        }).limit(max_conversations).to_list(length=max_conversations)
        
        logger.info(f"Found {len(high_quality_feedback)} high-quality conversations")
        
        # Convert to training examples
        examples = []
        
        for feedback in high_quality_feedback:
            # Get full session for context
            session = await self.sessions_collection.find_one({
                "session_id": feedback["session_id"]
            })
            
            if not session:
                continue
            
            # Build training example from conversation
            example = self._build_training_example(
                session,
                feedback,
                anonymize=anonymize
            )
            
            if example:
                examples.append(example)
        
        logger.info(f"Created {len(examples)} training examples")
        
        # Shuffle and split
        import random
        random.shuffle(examples)
        
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_file = output_path / "train.jsonl"
        val_file = output_path / "val.jsonl"
        
        if format_type == "openai":
            self._save_openai_format(train_examples, train_file)
            self._save_openai_format(val_examples, val_file)
        elif format_type == "huggingface":
            self._save_huggingface_format(train_examples, train_file)
            self._save_huggingface_format(val_examples, val_file)
        else:
            self._save_custom_format(train_examples, train_file)
            self._save_custom_format(val_examples, val_file)
        
        # Generate metadata
        metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "total_examples": len(examples),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "min_rating": min_rating,
            "format": format_type,
            "anonymized": anonymize,
            "train_file": str(train_file),
            "val_file": str(val_file)
        }
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}")
        logger.info(f"Train: {len(train_examples)}, Val: {len(val_examples)}")
        
        return metadata
    
    def _build_training_example(
        self,
        session: Dict[str, Any],
        feedback: Dict[str, Any],
        anonymize: bool = True
    ) -> Optional[FineTuningExample]:
        """
        Build a training example from session and feedback.
        
        Args:
            session: Session document
            feedback: Feedback document
            anonymize: Whether to anonymize
            
        Returns:
            Training example or None
        """
        turns = session.get("turns", [])
        
        if not turns:
            return None
        
        # Build message sequence
        messages = []
        
        # Add system message (simplified)
        messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant."
        })
        
        # Add conversation turns
        for turn in turns:
            content = turn["content"]
            
            # Anonymize if requested
            if anonymize:
                content = text_preprocessor.anonymize_pii(content)
            
            # Sanitize
            content = text_preprocessor.sanitize_text(content)
            
            messages.append({
                "role": turn["role"],
                "content": content
            })
        
        # Create example
        example = FineTuningExample(
            messages=messages,
            metadata={
                "session_id": session["session_id"],
                "rating": feedback.get("rating"),
                "timestamp": session.get("created_at")
            }
        )
        
        return example
    
    def _save_openai_format(self, examples: List[FineTuningExample], filepath: Path):
        """
        Save in OpenAI fine-tuning format (JSONL).
        
        Args:
            examples: List of examples
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            for example in examples:
                # OpenAI format: {"messages": [...]}
                line = json.dumps({"messages": example.messages})
                f.write(line + '\n')
        
        logger.info(f"Saved {len(examples)} examples in OpenAI format to {filepath}")
    
    def _save_huggingface_format(self, examples: List[FineTuningExample], filepath: Path):
        """
        Save in Hugging Face format (JSONL).
        
        Args:
            examples: List of examples
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            for example in examples:
                # Convert to text format for Hugging Face
                text_parts = []
                for msg in example.messages:
                    role = msg["role"].capitalize()
                    text_parts.append(f"{role}: {msg['content']}")
                
                text = "\n".join(text_parts)
                
                line = json.dumps({"text": text})
                f.write(line + '\n')
        
        logger.info(f"Saved {len(examples)} examples in Hugging Face format to {filepath}")
    
    def _save_custom_format(self, examples: List[FineTuningExample], filepath: Path):
        """
        Save in custom format with metadata.
        
        Args:
            examples: List of examples
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            for example in examples:
                line = json.dumps(example.model_dump())
                f.write(line + '\n')
        
        logger.info(f"Saved {len(examples)} examples in custom format to {filepath}")
    
    async def augment_data(
        self,
        examples: List[FineTuningExample],
        augmentation_factor: int = 2
    ) -> List[FineTuningExample]:
        """
        Apply data augmentation techniques.
        
        Args:
            examples: Original examples
            augmentation_factor: How many augmented versions per example
            
        Returns:
            Augmented examples
        """
        augmented = list(examples)  # Start with originals
        
        for example in examples:
            for _ in range(augmentation_factor - 1):
                # Simple augmentation: paraphrase user messages
                # In production, use more sophisticated techniques
                aug_example = self._augment_example(example)
                if aug_example:
                    augmented.append(aug_example)
        
        logger.info(f"Augmented dataset from {len(examples)} to {len(augmented)} examples")
        
        return augmented
    
    def _augment_example(self, example: FineTuningExample) -> Optional[FineTuningExample]:
        """
        Augment a single example.
        
        Args:
            example: Original example
            
        Returns:
            Augmented example or None
        """
        # Simple augmentation: add variation to system prompt
        # In production, use paraphrasing models
        
        augmented_messages = []
        
        for msg in example.messages:
            if msg["role"] == "system":
                # Vary system prompt slightly
                variations = [
                    "You are a helpful and friendly AI assistant.",
                    "You are an AI assistant designed to help users.",
                    "You are a knowledgeable AI assistant."
                ]
                import random
                augmented_messages.append({
                    "role": "system",
                    "content": random.choice(variations)
                })
            else:
                augmented_messages.append(msg)
        
        return FineTuningExample(
            messages=augmented_messages,
            metadata=example.metadata
        )
    
    async def validate_dataset(self, filepath: Path) -> Dict[str, Any]:
        """
        Validate a prepared dataset.
        
        Args:
            filepath: Path to dataset file
            
        Returns:
            Validation results
        """
        issues = []
        total_examples = 0
        total_tokens = 0
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    total_examples += 1
                    
                    # Check format
                    if "messages" not in data:
                        issues.append(f"Line {line_num}: Missing 'messages' field")
                        continue
                    
                    # Estimate tokens (rough approximation)
                    for msg in data["messages"]:
                        total_tokens += len(msg.get("content", "").split())
                    
                except json.JSONDecodeError:
                    issues.append(f"Line {line_num}: Invalid JSON")
        
        avg_tokens = total_tokens / total_examples if total_examples > 0 else 0
        
        return {
            "valid": len(issues) == 0,
            "total_examples": total_examples,
            "total_tokens": total_tokens,
            "avg_tokens_per_example": round(avg_tokens, 2),
            "issues": issues
        }
