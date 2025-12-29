"""
Utility functions for the Learning & Adaptation Module.
"""
import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 5000) -> str:
        """
        Sanitize and normalize text input.
        
        Args:
            text: Input text
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Truncate to max length
        text = text[:max_length]
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def anonymize_pii(text: str) -> str:
        """
        Basic PII anonymization.
        
        Args:
            text: Input text
            
        Returns:
            Anonymized text
        """
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers (basic patterns)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b', '[PHONE]', text)
        
        # Credit card numbers (basic pattern)
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 5) -> List[str]:
        """
        Extract keywords from text (simple frequency-based).
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Simple word frequency approach
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        words = [w for w in words if w not in stop_words]
        
        # Count frequencies
        from collections import Counter
        word_counts = Counter(words)
        
        return [word for word, _ in word_counts.most_common(top_k)]


class EmbeddingGenerator:
    """Generate embeddings for text."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embedding vectors
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        for idx, candidate in enumerate(candidate_embeddings):
            sim = self.cosine_similarity(query_embedding, candidate)
            similarities.append((idx, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class DataAnonymizer:
    """Anonymize sensitive data."""
    
    @staticmethod
    def hash_user_id(user_id: str, salt: str = "default_salt") -> str:
        """
        Hash user ID for anonymization.
        
        Args:
            user_id: Original user ID
            salt: Salt for hashing
            
        Returns:
            Hashed user ID
        """
        combined = f"{user_id}{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    @staticmethod
    def anonymize_conversation(conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize a conversation object.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Anonymized conversation
        """
        anonymized = conversation.copy()
        
        # Remove or hash sensitive fields
        if 'user_id' in anonymized:
            anonymized['user_id'] = DataAnonymizer.hash_user_id(anonymized['user_id'])
        
        if 'ip_address' in anonymized:
            del anonymized['ip_address']
        
        # Anonymize message content
        if 'messages' in anonymized:
            for msg in anonymized['messages']:
                if 'content' in msg:
                    msg['content'] = TextPreprocessor.anonymize_pii(msg['content'])
        
        return anonymized


class MetricsHelper:
    """Helper functions for metrics calculation."""
    
    @staticmethod
    def calculate_percentile(values: List[float], percentile: int) -> float:
        """
        Calculate percentile of values.
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        return float(np.percentile(values, percentile))
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics.
        
        Args:
            values: List of values
            
        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }
        
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(values)
        }
    
    @staticmethod
    def format_timestamp(dt: Optional[datetime] = None) -> str:
        """
        Format timestamp for logging.
        
        Args:
            dt: Datetime object (uses current time if None)
            
        Returns:
            Formatted timestamp string
        """
        if dt is None:
            dt = datetime.utcnow()
        return dt.isoformat() + "Z"


# Global instances
text_preprocessor = TextPreprocessor()
embedding_generator = EmbeddingGenerator()
data_anonymizer = DataAnonymizer()
metrics_helper = MetricsHelper()
