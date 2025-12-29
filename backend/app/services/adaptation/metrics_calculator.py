"""
Metrics Calculator - calculates various evaluation metrics for model responses.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates evaluation metrics for model responses."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self._bert_scorer = None
        self._rouge_scorer = None
    
    def calculate_bleu_score(
        self,
        reference: str,
        candidate: str,
        max_n: int = 4
    ) -> float:
        """
        Calculate BLEU score.
        
        Args:
            reference: Reference text
            candidate: Candidate text
            max_n: Maximum n-gram size
            
        Returns:
            BLEU score (0-1)
        """
        try:
            from sacrebleu import sentence_bleu
            
            score = sentence_bleu(candidate, [reference])
            return score.score / 100.0  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_rouge_scores(
        self,
        reference: str,
        candidate: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            reference: Reference text
            candidate: Candidate text
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        try:
            from rouge_score import rouge_scorer
            
            if self._rouge_scorer is None:
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=True
                )
            
            scores = self._rouge_scorer.score(reference, candidate)
            
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_bert_score(
        self,
        references: List[str],
        candidates: List[str],
        model_type: str = "microsoft/deberta-xlarge-mnli"
    ) -> Dict[str, float]:
        """
        Calculate BERTScore.
        
        Args:
            references: List of reference texts
            candidates: List of candidate texts
            model_type: BERT model to use
            
        Returns:
            Dictionary with precision, recall, F1
        """
        try:
            from bert_score import score
            
            P, R, F1 = score(
                candidates,
                references,
                model_type=model_type,
                verbose=False
            )
            
            return {
                "precision": float(P.mean()),
                "recall": float(R.mean()),
                "f1": float(F1.mean())
            }
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def calculate_cosine_similarity(
        self,
        text1: str,
        text2: str,
        embedding_generator=None
    ) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            embedding_generator: Optional embedding generator
            
        Returns:
            Similarity score (0-1)
        """
        try:
            if embedding_generator is None:
                from .utils import embedding_generator as default_gen
                embedding_generator = default_gen
            
            emb1 = embedding_generator.generate_embedding(text1)
            emb2 = embedding_generator.generate_embedding(text2)
            
            return embedding_generator.cosine_similarity(emb1, emb2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def calculate_exact_match(
        self,
        reference: str,
        candidate: str,
        case_sensitive: bool = False
    ) -> float:
        """
        Calculate exact match score.
        
        Args:
            reference: Reference text
            candidate: Candidate text
            case_sensitive: Whether to be case sensitive
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if not case_sensitive:
            reference = reference.lower().strip()
            candidate = candidate.lower().strip()
        else:
            reference = reference.strip()
            candidate = candidate.strip()
        
        return 1.0 if reference == candidate else 0.0
    
    def calculate_f1_score(
        self,
        reference: str,
        candidate: str
    ) -> float:
        """
        Calculate token-level F1 score.
        
        Args:
            reference: Reference text
            candidate: Candidate text
            
        Returns:
            F1 score (0-1)
        """
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())
        
        if len(cand_tokens) == 0:
            return 0.0
        
        common = ref_tokens & cand_tokens
        
        if len(common) == 0:
            return 0.0
        
        precision = len(common) / len(cand_tokens)
        recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_response_length_score(
        self,
        response: str,
        preferred_length: str = "medium"
    ) -> float:
        """
        Score response based on length preference.
        
        Args:
            response: Response text
            preferred_length: Preferred length (short/medium/detailed)
            
        Returns:
            Score (0-1)
        """
        word_count = len(response.split())
        
        # Define ideal ranges
        ranges = {
            "short": (10, 50),
            "medium": (50, 200),
            "detailed": (200, 500)
        }
        
        min_words, max_words = ranges.get(preferred_length, (50, 200))
        
        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words:
            return word_count / min_words
        else:
            # Penalize for being too long
            excess = word_count - max_words
            penalty = min(excess / max_words, 0.5)
            return 1.0 - penalty
    
    def calculate_diversity_score(self, responses: List[str]) -> float:
        """
        Calculate diversity score for a set of responses.
        
        Args:
            responses: List of responses
            
        Returns:
            Diversity score (0-1)
        """
        if not responses:
            return 0.0
        
        # Calculate unique n-grams
        all_bigrams = []
        for response in responses:
            words = response.lower().split()
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            all_bigrams.extend(bigrams)
        
        if not all_bigrams:
            return 0.0
        
        unique_ratio = len(set(all_bigrams)) / len(all_bigrams)
        return unique_ratio
    
    def calculate_toxicity_score(self, text: str) -> float:
        """
        Simple toxicity detection (placeholder).
        
        Args:
            text: Text to check
            
        Returns:
            Toxicity score (0-1, higher is more toxic)
        """
        # Simple keyword-based approach
        # In production, use a proper toxicity classifier
        
        toxic_keywords = [
            "hate", "stupid", "idiot", "dumb", "kill",
            "offensive", "inappropriate"
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in text_lower)
        
        # Normalize
        return min(toxic_count / 3.0, 1.0)
    
    def calculate_coherence_score(
        self,
        conversation_turns: List[str]
    ) -> float:
        """
        Calculate conversation coherence score.
        
        Args:
            conversation_turns: List of conversation turns
            
        Returns:
            Coherence score (0-1)
        """
        if len(conversation_turns) < 2:
            return 1.0
        
        # Calculate average similarity between consecutive turns
        from .utils import embedding_generator
        
        similarities = []
        for i in range(len(conversation_turns) - 1):
            sim = self.calculate_cosine_similarity(
                conversation_turns[i],
                conversation_turns[i + 1],
                embedding_generator
            )
            similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def calculate_all_metrics(
        self,
        reference: str,
        candidate: str,
        conversation_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            reference: Reference text
            candidate: Candidate text
            conversation_context: Optional conversation context
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Accuracy metrics
        metrics["exact_match"] = self.calculate_exact_match(reference, candidate)
        metrics["f1_score"] = self.calculate_f1_score(reference, candidate)
        metrics["bleu_score"] = self.calculate_bleu_score(reference, candidate)
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_scores(reference, candidate)
        metrics.update(rouge_scores)
        
        # Semantic similarity
        metrics["cosine_similarity"] = self.calculate_cosine_similarity(reference, candidate)
        
        # Quality metrics
        metrics["toxicity"] = self.calculate_toxicity_score(candidate)
        metrics["response_length"] = len(candidate.split())
        
        # Coherence if context provided
        if conversation_context:
            metrics["coherence"] = self.calculate_coherence_score(
                conversation_context + [candidate]
            )
        
        return metrics


# Global instance
metrics_calculator = MetricsCalculator()
