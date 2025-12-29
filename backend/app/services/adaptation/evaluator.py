"""
Evaluator - main evaluation framework for model performance assessment.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel
import logging

from .metrics_calculator import metrics_calculator
from .config import config

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Single evaluation result."""
    
    session_id: str
    user_message: str
    assistant_response: str
    reference_response: Optional[str] = None
    metrics: Dict[str, float]
    timestamp: datetime = datetime.utcnow()


class EvaluationReport(BaseModel):
    """Complete evaluation report."""
    
    report_id: str
    created_at: datetime
    evaluation_period: Dict[str, datetime]
    total_evaluations: int
    aggregate_metrics: Dict[str, Any]
    performance_by_category: Dict[str, Dict[str, float]]
    recommendations: List[str]
    metadata: Dict[str, Any]


class Evaluator:
    """Main evaluation framework."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize evaluator.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.sessions_collection = db.conversation_sessions
        self.feedback_collection = db.feedback
        self.evaluation_collection = db.evaluations
    
    async def run_evaluation(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_samples: int = 100
    ) -> EvaluationReport:
        """
        Run comprehensive evaluation.
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Evaluation report
        """
        import uuid
        
        logger.info("Starting evaluation run...")
        
        # Default to last 7 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        # Get conversations with feedback in the date range
        feedback_docs = await self.feedback_collection.find({
            "timestamp": {"$gte": start_date, "$lte": end_date},
            "rating": {"$ne": None},
            "user_message": {"$ne": None},
            "assistant_response": {"$ne": None}
        }).limit(max_samples).to_list(length=max_samples)
        
        logger.info(f"Evaluating {len(feedback_docs)} conversations")
        
        # Evaluate each conversation
        evaluation_results = []
        
        for feedback in feedback_docs:
            result = await self._evaluate_conversation(feedback)
            if result:
                evaluation_results.append(result)
        
        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(evaluation_results)
        
        # Analyze by category
        performance_by_category = self._analyze_by_category(evaluation_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(aggregate_metrics)
        
        # Create report
        report = EvaluationReport(
            report_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            evaluation_period={
                "start": start_date,
                "end": end_date
            },
            total_evaluations=len(evaluation_results),
            aggregate_metrics=aggregate_metrics,
            performance_by_category=performance_by_category,
            recommendations=recommendations,
            metadata={
                "max_samples": max_samples,
                "actual_samples": len(evaluation_results)
            }
        )
        
        # Save report
        await self.evaluation_collection.insert_one(report.model_dump())
        
        logger.info(f"Evaluation complete. Report ID: {report.report_id}")
        
        return report
    
    async def _evaluate_conversation(
        self,
        feedback: Dict[str, Any]
    ) -> Optional[EvaluationResult]:
        """
        Evaluate a single conversation.
        
        Args:
            feedback: Feedback document
            
        Returns:
            Evaluation result or None
        """
        try:
            user_message = feedback.get("user_message", "")
            assistant_response = feedback.get("assistant_response", "")
            
            if not user_message or not assistant_response:
                return None
            
            # Get conversation context
            session = await self.sessions_collection.find_one({
                "session_id": feedback["session_id"]
            })
            
            conversation_context = []
            if session and session.get("turns"):
                conversation_context = [
                    turn["content"] for turn in session["turns"][-5:]
                ]
            
            # Calculate metrics
            # Note: We don't have a reference response, so we use user satisfaction as proxy
            metrics = {
                "user_rating": feedback.get("rating", 0),
                "response_length": len(assistant_response.split()),
                "toxicity": metrics_calculator.calculate_toxicity_score(assistant_response)
            }
            
            # Add coherence if we have context
            if len(conversation_context) > 1:
                metrics["coherence"] = metrics_calculator.calculate_coherence_score(
                    conversation_context
                )
            
            # Calculate semantic relevance to user query
            metrics["relevance"] = metrics_calculator.calculate_cosine_similarity(
                user_message,
                assistant_response
            )
            
            result = EvaluationResult(
                session_id=feedback["session_id"],
                user_message=user_message,
                assistant_response=assistant_response,
                metrics=metrics,
                timestamp=feedback.get("timestamp", datetime.utcnow())
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating conversation: {e}")
            return None
    
    def _aggregate_metrics(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across all results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Aggregated metrics
        """
        if not results:
            return {}
        
        # Collect all metric values
        metric_values = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)
        
        # Calculate statistics for each metric
        aggregated = {}
        for metric_name, values in metric_values.items():
            import numpy as np
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        
        return aggregated
    
    def _analyze_by_category(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by category.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Performance metrics by category
        """
        # Categorize by response length
        short_responses = []
        medium_responses = []
        long_responses = []
        
        for result in results:
            length = result.metrics.get("response_length", 0)
            if length < 50:
                short_responses.append(result)
            elif length < 200:
                medium_responses.append(result)
            else:
                long_responses.append(result)
        
        categories = {
            "short_responses": short_responses,
            "medium_responses": medium_responses,
            "long_responses": long_responses
        }
        
        performance = {}
        for category_name, category_results in categories.items():
            if category_results:
                ratings = [r.metrics.get("user_rating", 0) for r in category_results]
                import numpy as np
                performance[category_name] = {
                    "count": len(category_results),
                    "avg_rating": float(np.mean(ratings)),
                    "avg_relevance": float(np.mean([
                        r.metrics.get("relevance", 0) for r in category_results
                    ]))
                }
        
        return performance
    
    def _generate_recommendations(
        self,
        aggregate_metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on metrics.
        
        Args:
            aggregate_metrics: Aggregated metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check user rating
        if "user_rating" in aggregate_metrics:
            avg_rating = aggregate_metrics["user_rating"]["mean"]
            if avg_rating < 3.5:
                recommendations.append(
                    "⚠️ Average user rating is below 3.5. Consider reviewing low-rated conversations for improvement opportunities."
                )
            elif avg_rating >= 4.5:
                recommendations.append(
                    "✅ Excellent user satisfaction! Current approach is working well."
                )
        
        # Check relevance
        if "relevance" in aggregate_metrics:
            avg_relevance = aggregate_metrics["relevance"]["mean"]
            if avg_relevance < 0.6:
                recommendations.append(
                    "⚠️ Response relevance is low. Consider improving prompt templates or context retrieval."
                )
        
        # Check toxicity
        if "toxicity" in aggregate_metrics:
            avg_toxicity = aggregate_metrics["toxicity"]["mean"]
            if avg_toxicity > 0.2:
                recommendations.append(
                    "⚠️ Elevated toxicity detected. Review and strengthen content filtering."
                )
        
        # Check coherence
        if "coherence" in aggregate_metrics:
            avg_coherence = aggregate_metrics["coherence"]["mean"]
            if avg_coherence < 0.5:
                recommendations.append(
                    "⚠️ Low conversation coherence. Improve memory module or context window size."
                )
        
        if not recommendations:
            recommendations.append(
                "✅ All metrics are within acceptable ranges. Continue monitoring."
            )
        
        return recommendations
    
    async def get_evaluation_reports(
        self,
        limit: int = 10
    ) -> List[EvaluationReport]:
        """
        Get recent evaluation reports.
        
        Args:
            limit: Maximum number of reports
            
        Returns:
            List of evaluation reports
        """
        cursor = self.evaluation_collection.find().sort(
            "created_at", -1
        ).limit(limit)
        
        docs = await cursor.to_list(length=limit)
        return [EvaluationReport(**doc) for doc in docs]
    
    async def compare_evaluations(
        self,
        report_id_1: str,
        report_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two evaluation reports.
        
        Args:
            report_id_1: First report ID
            report_id_2: Second report ID
            
        Returns:
            Comparison results
        """
        report1 = await self.evaluation_collection.find_one({"report_id": report_id_1})
        report2 = await self.evaluation_collection.find_one({"report_id": report_id_2})
        
        if not report1 or not report2:
            return {"error": "One or both reports not found"}
        
        # Compare key metrics
        comparison = {
            "report_1": {
                "id": report_id_1,
                "date": report1["created_at"],
                "total_evaluations": report1["total_evaluations"]
            },
            "report_2": {
                "id": report_id_2,
                "date": report2["created_at"],
                "total_evaluations": report2["total_evaluations"]
            },
            "metric_changes": {}
        }
        
        # Calculate changes
        metrics1 = report1.get("aggregate_metrics", {})
        metrics2 = report2.get("aggregate_metrics", {})
        
        for metric_name in metrics1.keys():
            if metric_name in metrics2:
                val1 = metrics1[metric_name].get("mean", 0)
                val2 = metrics2[metric_name].get("mean", 0)
                change = val2 - val1
                percent_change = (change / val1 * 100) if val1 != 0 else 0
                
                comparison["metric_changes"][metric_name] = {
                    "before": val1,
                    "after": val2,
                    "change": change,
                    "percent_change": round(percent_change, 2)
                }
        
        return comparison
    
    async def ensure_indexes(self):
        """Create necessary database indexes."""
        await self.evaluation_collection.create_index("report_id", unique=True)
        await self.evaluation_collection.create_index("created_at")
        logger.info("Evaluation indexes created")
