# Learning & Adaptation Module

## Overview

The Learning & Adaptation Module is a comprehensive system for personalizing AI conversations, collecting feedback, and continuously improving model performance. This module implements Member 5's responsibilities for the AI Conversational Assistant project.

## Features

### 1. **Personalization System**
- User preferences management (language, tone, response length)
- Conversation memory with sliding window
- Long-term memory with key fact extraction
- Semantic search for relevant past conversations
- Context-aware prompt generation

### 2. **Model Adaptation**
- RLHF (Reinforcement Learning from Human Feedback) system
- Feedback collection (thumbs up/down, ratings, comments)
- Fine-tuning dataset preparation
- Prompt template library with A/B testing support
- Performance tracking for templates

### 3. **Evaluation Framework**
- Comprehensive metrics (BLEU, ROUGE, BERTScore, F1, etc.)
- Automated evaluation pipeline
- Performance analysis by category
- Comparison between evaluation runs
- Recommendation generation

### 4. **Reporting**
- HTML evaluation reports with visualizations
- JSON export for programmatic access
- Comparison charts
- Summary statistics

## Architecture

```
adaptation/
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── preferences_manager.py # User preferences CRUD
├── memory_module.py       # Conversation memory
├── prompt_templates.py    # Template library
├── prompt_personalizer.py # Prompt customization
├── rlhf_feedback.py       # Feedback collection
├── fine_tuning_prep.py    # Dataset preparation
├── metrics_calculator.py  # Metrics computation
├── evaluator.py           # Evaluation framework
└── report_generator.py    # Report generation
```

## Installation

### Prerequisites
- Python 3.10+
- MongoDB (for data storage)
- Redis (optional, for caching)

### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Download NLTK Data (if needed)

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Configuration

Create a `.env` file in the `backend` directory:

```env
# Database
MONGO_URI=mongodb://localhost:27017/ai_assistant
REDIS_URI=redis://localhost:6379/0

# LLM Provider
OPENAI_API_KEY=your-api-key
LLM_PROVIDER=openai

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Personalization
ADAPTATION_MEMORY_WINDOW_SIZE=8
ADAPTATION_ENABLE_LONG_TERM_MEMORY=true
ADAPTATION_CACHE_TTL_SECONDS=3600

# Evaluation
ADAPTATION_MIN_FEEDBACK_RATING_FOR_TRAINING=4
```

## Usage

### 1. Initialize the Module

```python
from motor.motor_asyncio import AsyncIOMotorClient
from app.services.adaptation import (
    PreferencesManager,
    MemoryModule,
    PromptPersonalizer
)

# Connect to database
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.ai_assistant

# Initialize components
preferences_manager = PreferencesManager(db)
memory_module = MemoryModule(db)
personalizer = PromptPersonalizer(preferences_manager, memory_module)

# Ensure indexes
await preferences_manager.ensure_indexes()
await memory_module.ensure_indexes()
```

### 2. Manage User Preferences

```python
from app.services.adaptation.preferences_manager import UserPreferences

# Get preferences (returns defaults if not found)
prefs = await preferences_manager.get_preferences("user123")

# Update preferences
new_prefs = UserPreferences(
    language="en-IN",
    tone="professional",
    name="Alice",
    response_length="detailed",
    topics_of_interest=["technology", "science"]
)
await preferences_manager.update_preferences("user123", new_prefs)
```

### 3. Build Personalized Prompts

```python
# Build personalized prompt
prompt = await personalizer.build_personalized_prompt(
    user_message="Explain quantum computing",
    session_id="session123",
    user_id="user123"
)

# Use with LLM
system_prompt = prompt["system"]
user_prompt = prompt["user"]
```

### 4. Manage Conversation Memory

```python
# Add conversation turn
await memory_module.add_turn(
    session_id="session123",
    role="user",
    content="What is machine learning?",
    user_id="user123"
)

await memory_module.add_turn(
    session_id="session123",
    role="assistant",
    content="Machine learning is...",
    user_id="user123"
)

# Get recent context
context = await memory_module.get_recent_context("session123")

# Search relevant memories
relevant = await memory_module.search_relevant_memories(
    session_id="session123",
    query="machine learning algorithms"
)
```

### 5. Collect Feedback

```python
from app.services.adaptation.rlhf_feedback import RLHFFeedback, FeedbackType

rlhf = RLHFFeedback(db)

# Submit feedback
await rlhf.submit_feedback(
    session_id="session123",
    feedback_type=FeedbackType.RATING,
    rating=5,
    comments="Very helpful explanation!",
    user_id="user123",
    user_message="What is machine learning?",
    assistant_response="Machine learning is..."
)

# Get statistics
stats = await rlhf.get_feedback_stats()
print(f"Average rating: {stats['average_rating']}")
print(f"Satisfaction rate: {stats['satisfaction_rate']}%")
```

### 6. Run Evaluation

```python
from app.services.adaptation.evaluator import Evaluator
from app.services.adaptation.report_generator import report_generator
from datetime import datetime, timedelta

evaluator = Evaluator(db)

# Run evaluation for last 7 days
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=7)

report = await evaluator.run_evaluation(
    start_date=start_date,
    end_date=end_date,
    max_samples=100
)

# Generate HTML report
html_path = report_generator.generate_html_report(report.model_dump())
print(f"Report saved to: {html_path}")

# View recommendations
for rec in report.recommendations:
    print(f"- {rec}")
```

### 7. Prepare Fine-tuning Dataset

```python
from app.services.adaptation.fine_tuning_prep import FineTuningPrep

fine_tuning = FineTuningPrep(db)

# Prepare dataset
metadata = await fine_tuning.prepare_dataset(
    output_dir="./fine_tuning_data",
    min_rating=4,
    max_conversations=1000,
    train_split=0.8,
    format_type="openai",  # or "huggingface", "custom"
    anonymize=True
)

print(f"Dataset prepared:")
print(f"- Train examples: {metadata['train_examples']}")
print(f"- Val examples: {metadata['val_examples']}")
print(f"- Train file: {metadata['train_file']}")
```

## API Endpoints

### Preferences

```http
GET /api/v1/preferences?user_id=user123
PUT /api/v1/preferences?user_id=user123
DELETE /api/v1/preferences?user_id=user123
```

### Feedback

```http
POST /api/v1/feedback
GET /api/v1/feedback/stats
```

### Admin - Evaluation

```http
POST /api/v1/admin/evaluation/run
GET /api/v1/admin/evaluation/reports
GET /api/v1/admin/evaluation/compare?report_id_1=xxx&report_id_2=yyy
```

### Admin - Fine-tuning

```http
POST /api/v1/admin/finetuning/prepare
```

### Admin - Statistics

```http
GET /api/v1/admin/preferences/stats
```

## Testing

Run unit tests:

```bash
# All tests
pytest app/tests/ -v

# Specific test file
pytest app/tests/test_personalization.py -v
pytest app/tests/test_evaluation.py -v

# With coverage
pytest app/tests/ --cov=app/services/adaptation --cov-report=html
```

## Metrics Explained

### Accuracy Metrics
- **Exact Match**: Binary score for exact string match
- **F1 Score**: Token-level precision and recall
- **BLEU**: N-gram overlap with reference

### Relevancy Metrics
- **Cosine Similarity**: Semantic similarity using embeddings
- **BERTScore**: Contextual embedding similarity
- **ROUGE**: Recall-oriented overlap

### Quality Metrics
- **Toxicity**: Harmful content detection
- **Coherence**: Conversation flow consistency
- **Response Length**: Alignment with user preference

### User Satisfaction
- **User Rating**: Direct user feedback (1-5)
- **Acceptance Rate**: Percentage of positive feedback
- **Completion Rate**: Conversations completed successfully

## Best Practices

### 1. Personalization
- Always check if user has enabled memory before using it
- Cache preferences to reduce database calls
- Respect user's response length preference
- Use appropriate tone based on user preference

### 2. Memory Management
- Limit memory window size to prevent context overflow
- Regularly clean up old sessions (30+ days)
- Extract key facts for long-term memory
- Use semantic search for relevant context retrieval

### 3. Feedback Collection
- Make feedback submission easy (thumbs up/down)
- Collect detailed feedback for low ratings
- Store both user message and assistant response
- Anonymize PII before storing

### 4. Evaluation
- Run evaluations regularly (weekly recommended)
- Compare evaluation reports to track improvements
- Act on recommendations promptly
- Monitor key metrics: user rating, relevance, toxicity

### 5. Fine-tuning
- Use only high-quality conversations (rating ≥ 4)
- Anonymize all PII before fine-tuning
- Maintain proper train/val split (80/20)
- Validate dataset before uploading

## Performance Optimization

### Caching
- Enable preference caching (default: 1 hour TTL)
- Use Redis for distributed caching
- Cache embeddings for frequently accessed content

### Async Operations
- Use async/await for all database operations
- Run memory writes asynchronously
- Batch evaluation for large datasets

### Database Indexes
- Ensure indexes are created on:
  - `user_id` (preferences, sessions)
  - `session_id` (sessions, feedback)
  - `timestamp` (feedback, evaluations)
  - `rating` (feedback)

## Troubleshooting

### High Memory Usage
- Reduce memory window size
- Enable memory cleanup for old sessions
- Use smaller embedding models

### Slow Evaluation
- Reduce `max_samples` parameter
- Disable BERTScore (computationally expensive)
- Run evaluations during off-peak hours

### Low Relevance Scores
- Improve prompt templates
- Increase memory window size
- Use better embedding models
- Collect more context

## Contributing

When adding new features:
1. Update configuration in `config.py`
2. Add comprehensive docstrings
3. Write unit tests
4. Update this README
5. Add API documentation

## License

Part of the AI Conversational Assistant project.

## Support

For issues or questions:
- Check the main project documentation
- Review test files for usage examples
- Contact the team lead (Member 1)
