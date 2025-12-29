"""
Example script demonstrating the Learning & Adaptation Module.

This script shows how to:
1. Set up the module
2. Manage user preferences
3. Build personalized prompts
4. Collect feedback
5. Run evaluations
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta

from app.services.adaptation.preferences_manager import PreferencesManager, UserPreferences
from app.services.adaptation.memory_module import MemoryModule
from app.services.adaptation.prompt_personalizer import PromptPersonalizer
from app.services.adaptation.rlhf_feedback import RLHFFeedback, FeedbackType
from app.services.adaptation.evaluator import Evaluator
from app.services.adaptation.report_generator import report_generator


async def main():
    """Main example function."""
    
    # 1. Connect to database
    print("=" * 60)
    print("Learning & Adaptation Module - Example Usage")
    print("=" * 60)
    
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.ai_assistant_demo
    
    # 2. Initialize components
    print("\n1. Initializing components...")
    preferences_manager = PreferencesManager(db)
    memory_module = MemoryModule(db)
    personalizer = PromptPersonalizer(preferences_manager, memory_module)
    rlhf = RLHFFeedback(db)
    evaluator = Evaluator(db)
    
    # Ensure indexes
    await preferences_manager.ensure_indexes()
    await memory_module.ensure_indexes()
    await rlhf.ensure_indexes()
    
    print("✓ Components initialized")
    
    # 3. Set up user preferences
    print("\n2. Setting up user preferences...")
    user_id = "demo_user_001"
    
    prefs = UserPreferences(
        language="en-IN",
        tone="friendly",
        name="Alex",
        response_length="medium",
        topics_of_interest=["technology", "AI", "programming"],
        custom_instructions="Always provide code examples when discussing programming"
    )
    
    await preferences_manager.update_preferences(user_id, prefs)
    print(f"✓ Preferences set for user: {user_id}")
    print(f"  - Name: {prefs.name}")
    print(f"  - Tone: {prefs.tone}")
    print(f"  - Language: {prefs.language}")
    
    # 4. Simulate a conversation
    print("\n3. Simulating conversation...")
    session_id = "demo_session_001"
    
    # User message 1
    user_msg_1 = "What is machine learning?"
    await memory_module.add_turn(
        session_id=session_id,
        role="user",
        content=user_msg_1,
        user_id=user_id
    )
    
    # Build personalized prompt
    prompt_1 = await personalizer.build_personalized_prompt(
        user_message=user_msg_1,
        session_id=session_id,
        user_id=user_id
    )
    
    print(f"\n  User: {user_msg_1}")
    print(f"  Template used: {prompt_1['template_id']}")
    
    # Simulate assistant response
    assistant_msg_1 = "Machine learning is a subset of AI that enables systems to learn from data..."
    await memory_module.add_turn(
        session_id=session_id,
        role="assistant",
        content=assistant_msg_1,
        user_id=user_id
    )
    
    print(f"  Assistant: {assistant_msg_1[:80]}...")
    
    # User message 2
    user_msg_2 = "Can you give me a Python example?"
    await memory_module.add_turn(
        session_id=session_id,
        role="user",
        content=user_msg_2,
        user_id=user_id
    )
    
    # Build personalized prompt with context
    prompt_2 = await personalizer.build_personalized_prompt(
        user_message=user_msg_2,
        session_id=session_id,
        user_id=user_id
    )
    
    print(f"\n  User: {user_msg_2}")
    print(f"  Template used: {prompt_2['template_id']}")
    print(f"  Context included: ✓ (previous conversation)")
    
    assistant_msg_2 = "Here's a simple Python example of machine learning..."
    await memory_module.add_turn(
        session_id=session_id,
        role="assistant",
        content=assistant_msg_2,
        user_id=user_id
    )
    
    print(f"  Assistant: {assistant_msg_2[:80]}...")
    
    # 5. Collect feedback
    print("\n4. Collecting user feedback...")
    
    feedback_1 = await rlhf.submit_feedback(
        session_id=session_id,
        feedback_type=FeedbackType.RATING,
        rating=5,
        comments="Great explanation with examples!",
        user_id=user_id,
        user_message=user_msg_2,
        assistant_response=assistant_msg_2
    )
    
    print(f"✓ Feedback submitted: {feedback_1.feedback_id}")
    print(f"  - Rating: {feedback_1.rating}/5")
    print(f"  - Comments: {feedback_1.comments}")
    
    # Get feedback stats
    stats = await rlhf.get_feedback_stats()
    print(f"\n  Feedback Statistics:")
    print(f"  - Total feedback: {stats['total_feedback']}")
    print(f"  - Average rating: {stats.get('average_rating', 'N/A')}")
    print(f"  - Satisfaction rate: {stats['satisfaction_rate']}%")
    
    # 6. Check conversation memory
    print("\n5. Checking conversation memory...")
    
    recent_context = await memory_module.get_recent_context(session_id)
    print(f"✓ Recent context: {len(recent_context)} turns")
    
    for i, turn in enumerate(recent_context, 1):
        print(f"  {i}. {turn.role}: {turn.content[:60]}...")
    
    key_facts = await memory_module.get_key_facts(session_id)
    if key_facts:
        print(f"\n  Key facts extracted: {len(key_facts)}")
        for fact in key_facts:
            print(f"  - {fact.fact}")
    
    # 7. Run evaluation (if enough data)
    print("\n6. Running evaluation...")
    
    try:
        report = await evaluator.run_evaluation(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            max_samples=10
        )
        
        print(f"✓ Evaluation completed")
        print(f"  - Report ID: {report.report_id}")
        print(f"  - Total evaluations: {report.total_evaluations}")
        
        if report.aggregate_metrics:
            print(f"\n  Key Metrics:")
            for metric, values in report.aggregate_metrics.items():
                if isinstance(values, dict) and 'mean' in values:
                    print(f"  - {metric}: {values['mean']:.3f}")
        
        print(f"\n  Recommendations:")
        for rec in report.recommendations:
            print(f"  {rec}")
        
        # Generate HTML report
        html_path = report_generator.generate_html_report(report.model_dump())
        print(f"\n  HTML report saved to: {html_path}")
        
    except Exception as e:
        print(f"  Note: Evaluation requires more data. Error: {e}")
    
    # 8. Demonstrate preference retrieval
    print("\n7. Retrieving user preferences...")
    
    retrieved_prefs = await preferences_manager.get_preferences(user_id)
    print(f"✓ Preferences retrieved for {retrieved_prefs.name}")
    print(f"  - Tone: {retrieved_prefs.tone}")
    print(f"  - Response length: {retrieved_prefs.response_length}")
    print(f"  - Topics of interest: {', '.join(retrieved_prefs.topics_of_interest)}")
    
    # 9. Get session summary
    print("\n8. Getting session summary...")
    
    session = await memory_module.get_full_session(session_id)
    if session:
        print(f"✓ Session: {session.session_id}")
        print(f"  - Total turns: {len(session.turns)}")
        print(f"  - Created: {session.created_at}")
        print(f"  - Last updated: {session.updated_at}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Cleanup
    client.close()


if __name__ == "__main__":
    asyncio.run(main())
