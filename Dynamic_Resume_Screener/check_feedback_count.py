"""Quick script to check feedback count"""

from mongodb_database import MongoDatabase

try:
    db = MongoDatabase()
    
    # Count feedback
    feedback_count = db.db.feedback.count_documents({})
    analyses_count = db.db.analysis_results.count_documents({})
    
    print(f"\n{'='*50}")
    print(f"  FEEDBACK STATUS")
    print(f"{'='*50}")
    print(f"Total Analyses: {analyses_count}")
    print(f"Total Feedback: {feedback_count}")
    print(f"{'='*50}")
    
    if feedback_count >= 50:
        print(f"✅ READY TO TRAIN! ({feedback_count} samples)")
        print(f"\nRun: python train_improved_model.py")
    else:
        print(f"❌ Need {50 - feedback_count} more samples")
        print(f"\nRun: python add_feedback_quick.py")
    
    print(f"{'='*50}\n")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
