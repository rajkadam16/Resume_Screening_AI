"""
Quick Feedback Generator - Non-interactive version
Automatically adds feedback to 50 analyses for ML training
"""

from mongodb_database import MongoDatabase
from datetime import datetime

def add_feedback_quick():
    """Add feedback to 50 analyses quickly"""
    
    print("\n" + "=" * 70)
    print("  QUICK FEEDBACK GENERATOR")
    print("=" * 70 + "\n")
    
    try:
        mongo_db = MongoDatabase()
        
        # Get analyses without feedback
        analyses = list(mongo_db.db.analysis_results.find({}))
        print(f"📊 Found {len(analyses)} total analyses")
        
        # Filter out those that already have feedback
        analyses_without_feedback = []
        for analysis in analyses:
            analysis_id = analysis['_id']
            existing_feedback = mongo_db.db.feedback.find_one({'analysis_id': analysis_id})
            if not existing_feedback:
                analyses_without_feedback.append(analysis)
        
        print(f"📝 {len(analyses_without_feedback)} analyses need feedback\n")
        
        # Process up to 50
        to_process = min(50, len(analyses_without_feedback))
        print(f"🎯 Adding feedback to {to_process} analyses...\n")
        
        added_count = 0
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        
        for analysis in analyses_without_feedback[:to_process]:
            analysis_id = analysis['_id']
            
            # Get score
            score = analysis.get('match_percentage') or analysis.get('score', 0)
            
            # Determine feedback based on score
            if score >= 70:
                feedback_value = 1  # Positive
                feedback_text = "Good match based on score"
                positive_count += 1
            elif score >= 50:
                feedback_value = 0  # Neutral
                feedback_text = "Average match"
                neutral_count += 1
            else:
                feedback_value = -1  # Negative
                feedback_text = "Poor match based on score"
                negative_count += 1
            
            # Save feedback
            try:
                mongo_db.save_feedback(
                    analysis_id=str(analysis_id),
                    feedback_type='bulk_generated',
                    feedback_value=feedback_value,
                    comments=feedback_text
                )
                added_count += 1
                
                if added_count % 10 == 0:
                    print(f"  ✓ Added {added_count} feedbacks...")
            
            except Exception as e:
                print(f"  ⚠ Error adding feedback: {e}")
        
        print()
        print("=" * 70)
        print("  RESULTS")
        print("=" * 70)
        print(f"✅ Successfully added {added_count} feedbacks")
        print(f"   👍 Positive: {positive_count}")
        print(f"   😐 Neutral: {neutral_count}")
        print(f"   👎 Negative: {negative_count}")
        print()
        
        # Check total feedback
        total_feedback = mongo_db.db.feedback.count_documents({})
        print(f"📊 Total feedback in database: {total_feedback}")
        
        if total_feedback >= 50:
            print(f"✅ Ready for training! You have {total_feedback} labeled samples.")
            print(f"   Run: python train_improved_model.py")
        else:
            print(f"⚠️  Need {50 - total_feedback} more samples to train")
        
        print("=" * 70 + "\n")
        
        return added_count
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    add_feedback_quick()
