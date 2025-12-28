"""
Bulk Feedback Generator
Automatically add feedback to analyzed resumes for ML training
"""

from database import db
from datetime import datetime

def add_feedback_to_analyses(limit=None, strategy='score_based'):
    """
    Add feedback to analyses that don't have feedback yet
    
    Args:
        limit: Maximum number of feedbacks to add (None = all)
        strategy: 'score_based' or 'balanced'
            - score_based: Feedback based on match/score percentage
            - balanced: Try to balance positive/neutral/negative samples
    """
    
    print("=" * 70)
    print("  BULK FEEDBACK GENERATOR")
    print("=" * 70)
    print()
    
    # Get all analyses without feedback
    try:
        # This is a simplified approach - you may need to adjust based on your database structure
        from mongodb_database import MongoDatabase
        mongo_db = MongoDatabase()
        
        # Get analyses that don't have feedback
        analyses = list(mongo_db.db.analysis_results.find({}))
        
        print(f"📊 Found {len(analyses)} total analyses")
        
        # Filter out those that already have feedback
        analyses_without_feedback = []
        for analysis in analyses:
            analysis_id = analysis['_id']
            # Check if feedback exists
            existing_feedback = mongo_db.db.feedback.find_one({'analysis_id': analysis_id})
            if not existing_feedback:
                analyses_without_feedback.append(analysis)
        
        print(f"📝 {len(analyses_without_feedback)} analyses need feedback")
        
        if limit:
            analyses_to_process = analyses_without_feedback[:limit]
            print(f"🎯 Processing {len(analyses_to_process)} analyses (limit: {limit})")
        else:
            analyses_to_process = analyses_without_feedback
            print(f"🎯 Processing all {len(analyses_to_process)} analyses")
        
        print()
        
        # Add feedback
        added_count = 0
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        
        for analysis in analyses_to_process:
            analysis_id = analysis['_id']
            # Analysis data is stored directly in the document, not nested
            analysis_data = analysis
            
            # Get score (either match_percentage or score)
            score = analysis_data.get('match_percentage') or analysis_data.get('score', 0)
            
            # Determine feedback based on strategy
            if strategy == 'score_based':
                # Simple score-based feedback
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
            
            elif strategy == 'balanced':
                # Try to balance the dataset
                total = positive_count + neutral_count + negative_count
                
                if score >= 80:
                    feedback_value = 1
                    feedback_text = "Excellent match"
                    positive_count += 1
                elif score >= 60:
                    # Balance between positive and neutral
                    if positive_count > neutral_count:
                        feedback_value = 0
                        neutral_count += 1
                    else:
                        feedback_value = 1
                        positive_count += 1
                    feedback_text = "Good match"
                elif score >= 40:
                    feedback_value = 0
                    feedback_text = "Average match"
                    neutral_count += 1
                else:
                    feedback_value = -1
                    feedback_text = "Poor match"
                    negative_count += 1
            
            # Save feedback using the proper method
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
                print(f"  ⚠ Error adding feedback for analysis {analysis_id}: {e}")
        
        print()
        print("=" * 70)
        print("  RESULTS")
        print("=" * 70)
        print(f"✅ Successfully added {added_count} feedbacks")
        print(f"   👍 Positive: {positive_count}")
        print(f"   😐 Neutral: {neutral_count}")
        print(f"   👎 Negative: {negative_count}")
        print()
        
        # Check if ready for training
        total_feedback = mongo_db.db.feedback.count_documents({})
        print(f"📊 Total feedback in database: {total_feedback}")
        
        if total_feedback >= 50:
            print(f"✅ Ready for training! You have {total_feedback} labeled samples.")
            print(f"   Run: python train_improved_model.py")
        else:
            print(f"⚠️  Need {50 - total_feedback} more samples to train")
        
        print("=" * 70)
        print()
        
        return added_count
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Main function"""
    print()
    print("This script will add feedback to analyzed resumes.")
    print("This is useful for generating training data for the ML model.")
    print()
    
    # Ask user for preferences
    print("Options:")
    print("1. Add feedback to 50 analyses (minimum for training)")
    print("2. Add feedback to ALL analyses without feedback")
    print("3. Custom number")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    limit = None
    if choice == "1":
        limit = 50
    elif choice == "2":
        limit = None
    elif choice == "3":
        limit_input = input("How many feedbacks to add? ").strip()
        try:
            limit = int(limit_input)
        except ValueError:
            print("Invalid number, using 50")
            limit = 50
    else:
        print("Invalid choice, using 50")
        limit = 50
    
    print()
    print("Strategy:")
    print("1. Score-based (70+ = positive, 50-70 = neutral, <50 = negative)")
    print("2. Balanced (tries to balance positive/neutral/negative)")
    print()
    
    strategy_choice = input("Enter strategy (1-2): ").strip()
    strategy = 'score_based' if strategy_choice == "1" else 'balanced'
    
    print()
    confirm = input(f"Add feedback to {limit or 'all'} analyses using {strategy} strategy? (y/n): ").strip().lower()
    
    if confirm == 'y':
        add_feedback_to_analyses(limit=limit, strategy=strategy)
    else:
        print("\n❌ Cancelled\n")


if __name__ == "__main__":
    main()
