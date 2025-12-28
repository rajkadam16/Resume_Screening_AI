"""Delete all feedback and regenerate with balanced distribution"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mongodb_database import MongoDatabase

def main():
    print("\n" + "="*70)
    print("REGENERATE BALANCED FEEDBACK")
    print("="*70)
    
    try:
        db = MongoDatabase()
        
        # Delete all existing feedback
        print("\n1. Deleting existing feedback...")
        result = db.db.feedback.delete_many({})
        print(f"   Deleted {result.deleted_count} feedback records")
        
        # Get all analyses
        print("\n2. Getting analyses...")
        analyses = list(db.db.analysis_results.find({}))
        print(f"   Found {len(analyses)} analyses")
        
        # Generate balanced feedback
        print("\n3. Generating balanced feedback...")
        
        # Sort by score to ensure good distribution
        analyses_with_scores = []
        for analysis in analyses:
            score = analysis.get('match_percentage') or analysis.get('score', 0)
            analyses_with_scores.append((analysis, score))
        
        # Sort by score
        analyses_with_scores.sort(key=lambda x: x[1])
        
        # Generate feedback with balanced distribution
        # Bottom 30% -> negative (-1)
        # Middle 40% -> neutral (0)
        # Top 30% -> positive (1)
        
        total = len(analyses_with_scores)
        negative_count = int(total * 0.3)
        neutral_count = int(total * 0.4)
        
        added_count = 0
        positive_count = 0
        neutral_count_actual = 0
        negative_count_actual = 0
        
        for i, (analysis, score) in enumerate(analyses_with_scores):
            analysis_id = str(analysis['_id'])
            
            if i < negative_count:
                # Bottom 30% - negative
                feedback_value = -1
                feedback_text = f"Poor match (score: {score:.1f})"
                negative_count_actual += 1
            elif i < negative_count + neutral_count:
                # Middle 40% - neutral
                feedback_value = 0
                feedback_text = f"Average match (score: {score:.1f})"
                neutral_count_actual += 1
            else:
                # Top 30% - positive
                feedback_value = 1
                feedback_text = f"Good match (score: {score:.1f})"
                positive_count += 1
            
            # Save feedback
            db.save_feedback(
                analysis_id=analysis_id,
                feedback_type='balanced_generated',
                feedback_value=feedback_value,
                comments=feedback_text
            )
            added_count += 1
            
            if added_count % 50 == 0:
                print(f"   Added {added_count} feedbacks...")
        
        print(f"\n4. Results:")
        print(f"   Total feedback added: {added_count}")
        print(f"   👍 Positive (1):  {positive_count}")
        print(f"   😐 Neutral (0):   {neutral_count_actual}")
        print(f"   👎 Negative (-1): {negative_count_actual}")
        
        # Verify
        total_feedback = db.db.feedback.count_documents({})
        print(f"\n5. Verification:")
        print(f"   Total feedback in database: {total_feedback}")
        
        if total_feedback >= 50:
            print(f"\n✅ Ready for training!")
            print(f"   Run: python train_improved_model.py")
        
        print("\n" + "="*70)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
