"""Detailed debug - check training data format"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mongodb_database import MongoDatabase

def main():
    print("\n" + "="*70)
    print("TRAINING DATA DEBUG")
    print("="*70)
    
    try:
        db = MongoDatabase()
        
        # Get training data
        print("\nFetching training data...")
        raw_data = db.get_training_data(limit=5)
        
        print(f"\nGot {len(raw_data)} samples")
        
        if raw_data:
            print("\nFirst sample structure:")
            first_sample = raw_data[0]
            print(f"  Type: {type(first_sample)}")
            print(f"  Length: {len(first_sample)}")
            print(f"\n  Fields:")
            for i, field in enumerate(first_sample):
                print(f"    [{i}] {type(field).__name__}: {str(field)[:100]}")
            
            print("\nChecking skills fields specifically:")
            resume_text, jd_text, match_pct, score, quality_score, \
                skills_found, missing_skills, feedback_value = first_sample
            
            print(f"  skills_found type: {type(skills_found)}")
            print(f"  skills_found value: {skills_found}")
            print(f"  missing_skills type: {type(missing_skills)}")
            print(f"  missing_skills value: {missing_skills}")
            print(f"  feedback_value type: {type(feedback_value)}")
            print(f"  feedback_value value: {feedback_value}")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
