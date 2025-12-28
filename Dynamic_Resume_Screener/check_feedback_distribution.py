"""Check feedback value distribution"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mongodb_database import MongoDatabase

def main():
    print("\n" + "="*70)
    print("FEEDBACK DISTRIBUTION ANALYSIS")
    print("="*70)
    
    try:
        db = MongoDatabase()
        
        # Get all feedback
        all_feedback = list(db.db.feedback.find({}, {'feedback_value': 1}))
        
        print(f"\nTotal feedback records: {len(all_feedback)}")
        
        if all_feedback:
            # Count feedback values
            from collections import Counter
            feedback_values = [f['feedback_value'] for f in all_feedback]
            distribution = Counter(feedback_values)
            
            print(f"\nFeedback value distribution:")
            for value, count in sorted(distribution.items()):
                percentage = (count / len(all_feedback)) * 100
                print(f"  Value {value:2d}: {count:4d} ({percentage:5.1f}%)")
            
            # Check if we have both positive and negative
            unique_values = set(feedback_values)
            print(f"\nUnique feedback values: {sorted(unique_values)}")
            
            # Convert to binary labels (as the model does)
            binary_labels = [1 if val > 0 else 0 for val in feedback_values]
            binary_dist = Counter(binary_labels)
            
            print(f"\nBinary label distribution (for ML):")
            for label, count in sorted(binary_dist.items()):
                label_name = "Positive (1)" if label == 1 else "Negative (0)"
                percentage = (count / len(binary_labels)) * 100
                print(f"  {label_name}: {count:4d} ({percentage:5.1f}%)")
            
            if len(binary_dist) < 2:
                print(f"\n❌ PROBLEM: Only one class present!")
                print(f"   All feedback values are {'positive' if 1 in binary_dist else 'negative'}")
                print(f"   ML training requires both positive and negative samples")
                print(f"\n💡 Solution:")
                print(f"   1. Delete existing feedback: db.feedback.delete_many({{}})")
                print(f"   2. Re-run add_feedback_quick.py with better distribution")
            else:
                print(f"\n✅ Both classes present - training should work")
                minority_class = min(binary_dist.values())
                if minority_class < 10:
                    print(f"\n⚠️  Warning: Minority class has only {minority_class} samples")
                    print(f"   This may cause issues with train/test split")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
