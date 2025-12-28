"""Check model metadata in MongoDB"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mongodb_database import MongoDatabase

def main():
    print("\n" + "="*70)
    print("MODEL METADATA CHECK")
    print("="*70)
    
    try:
        db = MongoDatabase()
        
        # Check all model metadata
        all_models = list(db.db.model_metadata.find({}))
        print(f"\nTotal model records: {len(all_models)}")
        
        if all_models:
            print(f"\nAll models:")
            for model in all_models:
                print(f"\n  Model ID: {model['_id']}")
                print(f"  Type: {model.get('model_type', 'Unknown')}")
                print(f"  Version: {model.get('model_version', 'Unknown')}")
                print(f"  Active: {model.get('is_active', False)}")
                print(f"  Training time: {model.get('training_time', 'Unknown')}")
                print(f"  Training samples: {model.get('training_samples', 0)}")
                if 'metrics' in model:
                    metrics = model['metrics']
                    print(f"  Metrics:")
                    for key, value in metrics.items():
                        if value is not None:
                            print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
                print(f"  Model path: {model.get('model_path', 'Unknown')}")
                
                # Check if file exists
                model_path = model.get('model_path')
                if model_path:
                    exists = os.path.exists(model_path)
                    print(f"  File exists: {exists}")
        else:
            print("\n❌ No model metadata found in database!")
            print("\nPossible reasons:")
            print("  1. Model training didn't complete")
            print("  2. Model metadata save failed")
            print("  3. Wrong database connection")
        
        # Check active model
        print(f"\n" + "="*70)
        active_model = db.get_active_model('classifier')
        if active_model:
            print("✅ Active model found:")
            print(f"  Version: {active_model.get('model_version', 'Unknown')}")
            print(f"  F1 Score: {active_model.get('metrics', {}).get('f1_score', 0):.4f}")
        else:
            print("❌ No active model found")
        
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
