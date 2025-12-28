"""Check if training was successful"""

from mongodb_database import MongoDatabase
import os

try:
    db = MongoDatabase()
    
    print(f"\n{'='*60}")
    print(f"  TRAINING STATUS CHECK")
    print(f"{'='*60}\n")
    
    # Check for active models
    active_model = db.get_active_model('classifier')
    
    if active_model:
        print("✅ TRAINING SUCCESSFUL!")
        print(f"\nModel Details:")
        print(f"  Version: {active_model.get('version', 'Unknown')}")
        print(f"  Created: {active_model.get('created_at', 'Unknown')}")
        
        if 'metrics' in active_model:
            metrics = active_model['metrics']
            print(f"\n📊 Model Performance:")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
            print(f"  Precision: {metrics.get('precision', 0):.2%}")
            print(f"  Recall:    {metrics.get('recall', 0):.2%}")
            print(f"  F1 Score:  {metrics.get('f1_score', 0):.2%}")
        
        # Check if model files exist
        print(f"\n📁 Model Files:")
        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            print(f"  Found {len(model_files)} model file(s)")
            for f in model_files[:5]:  # Show first 5
                print(f"    - {f}")
        else:
            print(f"  Models directory not found")
    else:
        print("❌ NO ACTIVE MODEL FOUND")
        print("\nPossible reasons:")
        print("  1. Training failed")
        print("  2. Training is still in progress")
        print("  3. Model was not saved properly")
    
    print(f"\n{'='*60}\n")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
