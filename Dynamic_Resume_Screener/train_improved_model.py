# Train Improved Ensemble Model
# Run this script to train the improved ML model with XGBoost, LightGBM, etc.

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_pipeline_improved import trigger_improved_training, get_improved_training_status
from ml_models_improved import improved_model_manager


def main():
    """Train the improved ensemble model"""
    print("\n" + "="*70)
    print("IMPROVED RESUME SCREENING MODEL TRAINER")
    print("="*70)
    
    # Check status
    print("\n🔍 Checking training readiness...")
    status = get_improved_training_status()
    
    print(f"\n📊 Current Status:")
    print(f"   Model type: {status['model_type']}")
    print(f"   Ready for training: {status['ready_for_training']}")
    print(f"   {status['message']}")
    print(f"   Has active model: {status['has_active_model']}")
    
    stats = status['statistics']
    print(f"\n📈 Database Statistics:")
    print(f"   Total resumes: {stats.get('total_resumes', 0)}")
    print(f"   Total analyses: {stats.get('total_analyses', 0)}")
    print(f"   Total feedback: {stats.get('total_feedback', 0)}")
    print(f"   Labeled samples: {stats.get('labeled_samples', 0)}")
    print(f"   Active models: {stats.get('active_models', 0)}")
    
    # Check if ready
    if not status['ready_for_training']:
        print(f"\n❌ Cannot train: {status['message']}")
        print(f"\n💡 To get training data:")
        print(f"   1. Upload resumes through the web interface")
        print(f"   2. Provide feedback on the results")
        print(f"   3. Need at least {status['min_samples_required']} labeled samples")
        return
    
    # Ask for confirmation
    print(f"\n🚀 Ready to train improved ensemble model!")
    print(f"   This will train 4 models: XGBoost, LightGBM, Random Forest, Gradient Boosting")
    print(f"   SMOTE will be applied to balance classes")
    print(f"   Evaluation plots will be generated")
    
    response = input(f"\n   Continue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n❌ Training cancelled.")
        return
    
    # Train model
    print(f"\n🎯 Starting training...")
    result = trigger_improved_training(use_smote=True)
    
    if result['success']:
        print(f"\n✅ {result['message']}")
        
        metrics = result['metrics']
        print(f"\n🎊 Training completed successfully!")
        print(f"\n📊 Final Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.1%}")
        print(f"   Precision: {metrics['precision']:.1%}")
        print(f"   Recall:    {metrics['recall']:.1%}")
        print(f"   F1 Score:  {metrics['f1_score']:.1%}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.1%}")
        
        if 'cv_mean' in metrics:
            print(f"\n🔄 Cross-Validation F1: {metrics['cv_mean']:.1%} (+/- {metrics['cv_std']:.1%})")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Check evaluation plots in: models/evaluation_plots/")
        print(f"   2. Test the model by uploading new resumes")
        print(f"   3. The improved model is now active!")
        
    else:
        print(f"\n❌ Training failed: {result['message']}")
        if 'traceback' in result:
            print(f"\n📋 Error details:")
            print(result['traceback'])


if __name__ == '__main__':
    main()
