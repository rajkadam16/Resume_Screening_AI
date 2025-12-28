"""Debug training script - captures full error output"""

import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_pipeline_improved import trigger_improved_training, get_improved_training_status

def main():
    """Train with full error capture"""
    print("\n" + "="*70)
    print("DEBUG TRAINING SCRIPT")
    print("="*70)
    
    try:
        # Check status
        print("\n🔍 Checking status...")
        status = get_improved_training_status()
        
        print(f"\nReady for training: {status['ready_for_training']}")
        print(f"Message: {status['message']}")
        print(f"Labeled samples: {status['statistics'].get('labeled_samples', 0)}")
        
        if not status['ready_for_training']:
            print(f"\n❌ Not ready: {status['message']}")
            return
        
        print("\n🚀 Starting training...")
        result = trigger_improved_training(use_smote=True)
        
        if result['success']:
            print(f"\n✅ SUCCESS!")
            print(f"Message: {result['message']}")
            metrics = result['metrics']
            print(f"\nMetrics:")
            print(f"  Accuracy:  {metrics['accuracy']:.2%}")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall:    {metrics['recall']:.2%}")
            print(f"  F1 Score:  {metrics['f1_score']:.2%}")
        else:
            print(f"\n❌ FAILED!")
            print(f"Error: {result.get('error', 'Unknown')}")
            print(f"Message: {result.get('message', 'No message')}")
            if 'traceback' in result:
                print(f"\nFull traceback:")
                print(result['traceback'])
    
    except Exception as e:
        print(f"\n❌ EXCEPTION OCCURRED!")
        print(f"Error: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
