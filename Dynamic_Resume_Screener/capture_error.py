"""Capture full error to file"""

import sys
import os

# Redirect all output to file
log_file = open('training_error.log', 'w', encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from training_pipeline_improved import trigger_improved_training, get_improved_training_status
    
    print("="*70)
    print("TRAINING WITH FULL ERROR CAPTURE")
    print("="*70)
    
    # Check status
    print("\nChecking status...")
    status = get_improved_training_status()
    
    print(f"Ready: {status['ready_for_training']}")
    print(f"Message: {status['message']}")
    print(f"Samples: {status['statistics'].get('labeled_samples', 0)}")
    
    if not status['ready_for_training']:
        print(f"\nNot ready: {status['message']}")
    else:
        print("\nStarting training...")
        result = trigger_improved_training(use_smote=True)
        
        if result['success']:
            print("\nSUCCESS!")
            print(f"Message: {result['message']}")
            metrics = result['metrics']
            print(f"\nMetrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        else:
            print("\nFAILED!")
            print(f"Error: {result.get('error', 'Unknown')}")
            print(f"Message: {result.get('message', 'No message')}")
            if 'traceback' in result:
                print(f"\nFull traceback:")
                print(result['traceback'])

except Exception as e:
    print(f"\nEXCEPTION!")
    print(f"Error: {e}")
    print(f"\nFull traceback:")
    import traceback
    traceback.print_exc()

finally:
    log_file.close()
    # Restore stdout/stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Error log written to training_error.log")
