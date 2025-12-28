# Check ML Training Status
# Shows how many samples you have and what you need to do

from database import db
from training_pipeline import get_training_status
import json

print("\n" + "="*60)
print("  ML MODEL TRAINING STATUS")
print("="*60 + "\n")

# Get current status
status = get_training_status()
stats = status['statistics']

# Show progress bar
labeled = stats['labeled_samples']
required = status['min_samples_required']
percentage = (labeled / required) * 100 if required > 0 else 0

print(f"📊 Training Data Progress:")
print(f"   Labeled Samples: {labeled}/{required}")
print(f"   Progress: {percentage:.1f}%")

# Visual progress bar
bar_length = 40
filled = int((labeled / required) * bar_length) if required > 0 else 0
bar = "█" * filled + "░" * (bar_length - filled)
print(f"   [{bar}]")
print()

# Show what's needed
if status['ready_for_training']:
    print("✅ READY TO TRAIN!")
    print(f"   You have enough samples to train your model.")
    print()
    print("   Run this to train:")
    print("   python -c \"from training_pipeline import trigger_training; print(trigger_training())\"")
else:
    need_more = required - labeled
    print(f"⏳ NOT READY YET")
    print(f"   Need {need_more} more labeled samples")
    print()
    print("   How to get more samples:")
    print("   1. Upload resumes through the web app")
    print("   2. Click 👍 or 👎 on each result")
    print(f"   3. Repeat {need_more} more times")

print()
print("="*60)
print("  CURRENT DATABASE STATS")
print("="*60)
print(f"   Total Resumes: {stats['total_resumes']}")
print(f"   Total Analyses: {stats['total_analyses']}")
print(f"   Total Feedback: {stats['total_feedback']}")
print(f"   Labeled Samples: {stats['labeled_samples']}")
print(f"   Active Models: {stats['active_models']}")
print()

if stats['active_models'] > 0:
    print("✅ You have an active ML model!")
    print("   The system is using ML predictions.")
else:
    print("ℹ️  No ML model yet - using rule-based system")
    print("   Collect more feedback to train your first model!")

print()
print("="*60)
print("📚 For detailed guide, see: HOW_TO_TRAIN_MODEL.md")
print("="*60 + "\n")
