"""Check database status and explain labeled samples"""

from mongodb_database import MongoDatabase

def check_status():
    print("\n" + "=" * 70)
    print("  DATABASE STATUS CHECK")
    print("=" * 70 + "\n")
    
    try:
        db = MongoDatabase()
        stats = db.get_statistics()
        
        print("📊 Database Statistics:")
        print(f"   Total resumes: {stats['total_resumes']}")
        print(f"   Total analyses: {stats['total_analyses']}")
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   Labeled samples: {stats['labeled_samples']}")
        print(f"   Active models: {stats['active_models']}")
        print()
        
        # Check what's missing
        if stats['labeled_samples'] < 50:
            print(f"⚠️  NEED MORE LABELED SAMPLES")
            print(f"   Current: {stats['labeled_samples']}")
            print(f"   Required: 50")
            print(f"   Missing: {50 - stats['labeled_samples']}")
            print()
            print("💡 What are 'labeled samples'?")
            print("   Labeled samples = Analyses that have FEEDBACK")
            print()
            print("   You have:")
            print(f"   ✅ {stats['total_analyses']} resumes analyzed")
            print(f"   ❌ Only {stats['labeled_samples']} with feedback")
            print()
            print("📝 How to add feedback:")
            print("   1. Use web interface: http://localhost:5000")
            print("   2. Run: python add_feedback_quick.py")
            print()
        else:
            print(f"✅ READY FOR TRAINING!")
            print(f"   You have {stats['labeled_samples']} labeled samples")
            print()
            print("🚀 Next step:")
            print("   Run: python train_improved_model.py")
            print()
        
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_status()
