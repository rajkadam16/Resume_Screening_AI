"""
Quick Start: Feed Resumes to Your Model
Simple interactive script to get started quickly
"""

import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_option(number, title, description):
    """Print formatted option"""
    print(f"{number}. {title}")
    print(f"   {description}\n")

def main():
    print_header("Resume Screening AI - Quick Start")
    
    print("This script will help you feed resumes to your ML model.\n")
    
    print_option(
        "1",
        "Process Resumes from 'resumes' Folder",
        "Analyze all resumes in the resumes/ directory"
    )
    
    print_option(
        "2",
        "Download & Process Kaggle Dataset",
        "Download resume dataset from Kaggle and process it"
    )
    
    print_option(
        "3",
        "Train ML Model",
        "Train the model with existing data (requires feedback)"
    )
    
    print_option(
        "4",
        "Check System Status",
        "View database stats and training data availability"
    )
    
    print_option(
        "5",
        "Exit",
        "Exit the program"
    )
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        process_local_folder()
    elif choice == "2":
        process_kaggle_dataset()
    elif choice == "3":
        train_model()
    elif choice == "4":
        check_status()
    elif choice == "5":
        print("\nGoodbye! 👋\n")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please try again.\n")
        main()

def process_local_folder():
    """Process resumes from local folder"""
    print_header("Process Local Resumes")
    
    from batch_resume_processor import BatchResumeProcessor
    
    # Check if resumes folder exists and has files
    resumes_folder = 'resumes'
    if not os.path.exists(resumes_folder):
        print(f"❌ Folder '{resumes_folder}' not found!")
        print(f"   Please create it and add resume files.\n")
        return
    
    # Count files
    import glob
    resume_files = []
    for ext in ['*.pdf', '*.docx', '*.doc', '*.txt']:
        resume_files.extend(glob.glob(os.path.join(resumes_folder, ext)))
    
    if not resume_files:
        print(f"❌ No resume files found in '{resumes_folder}'!")
        print(f"   Please add PDF, DOCX, DOC, or TXT files.\n")
        return
    
    print(f"Found {len(resume_files)} resume file(s)\n")
    
    # Ask for job description
    use_jd = input("Do you want to use a job description? (y/n): ").strip().lower()
    
    job_description = None
    if use_jd == 'y':
        print("\nEnter job description (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        job_description = "\n".join(lines[:-1])  # Remove last empty line
    
    # Process
    print("\n🔄 Processing resumes...\n")
    processor = BatchResumeProcessor(resumes_folder=resumes_folder)
    results = processor.process_folder(
        job_description=job_description,
        save_to_db=True
    )
    
    # Export results
    output_file = 'batch_results.json'
    processor.export_results(results, output_file)
    
    print(f"\n✅ Results exported to: {output_file}\n")
    
    # Ask to continue
    input("Press Enter to return to main menu...")
    main()

def process_kaggle_dataset():
    """Download and process Kaggle dataset"""
    print_header("Process Kaggle Dataset")
    
    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed!")
        print("   Install with: pip install kagglehub\n")
        input("Press Enter to return to main menu...")
        main()
        return
    
    from batch_resume_processor import BatchResumeProcessor
    
    print("📥 Downloading dataset from Kaggle...")
    print("   (This may take a few minutes)\n")
    
    try:
        path = kagglehub.dataset_download("palaksood97/resume-dataset")
        print(f"✅ Dataset downloaded to: {path}\n")
    except Exception as e:
        print(f"❌ Download failed: {e}\n")
        input("Press Enter to return to main menu...")
        main()
        return
    
    # Ask for limit
    limit_input = input("How many resumes to process? (Enter number or 'all'): ").strip()
    
    limit = None
    if limit_input.lower() != 'all':
        try:
            limit = int(limit_input)
        except ValueError:
            print("Invalid number, processing all resumes...")
    
    # Ask for job description
    use_jd = input("\nDo you want to use a job description? (y/n): ").strip().lower()
    
    job_description = None
    if use_jd == 'y':
        print("\nEnter job description (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        job_description = "\n".join(lines[:-1])
    
    # Process
    print("\n🔄 Processing dataset...\n")
    processor = BatchResumeProcessor()
    results = processor.process_from_kaggle_dataset(
        dataset_path=path,
        job_description=job_description,
        limit=limit
    )
    
    # Export results
    output_file = 'kaggle_results.json'
    processor.export_results(results, output_file)
    
    print(f"\n✅ Results exported to: {output_file}\n")
    
    # Ask to continue
    input("Press Enter to return to main menu...")
    main()

def train_model():
    """Train the ML model"""
    print_header("Train ML Model")
    
    try:
        from database import db
        from training_pipeline_improved import trigger_improved_training, get_improved_training_status
    except ImportError:
        print("❌ ML modules not available!\n")
        input("Press Enter to return to main menu...")
        main()
        return
    
    # Check training data
    print("📊 Checking training data availability...\n")
    
    try:
        feedback_count = db.get_feedback_count()
        print(f"   Available feedback samples: {feedback_count}")
        
        if feedback_count < 50:
            print(f"\n⚠️  Need at least 50 samples to train")
            print(f"   You need {50 - feedback_count} more samples with feedback\n")
            print("   How to add feedback:")
            print("   1. Analyze resumes through the web interface")
            print("   2. Provide feedback (👍/😐/👎) on the results")
            print("   3. Come back when you have enough samples\n")
            input("Press Enter to return to main menu...")
            main()
            return
        
        print(f"   ✅ Sufficient data for training!\n")
        
        # Confirm training
        confirm = input("Start training? This may take several minutes (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("\n❌ Training cancelled\n")
            input("Press Enter to return to main menu...")
            main()
            return
        
        # Start training
        print("\n🔄 Starting training...\n")
        success, message = trigger_improved_training()
        
        if success:
            print(f"✅ {message}\n")
            
            # Monitor progress
            print("📊 Monitoring training progress...\n")
            import time
            while True:
                status = get_improved_training_status()
                print(f"   Status: {status['status']}")
                
                if status['status'] in ['completed', 'failed', 'idle']:
                    break
                
                time.sleep(2)
            
            if status['status'] == 'completed':
                print("\n✅ Training completed successfully!\n")
                if 'metrics' in status:
                    metrics = status['metrics']
                    print("   Model Performance:")
                    print(f"   - Accuracy: {metrics.get('accuracy', 0):.2%}")
                    print(f"   - Precision: {metrics.get('precision', 0):.2%}")
                    print(f"   - Recall: {metrics.get('recall', 0):.2%}")
                    print(f"   - F1 Score: {metrics.get('f1_score', 0):.2%}\n")
            else:
                print(f"\n❌ Training failed: {status.get('error', 'Unknown error')}\n")
        else:
            print(f"❌ {message}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    input("Press Enter to return to main menu...")
    main()

def check_status():
    """Check system status"""
    print_header("System Status")
    
    try:
        from database import db
        
        print("📊 Database Statistics:\n")
        
        # Get counts
        try:
            feedback_count = db.get_feedback_count()
            print(f"   Feedback samples: {feedback_count}")
        except:
            print("   Feedback samples: Unable to retrieve")
        
        # Check active model
        try:
            active_model = db.get_active_model('classifier')
            if active_model:
                print(f"   Active model: ✅ Yes")
                print(f"   Model version: {active_model.get('version', 'Unknown')}")
                if 'metrics' in active_model:
                    metrics = active_model['metrics']
                    print(f"   Model accuracy: {metrics.get('accuracy', 0):.2%}")
            else:
                print(f"   Active model: ❌ No")
        except:
            print("   Active model: Unable to check")
        
        print("\n📁 Files:\n")
        
        # Check resumes folder
        import glob
        resume_files = []
        for ext in ['*.pdf', '*.docx', '*.doc', '*.txt']:
            resume_files.extend(glob.glob(os.path.join('resumes', ext)))
        print(f"   Resumes in folder: {len(resume_files)}")
        
        # Check models folder
        if os.path.exists('models'):
            model_files = glob.glob('models/*.pkl')
            print(f"   Saved models: {len(model_files)}")
        else:
            print(f"   Saved models: 0 (models folder not found)")
        
        print("\n🌐 Web Application:\n")
        print(f"   URL: http://localhost:5000")
        print(f"   Status: Check terminal where app.py is running\n")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}\n")
    
    input("Press Enter to return to main menu...")
    main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
