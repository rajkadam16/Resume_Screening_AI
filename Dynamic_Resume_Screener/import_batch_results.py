"""
Import Batch Results to Database
Imports the 228 analyzed resumes from batch_results.json into the database
and adds automatic feedback to create labeled samples for ML training
"""

import json
from mongodb_database import MongoDatabase
from datetime import datetime

def import_batch_results(json_file='batch_results.json'):
    """Import batch results and add feedback"""
    
    print("\n" + "=" * 70)
    print("  IMPORT BATCH RESULTS")
    print("=" * 70 + "\n")
    
    try:
        # Load JSON file
        print(f"📂 Loading {json_file}...")
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        print(f"✅ Found {len(results)} analyzed resumes\n")
        
        # Connect to database
        db = MongoDatabase()
        
        imported_count = 0
        feedback_count = 0
        skipped_count = 0
        
        for idx, analysis in enumerate(results):
            try:
                filename = analysis.get('filename', f'imported_resume_{idx}.txt')
                resume_text = analysis.get('resume_text', '')
                
                if not resume_text or len(resume_text) < 50:
                    skipped_count += 1
                    continue
                
                # Save resume
                resume_id = db.save_resume(
                    filename=filename,
                    resume_text=resume_text,
                    word_count=len(resume_text.split()),
                    has_education=analysis.get('has_education', False),
                    experience_years=analysis.get('experience_years', 0)
                )
                
                # Save analysis
                analysis_id = db.save_analysis_result(
                    resume_id=resume_id,
                    analysis=analysis,
                    job_description_id=None
                )
                
                # Add automatic feedback based on score
                score = analysis.get('match_percentage') or analysis.get('score', 0)
                
                if score >= 70:
                    feedback_value = 1  # Positive
                    feedback_text = "Good match based on score"
                elif score >= 50:
                    feedback_value = 0  # Neutral
                    feedback_text = "Average match"
                else:
                    feedback_value = -1  # Negative
                    feedback_text = "Poor match based on score"
                
                # Save feedback
                db.save_feedback(
                    analysis_id=analysis_id,
                    feedback_type='auto_import',
                    feedback_value=feedback_value,
                    comments=feedback_text
                )
                
                imported_count += 1
                feedback_count += 1
                
                if imported_count % 20 == 0:
                    print(f"  ✓ Imported {imported_count} resumes...")
            
            except Exception as e:
                print(f"  ⚠ Error importing resume {idx}: {e}")
                skipped_count += 1
        
        print()
        print("=" * 70)
        print("  IMPORT COMPLETE")
        print("=" * 70)
        print(f"✅ Successfully imported: {imported_count} resumes")
        print(f"✅ Feedback added: {feedback_count}")
        print(f"⚠️  Skipped: {skipped_count}")
        print()
        
        # Check total feedback
        stats = db.get_statistics()
        print(f"📊 Total labeled samples in database: {stats['labeled_samples']}")
        
        if stats['labeled_samples'] >= 50:
            print(f"✅ READY FOR TRAINING!")
            print(f"   Run: python train_improved_model.py")
        else:
            print(f"⚠️  Need {50 - stats['labeled_samples']} more samples")
        
        print("=" * 70 + "\n")
        
        return imported_count
        
    except FileNotFoundError:
        print(f"❌ File not found: {json_file}")
        print(f"   Make sure you've run the batch processor first!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    import_batch_results()
