# Data Migration Script: SQLite to MongoDB
# Migrates all existing data from SQLite to MongoDB

import sqlite3
import json
from datetime import datetime
from mongodb_database import MongoDatabase
from bson import ObjectId
import config

def migrate_data():
    """Migrate all data from SQLite to MongoDB"""
    
    print("=" * 60)
    print("  SQLite to MongoDB Migration")
    print("=" * 60)
    print()
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(config.DATABASE_PATH)
    sqlite_conn.row_factory = sqlite3.Row  # Access columns by name
    cursor = sqlite_conn.cursor()
    
    # Connect to MongoDB
    mongo_db = MongoDatabase()
    
    # Track ID mappings for foreign keys
    resume_id_map = {}  # SQLite ID -> MongoDB ObjectId
    jd_id_map = {}
    analysis_id_map = {}
    
    try:
        # 1. Migrate Resumes
        print("📄 Migrating resumes...")
        cursor.execute("SELECT * FROM resumes")
        resumes = cursor.fetchall()
        
        for resume in resumes:
            mongo_id = mongo_db.save_resume(
                filename=resume['filename'],
                resume_text=resume['resume_text'],
                word_count=resume['word_count'],
                has_education=bool(resume['has_education']),
                experience_years=resume['experience_years']
            )
            resume_id_map[resume['id']] = mongo_id
        
        print(f"   ✅ Migrated {len(resumes)} resumes")
        
        # 2. Migrate Job Descriptions
        print("💼 Migrating job descriptions...")
        cursor.execute("SELECT * FROM job_descriptions")
        job_descriptions = cursor.fetchall()
        
        for jd in job_descriptions:
            required_skills = json.loads(jd['required_skills']) if jd['required_skills'] else []
            
            mongo_id = mongo_db.save_job_description(
                description_text=jd['description_text'],
                required_skills=required_skills,
                experience_years=jd['experience_years']
            )
            jd_id_map[jd['id']] = mongo_id
        
        print(f"   ✅ Migrated {len(job_descriptions)} job descriptions")
        
        # 3. Migrate Analysis Results
        print("📊 Migrating analysis results...")
        cursor.execute("SELECT * FROM analysis_results")
        analyses = cursor.fetchall()
        
        for analysis in analyses:
            # Parse JSON fields
            required_skills_found = json.loads(analysis['required_skills_found']) if analysis['required_skills_found'] else []
            missing_skills = json.loads(analysis['missing_skills']) if analysis['missing_skills'] else []
            contact_info = json.loads(analysis['contact_info']) if analysis['contact_info'] else {}
            certifications = json.loads(analysis['certifications']) if analysis['certifications'] else {}
            
            # Build analysis dict
            analysis_dict = {
                'mode': analysis['mode'],
                'match_percentage': analysis['match_percentage'],
                'score': analysis['score'],
                'recommendation': analysis['recommendation'],
                'recommendation_class': analysis['recommendation_class'],
                'required_skills_found': required_skills_found,
                'missing_skills': missing_skills,
                'contact_info': contact_info,
                'certifications': certifications,
                'quality': {'score': analysis['quality_score']}
            }
            
            # Map SQLite IDs to MongoDB ObjectIds
            resume_mongo_id = resume_id_map.get(analysis['resume_id'])
            jd_mongo_id = jd_id_map.get(analysis['job_description_id']) if analysis['job_description_id'] else None
            
            if resume_mongo_id:
                mongo_id = mongo_db.save_analysis_result(
                    resume_id=resume_mongo_id,
                    analysis=analysis_dict,
                    job_description_id=jd_mongo_id
                )
                analysis_id_map[analysis['id']] = mongo_id
        
        print(f"   ✅ Migrated {len(analyses)} analysis results")
        
        # 4. Migrate Feedback
        print("💬 Migrating feedback...")
        cursor.execute("SELECT * FROM feedback")
        feedbacks = cursor.fetchall()
        
        for feedback in feedbacks:
            analysis_mongo_id = analysis_id_map.get(feedback['analysis_id'])
            
            if analysis_mongo_id:
                mongo_db.save_feedback(
                    analysis_id=analysis_mongo_id,
                    feedback_type=feedback['feedback_type'],
                    feedback_value=feedback['feedback_value'],
                    comments=feedback['comments']
                )
        
        print(f"   ✅ Migrated {len(feedbacks)} feedback entries")
        
        # 5. Migrate Model Metadata
        print("🤖 Migrating model metadata...")
        cursor.execute("SELECT * FROM model_metadata")
        models = cursor.fetchall()
        
        for model in models:
            metrics = {
                'accuracy': model['accuracy'],
                'precision': model['precision_score'],
                'recall': model['recall'],
                'f1_score': model['f1_score']
            }
            
            mongo_db.save_model_metadata(
                model_version=model['model_version'],
                model_type=model['model_type'],
                training_samples=model['training_samples'],
                metrics=metrics,
                model_path=model['model_path']
            )
        
        print(f"   ✅ Migrated {len(models)} model metadata entries")
        
        # Verify migration
        print()
        print("=" * 60)
        print("  Migration Verification")
        print("=" * 60)
        
        stats = mongo_db.get_statistics()
        print(f"📊 MongoDB Statistics:")
        print(f"   - Resumes: {stats['total_resumes']}")
        print(f"   - Analyses: {stats['total_analyses']}")
        print(f"   - Feedback: {stats['total_feedback']}")
        print(f"   - Labeled Samples: {stats['labeled_samples']}")
        print(f"   - Active Models: {stats['active_models']}")
        
        print()
        print("✅ Migration completed successfully!")
        print()
        print("⚠️  IMPORTANT: Backup your SQLite database before removing it")
        print(f"   SQLite DB: {config.DATABASE_PATH}")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise
    
    finally:
        sqlite_conn.close()
        mongo_db.close()


if __name__ == "__main__":
    print()
    response = input("⚠️  This will migrate all data from SQLite to MongoDB. Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        migrate_data()
    else:
        print("Migration cancelled.")
