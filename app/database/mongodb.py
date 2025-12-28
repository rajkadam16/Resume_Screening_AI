# MongoDB Database Module for Resume Screening AI
# Replaces SQLite with MongoDB for better scalability

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from bson import ObjectId
import os
import config

class MongoDatabase:
    """MongoDB handler for resume screening data"""
    
    def __init__(self, connection_string: str = None, db_name: str = "resume_screener"):
        """Initialize MongoDB connection"""
        # Get connection string from environment or use default
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URI', 
            'mongodb://localhost:27017/'
        )
        self.db_name = db_name
        
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            print(f"[OK] Connected to MongoDB: {self.db_name}")
            
            # Initialize collections and indexes
            self.init_database()
        except ConnectionFailure as e:
            print(f"[ERROR] Failed to connect to MongoDB: {e}")
            raise
    
    def init_database(self):
        """Create collections and indexes"""
        # Create indexes for better query performance
        
        # Resumes collection
        self.db.resumes.create_index([("filename", ASCENDING)])
        self.db.resumes.create_index([("upload_time", DESCENDING)])
        
        # Job descriptions collection
        self.db.job_descriptions.create_index([("created_time", DESCENDING)])
        
        # Analysis results collection
        self.db.analysis_results.create_index([("resume_id", ASCENDING)])
        self.db.analysis_results.create_index([("job_description_id", ASCENDING)])
        self.db.analysis_results.create_index([("analysis_time", DESCENDING)])
        self.db.analysis_results.create_index([("match_percentage", DESCENDING)])
        self.db.analysis_results.create_index([("score", DESCENDING)])
        
        # Feedback collection
        self.db.feedback.create_index([("analysis_id", ASCENDING)])
        self.db.feedback.create_index([("feedback_time", DESCENDING)])
        
        # Model metadata collection
        self.db.model_metadata.create_index([("model_type", ASCENDING), ("is_active", DESCENDING)])
        self.db.model_metadata.create_index([("training_time", DESCENDING)])
    
    def save_resume(self, filename: str, resume_text: str, 
                   word_count: int, has_education: bool, 
                   experience_years: int) -> str:
        """Save resume to MongoDB"""
        resume_doc = {
            "filename": filename,
            "upload_time": datetime.utcnow(),
            "resume_text": resume_text,
            "word_count": word_count,
            "has_education": has_education,
            "experience_years": experience_years
        }
        
        result = self.db.resumes.insert_one(resume_doc)
        return str(result.inserted_id)
    
    def save_job_description(self, description_text: str, 
                            required_skills: List[str], 
                            experience_years: int) -> str:
        """Save job description to MongoDB"""
        jd_doc = {
            "description_text": description_text,
            "created_time": datetime.utcnow(),
            "required_skills": required_skills,  # Native array, not JSON string
            "experience_years": experience_years
        }
        
        result = self.db.job_descriptions.insert_one(jd_doc)
        return str(result.inserted_id)
    
    def save_analysis_result(self, resume_id: str, analysis: Dict, 
                            job_description_id: Optional[str] = None) -> str:
        """Save analysis result to MongoDB"""
        analysis_doc = {
            "resume_id": ObjectId(resume_id),
            "job_description_id": ObjectId(job_description_id) if job_description_id else None,
            "analysis_time": datetime.utcnow(),
            "mode": analysis.get('mode', 'standalone'),
            "match_percentage": analysis.get('match_percentage'),
            "score": analysis.get('score'),
            "recommendation": analysis.get('recommendation'),
            "recommendation_class": analysis.get('recommendation_class'),
            "required_skills_found": analysis.get('required_skills_found', []),
            "missing_skills": analysis.get('missing_skills', []),
            "contact_info": analysis.get('contact_info', {}),
            "certifications": analysis.get('certifications', {}),
            "quality_score": analysis.get('quality', {}).get('score'),
            "ml_used": analysis.get('used_ml', False),
            "ml_confidence": analysis.get('ml_confidence')
        }
        
        result = self.db.analysis_results.insert_one(analysis_doc)
        return str(result.inserted_id)
    
    def save_feedback(self, analysis_id: str, feedback_type: str, 
                     feedback_value: int, comments: str = None) -> str:
        """Save user feedback"""
        feedback_doc = {
            "analysis_id": ObjectId(analysis_id),
            "feedback_type": feedback_type,
            "feedback_value": feedback_value,
            "feedback_time": datetime.utcnow(),
            "comments": comments
        }
        
        result = self.db.feedback.insert_one(feedback_doc)
        return str(result.inserted_id)
    
    def get_training_data_count(self) -> int:
        """Get count of available training samples"""
        count = self.db.feedback.count_documents({
            "feedback_value": {"$ne": None}
        })
        return count
    
    def get_training_data(self, limit: Optional[int] = None) -> List[Tuple]:
        """Fetch training data with labels from feedback"""
        pipeline = [
            {
                "$match": {
                    "feedback_value": {"$ne": None}
                }
            },
            {
                "$lookup": {
                    "from": "analysis_results",
                    "localField": "analysis_id",
                    "foreignField": "_id",
                    "as": "analysis"
                }
            },
            {
                "$unwind": "$analysis"
            },
            {
                "$lookup": {
                    "from": "resumes",
                    "localField": "analysis.resume_id",
                    "foreignField": "_id",
                    "as": "resume"
                }
            },
            {
                "$unwind": "$resume"
            },
            {
                "$lookup": {
                    "from": "job_descriptions",
                    "localField": "analysis.job_description_id",
                    "foreignField": "_id",
                    "as": "job_description"
                }
            },
            {
                "$project": {
                    "resume_text": "$resume.resume_text",
                    "description_text": {
                        "$arrayElemAt": ["$job_description.description_text", 0]
                    },
                    "match_percentage": "$analysis.match_percentage",
                    "score": "$analysis.score",
                    "quality_score": "$analysis.quality_score",
                    "required_skills_found": "$analysis.required_skills_found",
                    "missing_skills": "$analysis.missing_skills",
                    "feedback_value": "$feedback_value"
                }
            }
        ]
        
        if limit:
            pipeline.append({"$limit": limit})
        
        results = list(self.db.feedback.aggregate(pipeline))
        
        # Convert to tuple format for compatibility
        training_data = []
        for doc in results:
            training_data.append((
                doc.get('resume_text', ''),
                doc.get('description_text', ''),
                doc.get('match_percentage'),
                doc.get('score'),
                doc.get('quality_score'),
                doc.get('required_skills_found', []),
                doc.get('missing_skills', []),
                doc.get('feedback_value')
            ))
        
        return training_data
    
    def save_model_metadata(self, model_version: str, model_type: str,
                           training_samples: int, metrics: Dict,
                           model_path: str) -> str:
        """Save model training metadata"""
        # Deactivate previous models of same type
        self.db.model_metadata.update_many(
            {"model_type": model_type, "is_active": True},
            {"$set": {"is_active": False}}
        )
        
        # Insert new model metadata
        model_doc = {
            "model_version": model_version,
            "model_type": model_type,
            "training_time": datetime.utcnow(),
            "training_samples": training_samples,
            "metrics": {
                "accuracy": metrics.get('accuracy'),
                "precision": metrics.get('precision'),
                "recall": metrics.get('recall'),
                "f1_score": metrics.get('f1_score')
            },
            "model_path": model_path,
            "is_active": True
        }
        
        result = self.db.model_metadata.insert_one(model_doc)
        return str(result.inserted_id)
    
    def get_active_model(self, model_type: str) -> Optional[Dict]:
        """Get active model metadata"""
        model = self.db.model_metadata.find_one(
            {"model_type": model_type, "is_active": True},
            sort=[("training_time", DESCENDING)]
        )
        
        if model:
            # Convert ObjectId to string for JSON serialization
            model['_id'] = str(model['_id'])
            return model
        return None
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {
            "total_resumes": self.db.resumes.count_documents({}),
            "total_analyses": self.db.analysis_results.count_documents({}),
            "total_feedback": self.db.feedback.count_documents({}),
            "labeled_samples": self.db.feedback.count_documents({"feedback_value": {"$ne": None}}),
            "active_models": self.db.model_metadata.count_documents({"is_active": True})
        }
        
        return stats
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")


# Initialize database on module import
try:
    db = MongoDatabase()
except Exception as e:
    print(f"Warning: Could not initialize MongoDB: {e}")
    print("Falling back to SQLite...")
    # Import SQLite database as fallback
    from database import db
