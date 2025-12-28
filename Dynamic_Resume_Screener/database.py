# Database module for ML-powered Resume Screening
# Handles data persistence for training and feedback

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import config


class Database:
    """Database handler for resume screening data"""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection"""
        self.db_path = db_path or config.DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Resumes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resume_text TEXT NOT NULL,
                word_count INTEGER,
                has_education BOOLEAN,
                experience_years INTEGER
            )
        ''')
        
        # Job descriptions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description_text TEXT NOT NULL,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                required_skills TEXT,
                experience_years INTEGER
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER NOT NULL,
                job_description_id INTEGER,
                analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mode TEXT NOT NULL,
                match_percentage REAL,
                score REAL,
                recommendation TEXT,
                recommendation_class TEXT,
                required_skills_found TEXT,
                missing_skills TEXT,
                contact_info TEXT,
                certifications TEXT,
                quality_score REAL,
                FOREIGN KEY (resume_id) REFERENCES resumes (id),
                FOREIGN KEY (job_description_id) REFERENCES job_descriptions (id)
            )
        ''')
        
        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                feedback_value INTEGER,
                feedback_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                comments TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analysis_results (id)
            )
        ''')
        
        # Training data table (preprocessed for ML)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER NOT NULL,
                job_description_id INTEGER,
                features TEXT NOT NULL,
                label INTEGER NOT NULL,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (resume_id) REFERENCES resumes (id),
                FOREIGN KEY (job_description_id) REFERENCES job_descriptions (id)
            )
        ''')
        
        # Model metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                training_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                training_samples INTEGER,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                model_path TEXT,
                is_active BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_resume(self, filename: str, resume_text: str, 
                   word_count: int, has_education: bool, 
                   experience_years: int) -> int:
        """Save resume to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resumes (filename, resume_text, word_count, has_education, experience_years)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, resume_text, word_count, has_education, experience_years))
        
        resume_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return resume_id
    
    def save_job_description(self, description_text: str, 
                            required_skills: List[str], 
                            experience_years: int) -> int:
        """Save job description to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        skills_json = json.dumps(required_skills)
        
        cursor.execute('''
            INSERT INTO job_descriptions (description_text, required_skills, experience_years)
            VALUES (?, ?, ?)
        ''', (description_text, skills_json, experience_years))
        
        jd_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jd_id
    
    def save_analysis_result(self, resume_id: int, analysis: Dict, 
                            job_description_id: Optional[int] = None) -> int:
        """Save analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract fields from analysis dict
        mode = analysis.get('mode', 'standalone')
        match_percentage = analysis.get('match_percentage')
        score = analysis.get('score')
        recommendation = analysis.get('recommendation')
        recommendation_class = analysis.get('recommendation_class')
        
        # Convert lists/dicts to JSON
        required_skills_found = json.dumps(analysis.get('required_skills_found', []))
        missing_skills = json.dumps(analysis.get('missing_skills', []))
        contact_info = json.dumps(analysis.get('contact_info', {}))
        certifications = json.dumps(analysis.get('certifications', {}))
        quality_score = analysis.get('quality', {}).get('score')
        
        cursor.execute('''
            INSERT INTO analysis_results (
                resume_id, job_description_id, mode, match_percentage, score,
                recommendation, recommendation_class, required_skills_found,
                missing_skills, contact_info, certifications, quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (resume_id, job_description_id, mode, match_percentage, score,
              recommendation, recommendation_class, required_skills_found,
              missing_skills, contact_info, certifications, quality_score))
        
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return analysis_id
    
    def save_feedback(self, analysis_id: int, feedback_type: str, 
                     feedback_value: int, comments: str = None) -> int:
        """Save user feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (analysis_id, feedback_type, feedback_value, comments)
            VALUES (?, ?, ?, ?)
        ''', (analysis_id, feedback_type, feedback_value, comments))
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return feedback_id
    
    def get_training_data_count(self) -> int:
        """Get count of available training samples"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM feedback
            WHERE feedback_value IS NOT NULL
        ''')
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def get_training_data(self, limit: Optional[int] = None) -> List[Tuple]:
        """Fetch training data with labels from feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT 
                r.resume_text,
                jd.description_text,
                ar.match_percentage,
                ar.score,
                ar.quality_score,
                ar.required_skills_found,
                ar.missing_skills,
                f.feedback_value
            FROM feedback f
            JOIN analysis_results ar ON f.analysis_id = ar.id
            JOIN resumes r ON ar.resume_id = r.id
            LEFT JOIN job_descriptions jd ON ar.job_description_id = jd.id
            WHERE f.feedback_value IS NOT NULL
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        data = cursor.fetchall()
        conn.close()
        
        return data
    
    def save_model_metadata(self, model_version: str, model_type: str,
                           training_samples: int, metrics: Dict,
                           model_path: str) -> int:
        """Save model training metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deactivate previous models of same type
        cursor.execute('''
            UPDATE model_metadata 
            SET is_active = 0 
            WHERE model_type = ? AND is_active = 1
        ''', (model_type,))
        
        # Insert new model metadata
        cursor.execute('''
            INSERT INTO model_metadata (
                model_version, model_type, training_samples,
                accuracy, precision_score, recall, f1_score,
                model_path, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', (model_version, model_type, training_samples,
              metrics.get('accuracy'), metrics.get('precision'),
              metrics.get('recall'), metrics.get('f1_score'),
              model_path))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return model_id
    
    def get_active_model(self, model_type: str) -> Optional[Dict]:
        """Get active model metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_metadata
            WHERE model_type = ? AND is_active = 1
            ORDER BY training_time DESC
            LIMIT 1
        ''', (model_type,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total resumes
        cursor.execute('SELECT COUNT(*) FROM resumes')
        stats['total_resumes'] = cursor.fetchone()[0]
        
        # Total analyses
        cursor.execute('SELECT COUNT(*) FROM analysis_results')
        stats['total_analyses'] = cursor.fetchone()[0]
        
        # Total feedback
        cursor.execute('SELECT COUNT(*) FROM feedback')
        stats['total_feedback'] = cursor.fetchone()[0]
        
        # Feedback with labels
        cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback_value IS NOT NULL')
        stats['labeled_samples'] = cursor.fetchone()[0]
        
        # Active models
        cursor.execute('SELECT COUNT(*) FROM model_metadata WHERE is_active = 1')
        stats['active_models'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats


# Initialize database on module import
db = Database()
