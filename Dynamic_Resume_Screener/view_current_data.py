# View Current SQLite Data
# Shows what data you currently have in your database

import sqlite3
import json
from datetime import datetime
import config

def view_sqlite_data():
    """Display current SQLite database contents"""
    
    print("=" * 60)
    print("  Current Database Contents (SQLite)")
    print("=" * 60)
    print()
    
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) as count FROM resumes")
    resume_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM analysis_results")
    analysis_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM feedback")
    feedback_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM job_descriptions")
    jd_count = cursor.fetchone()['count']
    
    print(f"📊 Database Statistics:")
    print(f"   - Resumes: {resume_count}")
    print(f"   - Job Descriptions: {jd_count}")
    print(f"   - Analyses: {analysis_count}")
    print(f"   - Feedback: {feedback_count}")
    print()
    
    # Show recent resumes
    if resume_count > 0:
        print("=" * 60)
        print("📄 Recent Resumes (Last 5)")
        print("=" * 60)
        cursor.execute("""
            SELECT id, filename, upload_time, word_count, experience_years
            FROM resumes
            ORDER BY upload_time DESC
            LIMIT 5
        """)
        
        for row in cursor.fetchall():
            print(f"\nID: {row['id']}")
            print(f"  Filename: {row['filename']}")
            print(f"  Uploaded: {row['upload_time']}")
            print(f"  Words: {row['word_count']}")
            print(f"  Experience: {row['experience_years']} years")
    
    # Show recent analyses
    if analysis_count > 0:
        print()
        print("=" * 60)
        print("📊 Recent Analyses (Last 5)")
        print("=" * 60)
        cursor.execute("""
            SELECT ar.id, r.filename, ar.mode, ar.match_percentage, 
                   ar.score, ar.recommendation, ar.analysis_time
            FROM analysis_results ar
            JOIN resumes r ON ar.resume_id = r.id
            ORDER BY ar.analysis_time DESC
            LIMIT 5
        """)
        
        for row in cursor.fetchall():
            print(f"\nAnalysis ID: {row['id']}")
            print(f"  Resume: {row['filename']}")
            print(f"  Mode: {row['mode']}")
            if row['match_percentage']:
                print(f"  Match: {row['match_percentage']}%")
            if row['score']:
                print(f"  Score: {row['score']}")
            print(f"  Recommendation: {row['recommendation']}")
            print(f"  Time: {row['analysis_time']}")
    
    conn.close()
    
    print()
    print("=" * 60)
    print("💡 Your data is currently in SQLite")
    print("   To migrate to MongoDB, follow the setup guides:")
    print("   - QUICK_START_MONGODB_ATLAS.md (cloud, easiest)")
    print("   - INSTALL_MONGODB_WINDOWS.md (local install)")
    print("=" * 60)

if __name__ == "__main__":
    view_sqlite_data()
