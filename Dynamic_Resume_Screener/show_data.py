import sqlite3
import config

conn = sqlite3.connect(config.DATABASE_PATH)
cursor = conn.cursor()

print("\n" + "="*60)
print("  YOUR SQLITE DATABASE")
print("="*60)

# Get counts
cursor.execute("SELECT COUNT(*) FROM resumes")
resumes = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM analysis_results")
analyses = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM feedback")
feedback = cursor.fetchone()[0]

print(f"\nTotal Records:")
print(f"  Resumes: {resumes}")
print(f"  Analyses: {analyses}")
print(f"  Feedback: {feedback}")

if resumes > 0:
    print(f"\n" + "="*60)
    print("Recent Resumes:")
    print("="*60)
    cursor.execute("SELECT filename, upload_time, experience_years FROM resumes ORDER BY upload_time DESC LIMIT 5")
    for row in cursor.fetchall():
        print(f"\n  File: {row[0]}")
        print(f"  Uploaded: {row[1]}")
        print(f"  Experience: {row[2]} years")

if analyses > 0:
    print(f"\n" + "="*60)
    print("Recent Analyses:")
    print("="*60)
    cursor.execute("""
        SELECT r.filename, ar.recommendation, ar.match_percentage, ar.score
        FROM analysis_results ar
        JOIN resumes r ON ar.resume_id = r.id
        ORDER BY ar.analysis_time DESC LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"\n  Resume: {row[0]}")
        print(f"  Result: {row[1]}")
        if row[2]:
            print(f"  Match: {row[2]}%")
        if row[3]:
            print(f"  Score: {row[3]}")

print("\n" + "="*60)
print("Database file: data/resume_screener.db")
print("="*60 + "\n")

conn.close()
