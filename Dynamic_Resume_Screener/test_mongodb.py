# Test MongoDB connection
from mongodb_database import MongoDatabase

def test_connection():
    """Test MongoDB connection and basic operations"""
    print("Testing MongoDB connection...")
    print()
    
    try:
        # Initialize database
        db = MongoDatabase()
        print("✅ MongoDB connection successful!")
        print()
        
        # Get statistics
        stats = db.get_statistics()
        print("📊 Database Statistics:")
        print(f"   - Total Resumes: {stats['total_resumes']}")
        print(f"   - Total Analyses: {stats['total_analyses']}")
        print(f"   - Total Feedback: {stats['total_feedback']}")
        print(f"   - Labeled Samples: {stats['labeled_samples']}")
        print(f"   - Active Models: {stats['active_models']}")
        print()
        
        # Test insert
        print("Testing insert operation...")
        resume_id = db.save_resume(
            filename="test_resume.pdf",
            resume_text="This is a test resume",
            word_count=5,
            has_education=True,
            experience_years=3
        )
        print(f"✅ Test resume inserted with ID: {resume_id}")
        print()
        
        # Clean up test data
        db.db.resumes.delete_one({"filename": "test_resume.pdf"})
        print("✅ Test data cleaned up")
        print()
        
        print("=" * 50)
        print("All tests passed! MongoDB is ready to use.")
        print("=" * 50)
        
        db.close()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure MongoDB is running")
        print("2. Check your connection string in .env")
        print("3. Verify network connectivity")
        return False
    
    return True

if __name__ == "__main__":
    test_connection()
