# Resume Screening AI

An intelligent resume screening system powered by Machine Learning and Natural Language Processing. Automatically analyze, score, and match resumes against job requirements with advanced ML ensemble models.

## 🚀 Features

### Core Capabilities
- **Smart Resume Analysis**: Advanced NLP-based parsing and analysis
- **Job Matching**: AI-powered resume-to-job matching with intelligent scoring
- **Ensemble ML Models**: XGBoost, LightGBM, Random Forest, Gradient Boosting
- **Skill Extraction**: Fuzzy matching for skills, certifications, and experience levels
- **Quality Scoring**: Comprehensive resume quality assessment and feedback
- **Batch Processing**: Process 100+ resumes per minute efficiently
- **Real-time Prediction**: Sub-100ms ML inference per resume
- **Database Integration**: MongoDB for scalability + SQLite fallback

### User Interface
- **Web Dashboard**: Clean, responsive interface for resume analysis
- **Bulk Upload**: Process multiple resumes simultaneously
- **Result Export**: CSV export for integration with HR systems
- **Feedback Loop**: Tag and collect training data for continuous improvement

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Database**: MongoDB 4.0+ (optional, SQLite fallback available)
- **System**: 4GB RAM minimum recommended
- **OS**: Windows, Linux, or macOS

## 🔧 Installation

### Step 1: Create Virtual Environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

For development/testing:
```bash
pip install -r requirements-dev.txt
```

### Step 3: Configure Environment (Optional)
```bash
# Database configuration (app uses SQLite by default)
# To use MongoDB, update config.py or set MONGODB_URI environment variable

# For development mode:
set FLASK_ENV=development
set FLASK_DEBUG=1
```

### Step 4: Run the Application
```bash
# Main web application
python Dynamic_Resume_Screener/app.py

# Opens at: http://localhost:5000
```

## 🎯 Quick Start

### 1. Upload & Analyze Resume
1. Navigate to `http://localhost:5000`
2. Upload a resume (PDF, DOCX, or TXT)
3. Optionally provide a job description
4. View instant analysis and ML scoring

### 2. Bulk Processing
```bash
python Dynamic_Resume_Screener/batch_resume_processor.py
```

### 3. Train Custom Model
```bash
# Step 1: Collect feedback (mark good/bad resumes)
python Dynamic_Resume_Screener/add_feedback_quick.py

# Step 2: Train model (requires 50+ samples)
python Dynamic_Resume_Screener/train_improved_model.py

# Step 3: Check results
python Dynamic_Resume_Screener/check_training_result.py
```

## 📁 Project Structure

```
Resume_Screening_AI/
├── Dynamic_Resume_Screener/    # Main application
│   ├── app.py                  # Flask web app
│   ├── ml_models_improved.py   # Ensemble ML models
│   ├── training_pipeline_improved.py  # Model training
│   ├── database.py             # Database interface
│   ├── mongodb_database.py     # MongoDB implementation
│   ├── config.py               # Configuration
│   ├── batch_resume_processor.py  # Bulk processing
│   └── templates/              # HTML templates
│       ├── index.html
│       ├── results.html
│       └── ...
├── app/                        # Package structure
│   ├── database/
│   │   ├── mongodb.py          # MongoDB drivers
│   │   └── sqlite.py           # SQLite drivers
│   ├── ml/
│   │   ├── models.py           # ML implementations
│   │   └── training.py         # Training logic
│   └── utils/
├── scripts/                    # Utility scripts
│   ├── data/                   # Data processing scripts
│   ├── ml/                     # ML training scripts
│   └── setup/                  # Initialization
├── tools/                      # Development tools
│   ├── check_db_status.py
│   ├── check_feedback_distribution.py
│   ├── view_data.py
│   └── ...
├── data/                       # Data storage
│   ├── database/               # Database files
│   ├── models/                 # Trained models
│   └── resumes/                # Sample resumes
├── tests/                      # Test suite
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                    # Package configuration
└── README.md                   # This file
```

## 🧪 Testing

```bash
# Run all tests with coverage
pytest --cov=app tests/

# Run specific test module
pytest tests/test_analyzer.py -v

# Run with detailed output
pytest -vv --tb=short
```

## 📊 Training & Improvement

### Data Collection
```bash
# Add feedback for resumes (mark as good/bad/neutral)
python Dynamic_Resume_Screener/add_feedback_quick.py

# Bulk import feedback from CSV
python Dynamic_Resume_Screener/import_batch_results.py

# View feedback distribution
python tools/check_feedback_distribution.py
```

### Model Training
```bash
# Train improved ensemble model (requires 50+ labeled samples)
python Dynamic_Resume_Screener/train_improved_model.py

# Check training status and results
python Dynamic_Resume_Screener/check_training_result.py

# View model metadata and performance
tools/check_model_metadata.py
```

### Development Tools
```bash
# Check database connection status
python tools/check_db_status.py

# View all stored resumes and scores
python tools/view_data.py

# Show feedback statistics
python Dynamic_Resume_Screener/check_feedback_count.py

# Verify feedback distribution
python Dynamic_Resume_Screener/check_feedback_distribution.py
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📚 API Documentation

### Upload Resume
```
POST /upload
Content-Type: multipart/form-data

Parameters:
- resume: file (PDF/DOCX/TXT)
- job_description: string (optional)

Response: JSON with analysis results
```

### Export Results
```
GET /export?format=csv
Response: CSV file download
```

See [docs/API.md](docs/API.md) for complete API documentation.

## 🛠️ Configuration

Edit `app/config.py` or use environment variables:

```python
# ML Configuration
MIN_TRAINING_SAMPLES = 50
CONFIDENCE_THRESHOLD = 0.7

# Database
MONGODB_URI = "mongodb://localhost:27017/"

# File Upload
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

## 📈 Performance

- **Analysis Speed**: ~2-5 seconds per resume
- **Batch Processing**: 100+ resumes/minute
- **ML Prediction**: <100ms per resume
- **Database**: Handles 10K+ resumes efficiently

## 🔒 Security Notes

⚠️ **Important**: This is a development version. For production use:

1. Enable authentication (see docs/DEPLOYMENT.md)
2. Use HTTPS/SSL
3. Configure proper MongoDB security
4. Set strong SECRET_KEY
5. Enable rate limiting
6. Review GDPR compliance

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👥 Authors

- **Raj Kadam** - Initial work

## 🙏 Acknowledgments

- Flask framework
- scikit-learn for ML
- MongoDB for database
- sentence-transformers for NLP

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Email: [your-email]
- Documentation: [docs/](docs/)

## 🗺️ Roadmap

- [ ] User authentication system
- [ ] REST API with JWT
- [ ] Advanced ML models (BERT, transformers)
- [ ] Real-time resume ranking
- [ ] Email notifications
- [ ] Multi-language support
- [ ] Mobile app

## 📊 Current Status

**Version**: 1.0.0-dev  
**Status**: Development  
**Production Ready**: No (see production_readiness_assessment.md)

---

**Note**: This project is currently in development. See `production_readiness_assessment.md` for production deployment requirements.
