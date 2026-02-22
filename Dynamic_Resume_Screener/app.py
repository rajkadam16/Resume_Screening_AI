# Dynamic Resume Screener Application - Optimized Version with ML
# Main Flask application file with advanced NLP, filtering, and ML capabilities
# Performance optimized: pre-compiled regex, cached constants, efficient matching

from flask import Flask, render_template, request, jsonify, session, send_file, make_response
from werkzeug.utils import secure_filename
import os
import re
from datetime import datetime
import secrets
from difflib import SequenceMatcher
import json
import csv
import io

# ============================================================
# PRE-COMPILED REGEX PATTERNS (compiled once at startup)
# ============================================================
RE_WHITESPACE = re.compile(r'\s+')
RE_SPECIAL_CHARS = re.compile(r'[^\w\s@.+\-():/]')
RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
RE_PHONE_PATTERNS = [
    re.compile(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    re.compile(r'\d{10}'),
]
RE_EXP_PATTERNS = [
    re.compile(r'(\d+)\+?\s*years?\s*(?:of)?\s*experience', re.IGNORECASE),
    re.compile(r'experience\s*[:–-]\s*(\d+)\+?\s*years?', re.IGNORECASE),
    re.compile(r'(\d+)\+?\s*years?\s*in', re.IGNORECASE),
]
RE_LINKEDIN = re.compile(r'linkedin\.com/in/[\w-]+', re.IGNORECASE)
RE_GITHUB = re.compile(r'github\.com/[\w-]+', re.IGNORECASE)

# ============================================================
# MODULE-LEVEL CONSTANTS (created once, reused everywhere)
# ============================================================
SKILLS_FLAT = [
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go',
    'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'asp.net', 'next.js',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'nosql', 'dynamodb',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform',
    'machine learning', 'deep learning', 'ai', 'data science', 'nlp', 'computer vision',
    'html', 'css', 'sass', 'bootstrap', 'tailwind',
    'rest api', 'graphql', 'microservices', 'agile', 'scrum', 'devops'
]

SKILLS_BY_CATEGORY = {
    'Programming Languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r'],
    'Web Frameworks': ['react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring boot', 'asp.net', 'laravel', 'next.js', 'svelte'],
    'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'nosql', 'dynamodb', 'cassandra', 'elasticsearch'],
    'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform', 'ansible'],
    'AI & Data Science': ['machine learning', 'deep learning', 'ai', 'data science', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'pandas', 'numpy'],
    'Frontend': ['html', 'css', 'sass', 'bootstrap', 'tailwind', 'jquery', 'webpack', 'responsive design'],
    'Other': ['rest api', 'graphql', 'microservices', 'agile', 'scrum', 'devops', 'testing', 'unit testing']
}

EDUCATION_KEYWORDS = frozenset([
    'bachelor', 'master', 'phd', 'degree', 'university', 'college',
    'b.tech', 'm.tech', 'mca', 'bca', 'b.s', 'm.s'
])

CERTIFICATION_KEYWORDS = {
    'Cloud': ['aws certified', 'azure certified', 'google cloud certified', 'gcp certified'],
    'Programming': ['oracle certified', 'microsoft certified', 'java certified', 'python certified'],
    'Project Management': ['pmp', 'prince2', 'scrum master', 'csm', 'psm'],
    'Security': ['cissp', 'ceh', 'security+', 'comptia security'],
    'Data': ['data science certified', 'machine learning certified', 'tensorflow certified'],
    'Other': ['certified', 'certification']
}

PROJECT_INDICATORS = frozenset([
    'project', 'developed', 'built', 'created', 'implemented',
    'designed', 'worked on', 'contributed to'
])

PROFESSIONAL_KEYWORDS = frozenset([
    'achieved', 'led', 'managed', 'improved', 'increased', 'developed'
])

SECTION_KEYWORDS = {
    'experience': ('experience', 'work history', 'employment'),
    'education': ('education', 'academic', 'degree'),
    'skills': ('skills', 'technical skills', 'competencies'),
}

MAX_RESUME_LENGTH = 5000  # Truncate oversized resumes for faster processing

# ============================================================
# ML IMPORTS - Try improved models first, fallback to original
# ============================================================
try:
    from database import db
    import config as ml_config

    try:
        from ml_models_improved import improved_model_manager as model_manager
        from training_pipeline_improved import improved_training_pipeline as training_pipeline
        from training_pipeline_improved import trigger_improved_training as trigger_training
        from training_pipeline_improved import get_improved_training_status as get_training_status
        print("Using IMPROVED ensemble models (XGBoost + LightGBM + RF + GB)")
        ML_ENABLED = True
        USING_IMPROVED_MODELS = True
    except ImportError as e:
        print(f"Improved models not available ({e}), using original models")
        from ml_models import model_manager
        from training_pipeline import training_pipeline, trigger_training, get_training_status
        ML_ENABLED = True
        USING_IMPROVED_MODELS = False
except ImportError as e:
    print(f"ML modules not available: {e}")
    ML_ENABLED = False
    USING_IMPROVED_MODELS = False

# ============================================================
# FLASK APP SETUP
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

UPLOAD_FOLDER = 'resumes'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ML system
if ML_ENABLED:
    try:
        model_manager.load_active_model()
        print("ML system initialized successfully")
        if model_manager.has_active_model():
            print("Active ML model loaded")
        else:
            print("No active model found - will use rule-based system")
    except Exception as e:
        print(f"Error initializing ML system: {e}")
        ML_ENABLED = False


# ============================================================
# UTILITY FUNCTIONS (optimized)
# ============================================================
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    text = RE_WHITESPACE.sub(' ', text)
    text = RE_SPECIAL_CHARS.sub(' ', text)
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    return text.strip()


def extract_experience_years(text_lower):
    """Extract years of experience from text (uses pre-compiled patterns)"""
    for pattern in RE_EXP_PATTERNS:
        matches = pattern.findall(text_lower)
        if matches:
            return max(int(m) for m in matches)
    return 0


def fuzzy_skill_match(skill, text_lower, threshold=0.85):
    """
    Match skills with fuzzy matching to catch variations.
    Optimized: accepts pre-lowered text, fast exact checks first.
    Returns: (matched, confidence_score)
    """
    skill_lower = skill.lower()

    # Fast exact substring match
    if skill_lower in text_lower:
        return True, 1.0

    # Try common variations (cheaper than SequenceMatcher)
    variations = [
        skill_lower.replace('.', ''),
        skill_lower.replace(' ', ''),
        skill_lower.replace('-', ''),
    ]

    # JS/JavaScript swap
    if 'js' in skill_lower:
        variations.append(skill_lower.replace('js', 'javascript'))
    elif 'javascript' in skill_lower:
        variations.append(skill_lower.replace('javascript', 'js'))

    for variation in variations:
        if variation and variation in text_lower:
            return True, 0.95

    # Expensive fuzzy matching — only for multi-char skills
    if len(skill_lower) >= 3:
        words = text_lower.split()
        for word in words:
            if len(word) >= 3:
                ratio = SequenceMatcher(None, skill_lower, word).ratio()
                if ratio >= threshold:
                    return True, ratio

    return False, 0.0


def extract_contact_info(text):
    """Extract contact information from resume (uses pre-compiled patterns)"""
    contact_info = {'email': None, 'phone': None, 'linkedin': None, 'github': None}

    # Email
    emails = RE_EMAIL.findall(text)
    if emails:
        contact_info['email'] = emails[0]

    # Phone
    for pattern in RE_PHONE_PATTERNS:
        phones = pattern.findall(text)
        if phones:
            contact_info['phone'] = phones[0]
            break

    # LinkedIn
    match = RE_LINKEDIN.search(text)
    if match:
        contact_info['linkedin'] = f"https://{match.group()}"

    # GitHub
    match = RE_GITHUB.search(text)
    if match:
        contact_info['github'] = f"https://{match.group()}"

    return contact_info


def extract_certifications(text):
    """Extract certifications from resume (uses cached keyword dict)"""
    text_lower = text.lower()
    found_certifications = {}

    for category, certs in CERTIFICATION_KEYWORDS.items():
        category_certs = []
        for cert in certs:
            if cert in text_lower:
                pattern = rf'([A-Z][A-Za-z\s]*{re.escape(cert)}[A-Za-z\s]*)'
                matches = re.findall(pattern, text, re.IGNORECASE)
                category_certs.append(matches[0].strip() if matches else cert.title())

        if category_certs:
            found_certifications[category] = list(set(category_certs))

    return found_certifications


def extract_projects(text):
    """Extract project information from resume (uses cached indicators)"""
    text_lower = text.lower()
    project_count = sum(text_lower.count(indicator) for indicator in PROJECT_INDICATORS)
    estimated_projects = min(project_count // 2, 10)

    return {
        'estimated_count': estimated_projects,
        'has_projects': estimated_projects > 0
    }


def calculate_keyword_density(text_lower, words_count, keywords):
    """Calculate keyword density and relevance (optimized: accepts pre-computed values)"""
    keyword_scores = {}
    for keyword in keywords:
        count = text_lower.count(keyword.lower())
        if count > 0:
            density = (count / words_count) * 100 if words_count > 0 else 0
            keyword_scores[keyword] = {'count': count, 'density': round(density, 2)}
    return keyword_scores


def assess_resume_quality(text, text_lower, contact_info):
    """Assess overall resume quality (optimized: accepts pre-lowered text)"""
    score = 0
    feedback = []
    word_count = len(text.split())

    # Length check
    if 400 <= word_count <= 1500:
        score += 20
        feedback.append("Good resume length")
    elif word_count < 400:
        score += 10
        feedback.append("Resume is quite short, consider adding more details")
    else:
        score += 15
        feedback.append("Resume is lengthy, consider being more concise")

    # Contact info
    contact_score = sum([
        10 if contact_info['email'] else 0,
        10 if contact_info['phone'] else 0,
        5 if contact_info['linkedin'] else 0,
        5 if contact_info['github'] else 0,
    ])
    score += contact_score
    feedback.append("Complete contact information" if contact_score >= 20 else "Missing some contact information")

    # Key sections
    for section_name, keywords in SECTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            score += 10
            feedback.append(f"{section_name.title()} section present")

    # Formatting
    if '\n' in text and text.count('\n') > 5:
        score += 10
        feedback.append("Well-structured with line breaks")

    # Professional keywords
    if sum(1 for kw in PROFESSIONAL_KEYWORDS if kw in text_lower) >= 3:
        score += 10
        feedback.append("Uses strong action verbs")

    return {'score': min(score, 100), 'feedback': feedback, 'word_count': word_count}


# ============================================================
# FILE EXTRACTION
# ============================================================
def extract_text_from_file(filepath):
    """Extract text from uploaded resume file"""
    try:
        file_ext = filepath.rsplit('.', 1)[1].lower()

        if file_ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif file_ext == 'pdf':
            try:
                import PyPDF2
                with open(filepath, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = '\n'.join(page.extract_text() or '' for page in pdf_reader.pages)
            except ImportError:
                return "PDF parsing requires PyPDF2. Install with: pip install PyPDF2"
        elif file_ext in ('doc', 'docx'):
            try:
                import docx
                doc = docx.Document(filepath)
                text = '\n'.join(para.text for para in doc.paragraphs)
            except ImportError:
                return "DOCX parsing requires python-docx. Install with: pip install python-docx"
        else:
            return "Unsupported file format"

        # Truncate oversized resumes for faster processing
        processed = preprocess_text(text)
        if len(processed) > MAX_RESUME_LENGTH:
            processed = processed[:MAX_RESUME_LENGTH]

        return processed
    except Exception as e:
        return f"Error extracting text: {str(e)}"


# ============================================================
# SKILL EXTRACTION (optimized — single pass over text)
# ============================================================
def extract_keywords_from_jd(job_description):
    """Extract requirements from job description with fuzzy matching"""
    jd_lower = job_description.lower()

    # Find skills (uses cached constant list)
    found_skills = [
        skill for skill in SKILLS_FLAT
        if fuzzy_skill_match(skill, jd_lower, threshold=0.85)[0]
    ]

    # Experience
    experience_years = extract_experience_years(jd_lower)

    # Education
    requires_education = any(kw in jd_lower for kw in EDUCATION_KEYWORDS)

    return {
        'required_skills': found_skills[:10],
        'preferred_skills': found_skills[10:],
        'experience_years': experience_years,
        'requires_education': requires_education
    }


def extract_all_skills(text_lower):
    """Extract all skills from resume text (optimized: accepts pre-lowered text)"""
    found_skills = {}
    for category, skills in SKILLS_BY_CATEGORY.items():
        category_skills = [
            skill for skill in skills
            if fuzzy_skill_match(skill, text_lower, threshold=0.85)[0]
        ]
        if category_skills:
            found_skills[category] = category_skills
    return found_skills


# ============================================================
# ANALYSIS FUNCTIONS (optimized — pre-compute shared values)
# ============================================================
def analyze_resume_with_jd(resume_text, job_description):
    """Analyze resume against job description with enhanced features"""
    jd_requirements = extract_keywords_from_jd(job_description)
    text_lower = resume_text.lower()

    # Pre-compute shared extractions
    contact_info = extract_contact_info(resume_text)
    certifications = extract_certifications(resume_text)
    projects = extract_projects(resume_text)
    quality = assess_resume_quality(resume_text, text_lower, contact_info)

    # Find matching skills
    found_required = [s for s in jd_requirements['required_skills'] if fuzzy_skill_match(s, text_lower)[0]]
    found_preferred = [s for s in jd_requirements['preferred_skills'] if fuzzy_skill_match(s, text_lower)[0]]
    missing_skills = [s for s in jd_requirements['required_skills'] if s not in found_required]

    # Education
    has_education = any(kw in text_lower for kw in EDUCATION_KEYWORDS)

    # Experience
    experience_years = extract_experience_years(text_lower)

    # Keyword density  
    words_count = len(resume_text.split())
    all_jd_skills = jd_requirements['required_skills'] + jd_requirements['preferred_skills']
    keyword_density = calculate_keyword_density(text_lower, words_count, all_jd_skills)

    # Calculate match percentage
    total_required = len(jd_requirements['required_skills'])
    matched_required = len(found_required)

    if total_required > 0:
        skill_match_percentage = (matched_required / total_required) * 60
    else:
        skill_match_percentage = 30

    # Experience match (20 points)
    if jd_requirements['experience_years'] > 0:
        req_exp = jd_requirements['experience_years']
        if experience_years >= req_exp:
            exp_match = 20
        elif experience_years >= req_exp * 0.7:
            exp_match = 15
        elif experience_years >= req_exp * 0.5:
            exp_match = 10
        elif experience_years > 0:
            exp_match = 5
        else:
            exp_match = 0
    else:
        if experience_years >= 5:
            exp_match = 15
        elif experience_years >= 2:
            exp_match = 10
        elif experience_years > 0:
            exp_match = 5
        else:
            exp_match = 0

    # Education match (10 points)
    if jd_requirements['requires_education']:
        edu_match = 10 if has_education else 0
    else:
        edu_match = 5 if has_education else 0

    # Quality bonus (10 points)
    quality_bonus = (quality['score'] / 100) * 10

    # Total (capped at 95%)
    match_percentage = min(95, skill_match_percentage + exp_match + edu_match + quality_bonus)

    # Recommendation
    if match_percentage >= 80:
        recommendation, recommendation_class = "Highly Recommended", "excellent"
    elif match_percentage >= 65:
        recommendation, recommendation_class = "Recommended", "good"
    elif match_percentage >= 50:
        recommendation, recommendation_class = "Maybe", "average"
    else:
        recommendation, recommendation_class = "Not Recommended", "poor"

    return {
        'mode': 'job_matching',
        'match_percentage': round(match_percentage, 1),
        'required_skills_found': found_required,
        'preferred_skills_found': found_preferred,
        'missing_skills': missing_skills,
        'has_education': has_education,
        'experience_years': experience_years,
        'required_experience': jd_requirements['experience_years'],
        'recommendation': recommendation,
        'recommendation_class': recommendation_class,
        'job_description_summary': job_description[:200] + '...' if len(job_description) > 200 else job_description,
        'contact_info': contact_info,
        'certifications': certifications,
        'projects': projects,
        'quality': quality,
        'keyword_density': keyword_density
    }


def analyze_resume_standalone(resume_text):
    """Analyze resume without job description with enhanced features"""
    text_lower = resume_text.lower()

    # Pre-compute shared extractions
    contact_info = extract_contact_info(resume_text)
    certifications = extract_certifications(resume_text)
    projects = extract_projects(resume_text)
    quality = assess_resume_quality(resume_text, text_lower, contact_info)

    # Extract skills
    skills_by_category = extract_all_skills(text_lower)
    all_skills_found = [skill for skills in skills_by_category.values() for skill in skills]

    # Education & Experience
    has_education = any(kw in text_lower for kw in EDUCATION_KEYWORDS)
    experience_years = extract_experience_years(text_lower)

    # Generate insights
    strengths = []
    suggestions = []

    total_skills = len(all_skills_found)
    if total_skills >= 10:
        strengths.append("Strong technical skill set with diverse expertise")
    elif total_skills >= 5:
        strengths.append("Good range of technical skills")
    else:
        suggestions.append("Consider highlighting more technical skills")

    if experience_years >= 5:
        strengths.append(f"Extensive experience ({experience_years}+ years)")
    elif experience_years >= 2:
        strengths.append(f"Solid professional experience ({experience_years} years)")
    else:
        suggestions.append("Highlight projects and achievements to compensate for limited experience")

    if has_education:
        strengths.append("Educational qualifications mentioned")
    else:
        suggestions.append("Include educational background if applicable")

    if certifications:
        cert_count = sum(len(certs) for certs in certifications.values())
        strengths.append(f"Has {cert_count} certification(s)")

    if projects['has_projects']:
        strengths.append(f"Mentions {projects['estimated_count']} project(s)")

    if contact_info['email'] and contact_info['phone']:
        strengths.append("Complete contact information provided")
    else:
        suggestions.append("Add missing contact information (email/phone)")

    if quality['score'] >= 70:
        strengths.append("Well-formatted and structured resume")
    elif quality['score'] < 50:
        suggestions.append("Improve resume formatting and structure")

    if 'Programming Languages' in skills_by_category and len(skills_by_category['Programming Languages']) >= 3:
        strengths.append("Proficient in multiple programming languages")

    if not strengths:
        strengths.append("Resume uploaded successfully")
    if not suggestions:
        suggestions.append("Resume looks comprehensive")

    # Score (capped at 95%)
    score = min(95,
                total_skills * 4 +
                experience_years * 3 +
                (15 if has_education else 0) +
                (quality['score'] * 0.2) +
                (projects['estimated_count'] * 2))

    return {
        'mode': 'standalone',
        'score': round(score, 1),
        'skills_by_category': skills_by_category,
        'total_skills': total_skills,
        'has_education': has_education,
        'experience_years': experience_years,
        'strengths': strengths,
        'suggestions': suggestions,
        'recommendation': 'Profile Analyzed',
        'recommendation_class': 'good',
        'contact_info': contact_info,
        'certifications': certifications,
        'projects': projects,
        'quality': quality
    }


def apply_filters(analysis, filters):
    """Apply filters to analysis results"""
    # Minimum experience
    if filters.get('min_experience'):
        if analysis.get('experience_years', 0) < int(filters['min_experience']):
            return False

    # Required skills
    if filters.get('required_skills'):
        required = [s.strip().lower() for s in filters['required_skills'].split(',')]
        if analysis['mode'] == 'job_matching':
            found = [s.lower() for s in analysis['required_skills_found']]
        else:
            found = [s.lower() for skills in analysis['skills_by_category'].values() for s in skills]

        if not all(req in found for req in required):
            return False

    # Minimum score
    if filters.get('min_score'):
        score = analysis.get('match_percentage') or analysis.get('score', 0)
        if score < float(filters['min_score']):
            return False

    # Education
    if filters.get('requires_education') == 'true':
        if not analysis.get('has_education', False):
            return False

    return True


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def landing():
    """Render the landing page"""
    return render_template('landing.html')


@app.route('/upload', methods=['GET'])
def upload_page():
    """Render the upload page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload and screening with ML integration"""
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400

    # Secure the filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Extract text from resume
        resume_text = extract_text_from_file(filepath)

        # Get job description if provided
        job_description = request.form.get('job_description', '').strip()

        # Analyze based on whether JD is provided (rule-based)
        if job_description:
            analysis_results = analyze_resume_with_jd(resume_text, job_description)
        else:
            analysis_results = analyze_resume_standalone(resume_text)

        # Try ML prediction if model is available
        if ML_ENABLED and model_manager.has_active_model():
            try:
                prediction, confidence, probabilities = model_manager.active_model.predict(
                    resume_text, analysis_results
                )

                if confidence >= ml_config.CONFIDENCE_THRESHOLD:
                    recommendation, rec_class = model_manager.active_model.get_recommendation_from_prediction(
                        prediction, confidence
                    )
                    analysis_results['recommendation'] = recommendation
                    analysis_results['recommendation_class'] = rec_class
                    analysis_results['ml_confidence'] = confidence
                    analysis_results['used_ml'] = True
                else:
                    analysis_results['used_ml'] = False
                    analysis_results['ml_confidence'] = confidence
            except Exception as e:
                print(f"ML prediction error: {e}")
                analysis_results['used_ml'] = False
        else:
            analysis_results['used_ml'] = False

        # Save to database if ML is enabled
        analysis_id = None
        if ML_ENABLED:
            try:
                resume_id = db.save_resume(
                    filename=filename,
                    resume_text=resume_text,
                    word_count=len(resume_text.split()),
                    has_education=analysis_results.get('has_education', False),
                    experience_years=analysis_results.get('experience_years', 0)
                )

                jd_id = None
                if job_description:
                    jd_requirements = extract_keywords_from_jd(job_description)
                    jd_id = db.save_job_description(
                        description_text=job_description,
                        required_skills=jd_requirements['required_skills'],
                        experience_years=jd_requirements['experience_years']
                    )

                analysis_id = db.save_analysis_result(
                    resume_id=resume_id,
                    analysis=analysis_results,
                    job_description_id=jd_id
                )
            except Exception as e:
                print(f"Database save error: {e}")

        # Store results in session
        session['last_result'] = {
            'filename': filename,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': analysis_results,
            'resume_text': resume_text[:500],
            'analysis_id': analysis_id
        }

        return jsonify({
            'success': True,
            'message': 'Resume analyzed successfully',
            'filename': filename,
            'redirect': '/results'
        }), 200

    finally:
        # Clean up uploaded file after processing to save disk space
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass


@app.route('/results')
def results():
    """Display screening results"""
    result = session.get('last_result')
    if not result:
        return render_template('no_results.html')
    return render_template('results.html', result=result)


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback on analysis"""
    if not ML_ENABLED:
        return jsonify({'error': 'ML system not enabled'}), 400

    data = request.json
    analysis_id = data.get('analysis_id')
    feedback_type = data.get('feedback_type', 'recommendation')
    feedback_value = data.get('feedback_value')
    comments = data.get('comments', '')

    if not analysis_id or feedback_value is None:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        feedback_id = db.save_feedback(
            analysis_id=analysis_id,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            comments=comments
        )

        if training_pipeline.auto_retrain_check():
            try:
                trigger_training()
            except Exception as e:
                print(f"Auto-retrain error: {e}")

        return jsonify({
            'success': True,
            'feedback_id': feedback_id,
            'message': 'Feedback submitted successfully'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train-model', methods=['POST'])
def train_model():
    """Manually trigger model training"""
    if not ML_ENABLED:
        return jsonify({'error': 'ML system not enabled'}), 400

    model_type = request.json.get('model_type', 'random_forest')
    result = trigger_training(model_type)

    return jsonify(result), 200 if result['success'] else 400


@app.route('/model-status')
def model_status():
    """Get ML model status and training readiness"""
    if not ML_ENABLED:
        return jsonify({'ml_enabled': False, 'message': 'ML system not available'}), 200

    status = get_training_status()
    status['ml_enabled'] = True
    return jsonify(status), 200


@app.route('/training-data-stats')
def training_data_stats():
    """Get training data statistics"""
    if not ML_ENABLED:
        return jsonify({'error': 'ML system not enabled'}), 400

    stats = db.get_statistics()
    return jsonify(stats), 200


@app.route('/export-csv')
def export_csv():
    """Export results to CSV"""
    result = session.get('last_result')
    if not result:
        return jsonify({'error': 'No results to export'}), 400

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(['Field', 'Value'])
    writer.writerow(['Filename', result['filename']])
    writer.writerow(['Analysis Time', result['upload_time']])
    writer.writerow(['Mode', result['analysis']['mode']])

    analysis = result['analysis']

    if analysis['mode'] == 'job_matching':
        writer.writerow(['Match Percentage', f"{analysis['match_percentage']}%"])
        writer.writerow(['Recommendation', analysis['recommendation']])
        writer.writerow(['Required Skills Found', ', '.join(analysis['required_skills_found'])])
        writer.writerow(['Missing Skills', ', '.join(analysis['missing_skills'])])
        writer.writerow(['Experience Years', analysis['experience_years']])
    else:
        writer.writerow(['Score', analysis['score']])
        writer.writerow(['Total Skills', analysis['total_skills']])
        writer.writerow(['Experience Years', analysis['experience_years']])
        writer.writerow(['Strengths', ' | '.join(analysis['strengths'])])
        writer.writerow(['Suggestions', ' | '.join(analysis['suggestions'])])

    contact = analysis.get('contact_info', {})
    writer.writerow(['Email', contact.get('email', 'N/A')])
    writer.writerow(['Phone', contact.get('phone', 'N/A')])
    writer.writerow(['LinkedIn', contact.get('linkedin', 'N/A')])
    writer.writerow(['GitHub', contact.get('github', 'N/A')])

    quality = analysis.get('quality', {})
    writer.writerow(['Resume Quality Score', quality.get('score', 'N/A')])

    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename=resume_analysis_{result["filename"]}.csv'
    return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
