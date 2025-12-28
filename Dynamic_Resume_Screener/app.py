# Dynamic Resume Screener Application - Enhanced Version with ML
# Main Flask application file with advanced NLP, filtering, and ML capabilities

from flask import Flask, render_template, request, jsonify, session, send_file, make_response
import os
import re
from datetime import datetime
import secrets
from difflib import SequenceMatcher
from collections import Counter
import json
import csv
import io

# ML imports - Try improved models first, fallback to original
try:
    from database import db
    import config as ml_config
    
    # Try to import improved models
    try:
        from ml_models_improved import improved_model_manager as model_manager
        from training_pipeline_improved import improved_training_pipeline as training_pipeline
        from training_pipeline_improved import trigger_improved_training as trigger_training
        from training_pipeline_improved import get_improved_training_status as get_training_status
        print("✓ Using IMPROVED ensemble models (XGBoost + LightGBM + RF + GB)")
        ML_ENABLED = True
        USING_IMPROVED_MODELS = True
    except ImportError as e:
        # Fallback to original models
        print(f"Improved models not available ({e}), using original models")
        from ml_models import model_manager
        from training_pipeline import training_pipeline, trigger_training, get_training_status
        ML_ENABLED = True
        USING_IMPROVED_MODELS = False
except ImportError as e:
    print(f"ML modules not available: {e}")
    ML_ENABLED = False
    USING_IMPROVED_MODELS = False

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
UPLOAD_FOLDER = 'resumes'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure the resumes folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ML system
if ML_ENABLED:
    try:
        # Try to load active model
        model_manager.load_active_model()
        print("ML system initialized successfully")
        if model_manager.has_active_model():
            print("Active ML model loaded")
        else:
            print("No active model found - will use rule-based system")
    except Exception as e:
        print(f"Error initializing ML system: {e}")
        ML_ENABLED = False



def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s@.+\-():/]', ' ', text)
    
    # Normalize common variations
    text = text.replace('–', '-').replace('—', '-')
    
    return text.strip()


def fuzzy_skill_match(skill, text, threshold=0.85):
    """
    Match skills with fuzzy matching to catch variations
    Returns: (matched, confidence_score)
    """
    text_lower = text.lower()
    skill_lower = skill.lower()
    
    # Exact match
    if skill_lower in text_lower:
        return True, 1.0
    
    # Try common variations
    variations = [
        skill_lower.replace('  .', ''),
        skill_lower.replace(' ', ''),
        skill_lower.replace('-', ''),
        skill_lower.replace('js', 'javascript'),
        skill_lower.replace('javascript', 'js'),
    ]
    
    for variation in variations:
        if variation in text_lower:
            return True, 0.95
    
    # Fuzzy matching for close matches
    words = text_lower.split()
    for word in words:
        if len(word) >= 3 and len(skill_lower) >= 3:
            ratio = SequenceMatcher(None, skill_lower, word).ratio()
            if ratio >= threshold:
                return True, ratio
    
    return False, 0.0


def extract_contact_info(text):
    """Extract contact information from resume"""
    contact_info = {
        'email': None,
        'phone': None,
        'linkedin': None,
        'github': None
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contact_info['email'] = emails[0]
    
    # Extract phone number (various formats)
    phone_patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{10}',
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            contact_info['phone'] = phones[0]
            break
    
    # Extract LinkedIn
    linkedin_pattern = r'linkedin\.com/in/[\w-]+'
    linkedin_matches = re.findall(linkedin_pattern, text.lower())
    if linkedin_matches:
        contact_info['linkedin'] = f"https://{linkedin_matches[0]}"
    
    # Extract GitHub
    github_pattern = r'github\.com/[\w-]+'
    github_matches = re.findall(github_pattern, text.lower())
    if github_matches:
        contact_info['github'] = f"https://{github_matches[0]}"
    
    return contact_info


def extract_certifications(text):
    """Extract certifications from resume"""
    text_lower = text.lower()
    
    certification_keywords = {
        'Cloud': ['aws certified', 'azure certified', 'google cloud certified', 'gcp certified'],
        'Programming': ['oracle certified', 'microsoft certified', 'java certified', 'python certified'],
        'Project Management': ['pmp', 'prince2', 'scrum master', 'csm', 'psm'],
        'Security': ['cissp', 'ceh', 'security+', 'comptia security'],
        'Data': ['data science certified', 'machine learning certified', 'tensorflow certified'],
        'Other': ['certified', 'certification']
    }
    
    found_certifications = {}
    for category, certs in certification_keywords.items():
        category_certs = []
        for cert in certs:
            if cert in text_lower:
                # Try to extract the full certification name
                pattern = rf'([A-Z][A-Za-z\s]*{cert}[A-Za-z\s]*)'
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    category_certs.append(matches[0].strip())
                else:
                    category_certs.append(cert.title())
        
        if category_certs:
            found_certifications[category] = list(set(category_certs))
    
    return found_certifications


def extract_projects(text):
    """Extract project information from resume"""
    project_indicators = [
        'project', 'developed', 'built', 'created', 'implemented',
        'designed', 'worked on', 'contributed to'
    ]
    
    text_lower = text.lower()
    project_count = 0
    
    for indicator in project_indicators:
        project_count += text_lower.count(indicator)
    
    # Estimate number of unique projects (rough heuristic)
    estimated_projects = min(project_count // 2, 10)
    
    return {
        'estimated_count': estimated_projects,
        'has_projects': estimated_projects > 0
    }


def calculate_keyword_density(text, keywords):
    """Calculate keyword density and relevance"""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words)
    
    keyword_scores = {}
    for keyword in keywords:
        count = text_lower.count(keyword.lower())
        if count > 0:
            density = (count / total_words) * 100 if total_words > 0 else 0
            keyword_scores[keyword] = {
                'count': count,
                'density': round(density, 2)
            }
    
    return keyword_scores


def assess_resume_quality(text, contact_info):
    """Assess overall resume quality"""
    score = 0
    feedback = []
    
    # Check length (optimal: 400-1500 words)
    word_count = len(text.split())
    if 400 <= word_count <= 1500:
        score += 20
        feedback.append("Good resume length")
    elif word_count < 400:
        score += 10
        feedback.append("Resume is quite short, consider adding more details")
    else:
        score += 15
        feedback.append("Resume is lengthy, consider being more concise")
    
    # Check contact information
    contact_score = 0
    if contact_info['email']:
        contact_score += 10
    if contact_info['phone']:
        contact_score += 10
    if contact_info['linkedin']:
        contact_score += 5
    if contact_info['github']:
        contact_score += 5
    
    score += contact_score
    if contact_score >= 20:
        feedback.append("Complete contact information")
    else:
        feedback.append("Missing some contact information")
    
    # Check for key sections
    text_lower = text.lower()
    sections = {
        'experience': ['experience', 'work history', 'employment'],
        'education': ['education', 'academic', 'degree'],
        'skills': ['skills', 'technical skills', 'competencies'],
    }
    
    section_score = 0
    for section_name, keywords in sections.items():
        if any(keyword in text_lower for keyword in keywords):
            section_score += 10
            feedback.append(f"{section_name.title()} section present")
    
    score += section_score
    
    # Check formatting indicators
    if '\n' in text and text.count('\n') > 5:
        score += 10
        feedback.append("Well-structured with line breaks")
    
    # Bonus for professional keywords
    professional_keywords = ['achieved', 'led', 'managed', 'improved', 'increased', 'developed']
    professional_count = sum(1 for keyword in professional_keywords if keyword in text_lower)
    if professional_count >= 3:
        score += 10
        feedback.append("Uses strong action verbs")
    
    return {
        'score': min(score, 100),
        'feedback': feedback,
        'word_count': word_count
    }


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
                    text = ''
                    for page in pdf_reader.pages:
                        text += page.extract_text() + '\n'
            except ImportError:
                return "PDF parsing requires PyPDF2. Install with: pip install PyPDF2"
        elif file_ext in ['doc', 'docx']:
            try:
                import docx
                doc = docx.Document(filepath)
                text = '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                return "DOCX parsing requires python-docx. Install with: pip install python-docx"
        else:
            return "Unsupported file format"
        
        return preprocess_text(text)
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def extract_keywords_from_jd(job_description):
    """Extract requirements from job description with fuzzy matching"""
    jd_lower = job_description.lower()
    
    # Comprehensive skill list
    all_skills = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go',
        'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'asp.net', 'next.js',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'nosql', 'dynamodb',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform',
        'machine learning', 'deep learning', 'ai', 'data science', 'nlp', 'computer vision',
        'html', 'css', 'sass', 'bootstrap', 'tailwind',
        'rest api', 'graphql', 'microservices', 'agile', 'scrum', 'devops'
    ]
    
    found_skills = []
    for skill in all_skills:
        matched, confidence = fuzzy_skill_match(skill, job_description, threshold=0.85)
        if matched:
            found_skills.append(skill)
    
    # Extract experience requirements
    experience_years = 0
    exp_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*[:–-]\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*in',
    ]
    for pattern in exp_patterns:
        matches = re.findall(pattern, jd_lower)
        if matches:
            experience_years = max([int(m) for m in matches])
            break
    
    # Check for education requirements
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'b.tech', 'm.tech', 'mca', 'bca', 'b.s', 'm.s']
    requires_education = any(keyword in jd_lower for keyword in education_keywords)
    
    return {
        'required_skills': found_skills[:10] if len(found_skills) > 10 else found_skills,
        'preferred_skills': found_skills[10:] if len(found_skills) > 10 else [],
        'experience_years': experience_years,
        'requires_education': requires_education
    }


def extract_all_skills(text):
    """Extract all skills from resume text with fuzzy matching"""
    # Comprehensive skill list
    all_skills = {
        'Programming Languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r'],
        'Web Frameworks': ['react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring boot', 'asp.net', 'laravel', 'next.js', 'svelte'],
        'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'nosql', 'dynamodb', 'cassandra', 'elasticsearch'],
        'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform', 'ansible'],
        'AI & Data Science': ['machine learning', 'deep learning', 'ai', 'data science', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'pandas', 'numpy'],
        'Frontend': ['html', 'css', 'sass', 'bootstrap', 'tailwind', 'jquery', 'webpack', 'responsive design'],
        'Other': ['rest api', 'graphql', 'microservices', 'agile', 'scrum', 'devops', 'testing', 'unit testing']
    }
    
    found_skills = {}
    for category, skills in all_skills.items():
        category_skills = []
        for skill in skills:
            matched, confidence = fuzzy_skill_match(skill, text, threshold=0.85)
            if matched and skill not in category_skills:
                category_skills.append(skill)
        if category_skills:
            found_skills[category] = category_skills
    
    return found_skills


def analyze_resume_with_jd(resume_text, job_description):
    """Analyze resume against job description with enhanced features"""
    jd_requirements = extract_keywords_from_jd(job_description)
    text_lower = resume_text.lower()
    
    # Extract contact info
    contact_info = extract_contact_info(resume_text)
    
    # Extract certifications
    certifications = extract_certifications(resume_text)
    
    # Extract projects
    projects = extract_projects(resume_text)
    
    # Assess resume quality
    quality = assess_resume_quality(resume_text, contact_info)
    
    # Find matching skills with fuzzy matching
    found_required_skills = []
    for skill in jd_requirements['required_skills']:
        matched, confidence = fuzzy_skill_match(skill, resume_text, threshold=0.85)
        if matched:
            found_required_skills.append(skill)
    
    found_preferred_skills = []
    for skill in jd_requirements['preferred_skills']:
        matched, confidence = fuzzy_skill_match(skill, resume_text, threshold=0.85)
        if matched:
            found_preferred_skills.append(skill)
    
    # Find missing skills
    missing_skills = [skill for skill in jd_requirements['required_skills'] if skill not in found_required_skills]
    
    # Check education
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'b.tech', 'm.tech', 'mca', 'bca']
    has_education = any(keyword in text_lower for keyword in education_keywords)
    
    # Extract experience
    experience_years = 0
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*[:–-]\s*(\d+)\+?\s*years?',
    ]
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            experience_years = max([int(m) for m in matches])
            break
    
    # Calculate keyword density
    all_jd_skills = jd_requirements['required_skills'] + jd_requirements['preferred_skills']
    keyword_density = calculate_keyword_density(resume_text, all_jd_skills)
    
    # Calculate match percentage
    total_required = len(jd_requirements['required_skills'])
    matched_required = len(found_required_skills)
    
    if total_required > 0:
        skill_match_percentage = (matched_required / total_required) * 60
    else:
        # If no skills specified in JD, give partial credit based on resume skills
        skill_match_percentage = 30  # Changed from 0 to 30 for having skills
    
    # Experience match (20 points)
    exp_match = 0
    if jd_requirements['experience_years'] > 0:
        if experience_years >= jd_requirements['experience_years']:
            exp_match = 20
        elif experience_years >= jd_requirements['experience_years'] * 0.7:
            exp_match = 15
        elif experience_years >= jd_requirements['experience_years'] * 0.5:
            exp_match = 10
        elif experience_years > 0:
            exp_match = 5  # Some experience is better than none
    else:
        # If no experience requirement specified, give partial credit
        if experience_years >= 5:
            exp_match = 15
        elif experience_years >= 2:
            exp_match = 10
        elif experience_years > 0:
            exp_match = 5
        else:
            exp_match = 0  # Changed from 20 to 0 when no experience
    
    # Education match (10 points)
    if jd_requirements['requires_education']:
        edu_match = 10 if has_education else 0
    else:
        # If no education requirement, give partial credit for having it
        edu_match = 5 if has_education else 0  # Changed from 10 to 5
    
    # Quality bonus (10 points)
    quality_bonus = (quality['score'] / 100) * 10
    
    # Calculate total and cap at 95% (no resume is perfect)
    match_percentage = min(95, skill_match_percentage + exp_match + edu_match + quality_bonus)
    
    # Determine recommendation
    if match_percentage >= 80:
        recommendation = "Highly Recommended"
        recommendation_class = "excellent"
    elif match_percentage >= 65:
        recommendation = "Recommended"
        recommendation_class = "good"
    elif match_percentage >= 50:
        recommendation = "Maybe"
        recommendation_class = "average"
    else:
        recommendation = "Not Recommended"
        recommendation_class = "poor"
    
    return {
        'mode': 'job_matching',
        'match_percentage': round(match_percentage, 1),
        'required_skills_found': found_required_skills,
        'preferred_skills_found': found_preferred_skills,
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
    
    # Extract contact info
    contact_info = extract_contact_info(resume_text)
    
    # Extract certifications
    certifications = extract_certifications(resume_text)
    
    # Extract projects
    projects = extract_projects(resume_text)
    
    # Assess resume quality
    quality = assess_resume_quality(resume_text, contact_info)
    
    # Extract all skills
    skills_by_category = extract_all_skills(resume_text)
    
    # Flatten skills for total count
    all_skills_found = []
    for category_skills in skills_by_category.values():
        all_skills_found.extend(category_skills)
    
    # Extract education
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'b.tech', 'm.tech', 'mca', 'bca']
    has_education = any(keyword in text_lower for keyword in education_keywords)
    
    # Extract experience
    experience_years = 0
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*[:–-]\s*(\d+)\+?\s*years?',
    ]
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            experience_years = max([int(m) for m in matches])
            break
    
    # Generate insights
    strengths = []
    suggestions = []
    
    if len(all_skills_found) >= 10:
        strengths.append("Strong technical skill set with diverse expertise")
    elif len(all_skills_found) >= 5:
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
        strengths.append(f"Has {sum(len(certs) for certs in certifications.values())} certification(s)")
    
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
    
    # Calculate overall score (cap at 95% for realism)
    score = min(95, 
                len(all_skills_found) * 4 + 
                experience_years * 3 + 
                (15 if has_education else 0) +
                (quality['score'] * 0.2) +
                (projects['estimated_count'] * 2))
    
    return {
        'mode': 'standalone',
        'score': round(score, 1),
        'skills_by_category': skills_by_category,
        'total_skills': len(all_skills_found),
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
    passes = True
    
    # Minimum experience filter
    if 'min_experience' in filters and filters['min_experience']:
        if analysis.get('experience_years', 0) < int(filters['min_experience']):
            passes = False
    
    # Required skills filter
    if 'required_skills' in filters and filters['required_skills']:
        required = [s.strip().lower() for s in filters['required_skills'].split(',')]
        if analysis['mode'] == 'job_matching':
            found = [s.lower() for s in analysis['required_skills_found']]
        else:
            all_skills = []
            for skills in analysis['skills_by_category'].values():
                all_skills.extend([s.lower() for s in skills])
            found = all_skills
        
        if not all(req in found for req in required):
            passes = False
    
    # Minimum score filter
    if 'min_score' in filters and filters['min_score']:
        score = analysis.get('match_percentage') or analysis.get('score', 0)
        if score < float(filters['min_score']):
            passes = False
    
    # Education filter
    if 'requires_education' in filters and filters['requires_education'] == 'true':
        if not analysis.get('has_education', False):
            passes = False
    
    return passes


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
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
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
        ml_prediction = None
        if ML_ENABLED and model_manager.has_active_model():
            try:
                prediction, confidence, probabilities = model_manager.active_model.predict(
                    resume_text, analysis_results
                )
                
                # Use ML prediction if confidence is high enough
                if confidence >= ml_config.CONFIDENCE_THRESHOLD:
                    recommendation, rec_class = model_manager.active_model.get_recommendation_from_prediction(
                        prediction, confidence
                    )
                    
                    ml_prediction = {
                        'recommendation': recommendation,
                        'recommendation_class': rec_class,
                        'confidence': confidence,
                        'probabilities': probabilities
                    }
                    
                    # Override rule-based recommendation with ML
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
                # Save resume
                resume_id = db.save_resume(
                    filename=filename,
                    resume_text=resume_text,
                    word_count=len(resume_text.split()),
                    has_education=analysis_results.get('has_education', False),
                    experience_years=analysis_results.get('experience_years', 0)
                )
                
                # Save job description if provided
                jd_id = None
                if job_description:
                    jd_requirements = extract_keywords_from_jd(job_description)
                    jd_id = db.save_job_description(
                        description_text=job_description,
                        required_skills=jd_requirements['required_skills'],
                        experience_years=jd_requirements['experience_years']
                    )
                
                # Save analysis result
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
            'resume_text': resume_text[:500],  # Store snippet for export
            'analysis_id': analysis_id
        }
        
        return jsonify({
            'success': True,
            'message': 'Resume analyzed successfully',
            'filename': filename,
            'redirect': '/results'
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400


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
    feedback_type = data.get('feedback_type', 'recommendation')  # recommendation, quality, etc.
    feedback_value = data.get('feedback_value')  # 1 = positive, 0 = neutral, -1 = negative
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
        
        # Check if auto-retrain should be triggered
        if training_pipeline.auto_retrain_check():
            # Trigger training in background (simplified - in production use celery/background task)
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
    
    if result['success']:
        return jsonify(result), 200
    else:
        return jsonify(result), 400


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
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['Field', 'Value'])
    
    # Write basic info
    writer.writerow(['Filename', result['filename']])
    writer.writerow(['Analysis Time', result['upload_time']])
    writer.writerow(['Mode', result['analysis']['mode']])
    
    # Write analysis results
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
    
    # Contact info
    contact = analysis.get('contact_info', {})
    writer.writerow(['Email', contact.get('email', 'N/A')])
    writer.writerow(['Phone', contact.get('phone', 'N/A')])
    writer.writerow(['LinkedIn', contact.get('linkedin', 'N/A')])
    writer.writerow(['GitHub', contact.get('github', 'N/A')])
    
    # Quality score
    quality = analysis.get('quality', {})
    writer.writerow(['Resume Quality Score', quality.get('score', 'N/A')])
    
    # Create response
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename=resume_analysis_{result["filename"]}.csv'
    
    return response


if __name__ == '__main__':
    app.run(debug=True)
