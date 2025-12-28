# Improved ML Models for Resume Screening with Ensemble Methods
# Enhanced with XGBoost, LightGBM, and advanced techniques

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# New imports for improved models
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using TF-IDF only.")

import config


class ImprovedFeatureExtractor:
    """Enhanced feature extractor with more sophisticated features"""
    
    def __init__(self):
        """Initialize feature extractors"""
        self.tfidf = TfidfVectorizer(
            max_features=config.MAX_FEATURES_TFIDF,
            stop_words='english',
            ngram_range=(1, 3),  # Increased to trigrams
            min_df=2,  # Ignore very rare terms
            max_df=0.95  # Ignore very common terms
        )
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(config.EMBEDDING_MODEL)
            except:
                self.sentence_model = None
                print("Warning: Could not load sentence transformer model")
        else:
            self.sentence_model = None
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_metadata_features(self, resume_text: str, analysis: Dict) -> np.ndarray:
        """Extract enhanced metadata features"""
        features = []
        
        # Text length features
        features.append(len(resume_text))
        features.append(len(resume_text.split()))
        features.append(len(set(resume_text.split())))  # Unique words
        
        # Lexical diversity
        words = resume_text.split()
        if len(words) > 0:
            features.append(len(set(words)) / len(words))  # Type-token ratio
        else:
            features.append(0)
        
        # Experience
        exp_years = analysis.get('experience_years', 0)
        features.append(exp_years)
        features.append(min(exp_years / 10.0, 1.0))  # Normalized experience
        
        # Education
        features.append(1 if analysis.get('has_education', False) else 0)
        
        # Skills count
        if analysis.get('mode') == 'job_matching':
            required_skills = len(analysis.get('required_skills_found', []))
            missing_skills = len(analysis.get('missing_skills', []))
            preferred_skills = len(analysis.get('preferred_skills_found', []))
            
            features.append(required_skills)
            features.append(missing_skills)
            features.append(preferred_skills)
            
            # Skill match ratio
            total_required = required_skills + missing_skills
            if total_required > 0:
                features.append(required_skills / total_required)
            else:
                features.append(0)
        else:
            total_skills = analysis.get('total_skills', 0)
            features.append(total_skills)
            features.append(0)  # missing skills
            features.append(0)  # preferred skills
            features.append(min(total_skills / 10.0, 1.0))  # Normalized skills
        
        # Quality score
        quality = analysis.get('quality', {})
        quality_score = quality.get('score', 0)
        features.append(quality_score)
        features.append(quality_score / 100.0)  # Normalized
        features.append(quality.get('word_count', 0))
        
        # Contact info completeness
        contact = analysis.get('contact_info', {})
        contact_score = sum([
            1 if contact.get('email') else 0,
            1 if contact.get('phone') else 0,
            1 if contact.get('linkedin') else 0,
            1 if contact.get('github') else 0
        ])
        features.append(contact_score)
        features.append(contact_score / 4.0)  # Normalized
        
        # Certifications
        certs = analysis.get('certifications', {})
        cert_count = sum(len(v) for v in certs.values()) if certs else 0
        features.append(cert_count)
        features.append(min(cert_count / 5.0, 1.0))  # Normalized
        
        # Projects
        projects = analysis.get('projects', {})
        project_count = projects.get('estimated_count', 0) if projects else 0
        features.append(project_count)
        features.append(min(project_count / 5.0, 1.0))  # Normalized
        
        # Match percentage or score
        if analysis.get('mode') == 'job_matching':
            match_pct = analysis.get('match_percentage', 0)
            features.append(match_pct)
            features.append(match_pct / 100.0)  # Normalized
        else:
            score = analysis.get('score', 0)
            features.append(score)
            features.append(score / 100.0)  # Normalized
        
        return np.array(features)
    
    def extract_text_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Extract TF-IDF features from text"""
        if fit:
            tfidf_features = self.tfidf.fit_transform(texts)
        else:
            tfidf_features = self.tfidf.transform(texts)
        
        return tfidf_features.toarray()
    
    def extract_semantic_features(self, texts: List[str]) -> Optional[np.ndarray]:
        """Extract semantic embeddings using sentence transformers"""
        if self.sentence_model is None:
            return None
        
        try:
            embeddings = self.sentence_model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            print(f"Error extracting semantic features: {e}")
            return None
    
    def extract_features(self, resume_texts: List[str], analyses: List[Dict], 
                        fit: bool = False) -> np.ndarray:
        """Extract all features and combine them"""
        # Metadata features
        metadata_features = np.array([
            self.extract_metadata_features(text, analysis)
            for text, analysis in zip(resume_texts, analyses)
        ])
        
        # Text features (TF-IDF)
        text_features = self.extract_text_features(resume_texts, fit=fit)
        
        # Semantic features (optional)
        semantic_features = self.extract_semantic_features(resume_texts)
        
        # Combine features
        if semantic_features is not None:
            all_features = np.hstack([metadata_features, text_features, semantic_features])
        else:
            all_features = np.hstack([metadata_features, text_features])
        
        # Scale features
        if fit:
            all_features = self.scaler.fit_transform(all_features)
            self.is_fitted = True
        else:
            if self.is_fitted:
                all_features = self.scaler.transform(all_features)
        
        return all_features
    
    def save(self, filepath: str):
        """Save feature extractor"""
        joblib.dump({
            'tfidf': self.tfidf,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load(self, filepath: str):
        """Load feature extractor"""
        data = joblib.load(filepath)
        self.tfidf = data['tfidf']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']


class EnsembleResumeClassifier:
    """Ensemble classifier combining multiple models for better accuracy"""
    
    def __init__(self, use_smote: bool = True):
        """Initialize ensemble classifier"""
        self.feature_extractor = ImprovedFeatureExtractor()
        self.use_smote = use_smote
        
        # Define base models with optimized hyperparameters
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=config.RANDOM_STATE
            )
        }
        
        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        self.is_trained = False
        self.classes_ = None
        self.training_metrics = {}
        self.individual_metrics = {}
    
    def prepare_labels(self, feedback_values: List[int]) -> np.ndarray:
        """Convert feedback values to binary labels"""
        labels = np.array([1 if val > 0 else 0 for val in feedback_values])
        return labels
    
    def apply_smote(self, X, y):
        """Apply SMOTE to balance classes"""
        try:
            # Check if we have at least 2 classes
            unique_classes, class_counts = np.unique(y, return_counts=True)
            if len(unique_classes) < 2:
                print(f"SMOTE skipped: Only one class present ({unique_classes[0]})")
                return X, y
            
            # Check if we have enough samples in minority class
            min_class_count = min(class_counts)
            if min_class_count < 2:
                print(f"SMOTE skipped: Minority class has only {min_class_count} sample(s)")
                return X, y
            
            # Apply SMOTE with appropriate k_neighbors
            k_neighbors = min(5, min_class_count - 1)
            if k_neighbors < 1:
                print(f"SMOTE skipped: Not enough samples for k_neighbors")
                return X, y
            
            smote = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"SMOTE applied: {len(y)} -> {len(y_resampled)} samples")
            print(f"  Original class distribution: {dict(zip(unique_classes, class_counts))}")
            unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
            print(f"  Resampled class distribution: {dict(zip(unique_resampled, counts_resampled))}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def train(self, resume_texts: List[str], analyses: List[Dict], 
             feedback_values: List[int]) -> Dict:
        """Train the ensemble classifier"""
        print("Extracting features...")
        X = self.feature_extractor.extract_features(resume_texts, analyses, fit=True)
        y = self.prepare_labels(feedback_values)
        
        print(f"Original dataset: {len(y)} samples, Class distribution: {np.bincount(y)}")
        
        # Apply SMOTE if enabled
        if self.use_smote and len(np.unique(y)) > 1:
            X, y = self.apply_smote(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        print(f"Training set: {len(y_train)}, Test set: {len(y_test)}")
        
        # Train individual models and evaluate
        print("\nTraining individual models...")
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate individual model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            self.individual_metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0
            }
            
            print(f"{name} - F1: {self.individual_metrics[name]['f1_score']:.3f}, "
                  f"Accuracy: {self.individual_metrics[name]['accuracy']:.3f}")
        
        # Train ensemble
        print("\nTraining ensemble model...")
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = self.ensemble.classes_
        
        # Evaluate ensemble
        y_pred = self.ensemble.predict(X_test)
        y_pred_proba = self.ensemble.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0,
            'training_samples': len(X),
            'test_samples': len(X_test),
            'used_smote': self.use_smote
        }
        
        # Cross-validation score
        if len(X) >= config.CV_FOLDS:
            print("\nPerforming cross-validation...")
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
            cv_scores = cross_val_score(self.ensemble, X, y, cv=cv, scoring='f1')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            print(f"Cross-validation F1: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
        
        self.training_metrics = metrics
        
        # Print summary
        print("\n" + "="*60)
        print("ENSEMBLE MODEL PERFORMANCE")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1_score']:.3f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
        print("="*60)
        
        return metrics
    
    def predict(self, resume_text: str, analysis: Dict) -> Tuple[int, float, Dict]:
        """Predict recommendation for a resume"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Extract features
        X = self.feature_extractor.extract_features([resume_text], [analysis], fit=False)
        
        # Predict
        prediction = self.ensemble.predict(X)[0]
        probabilities = self.ensemble.predict_proba(X)[0]
        confidence = probabilities.max()
        
        # Get class probabilities
        prob_dict = {
            'not_recommend': probabilities[0] if len(probabilities) > 0 else 0,
            'recommend': probabilities[1] if len(probabilities) > 1 else 0
        }
        
        return prediction, confidence, prob_dict
    
    def get_recommendation_from_prediction(self, prediction: int, confidence: float) -> Tuple[str, str]:
        """Convert prediction to recommendation text"""
        if prediction == 1:
            if confidence >= config.HIGHLY_RECOMMENDED_THRESHOLD:
                return "Highly Recommended", "excellent"
            elif confidence >= config.RECOMMENDED_THRESHOLD:
                return "Recommended", "good"
            else:
                return "Maybe", "average"
        else:
            return "Not Recommended", "poor"
    
    def save(self, filepath: str):
        """Save ensemble model and feature extractor"""
        model_dir = os.path.dirname(filepath)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble model
        joblib.dump({
            'ensemble': self.ensemble,
            'models': self.models,
            'is_trained': self.is_trained,
            'classes': self.classes_,
            'training_metrics': self.training_metrics,
            'individual_metrics': self.individual_metrics,
            'use_smote': self.use_smote
        }, filepath)
        
        # Save feature extractor
        feature_extractor_path = filepath.replace('.pkl', '_features.pkl')
        self.feature_extractor.save(feature_extractor_path)
        
        print(f"Model saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load ensemble model and feature extractor"""
        # Load model
        data = joblib.load(filepath)
        self.ensemble = data['ensemble']
        self.models = data['models']
        self.is_trained = data['is_trained']
        self.classes_ = data['classes']
        self.training_metrics = data.get('training_metrics', {})
        self.individual_metrics = data.get('individual_metrics', {})
        self.use_smote = data.get('use_smote', True)
        
        # Load feature extractor
        feature_extractor_path = filepath.replace('.pkl', '_features.pkl')
        if os.path.exists(feature_extractor_path):
            self.feature_extractor.load(feature_extractor_path)
        
        print(f"Model loaded from: {filepath}")


class ImprovedModelManager:
    """Manage improved ML models"""
    
    def __init__(self):
        """Initialize model manager"""
        self.models = {}
        self.active_model = None
    
    def get_model_path(self, model_type: str, version: str) -> str:
        """Get path for model file"""
        filename = f"{model_type}_ensemble_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        return os.path.join(config.MODELS_DIR, filename)
    
    def load_active_model(self, model_type: str = 'classifier') -> Optional[EnsembleResumeClassifier]:
        """Load the active model from disk"""
        from mongodb_database import MongoDatabase
        
        db = MongoDatabase()
        model_metadata = db.get_active_model(model_type)
        
        if model_metadata and os.path.exists(model_metadata['model_path']):
            try:
                classifier = EnsembleResumeClassifier()
                classifier.load(model_metadata['model_path'])
                self.active_model = classifier
                print(f"Active model loaded: F1={classifier.training_metrics.get('f1_score', 0):.3f}")
                return classifier
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        
        return None
    
    def save_model(self, classifier: EnsembleResumeClassifier, version: str) -> str:
        """Save model to disk"""
        model_path = self.get_model_path('classifier', version)
        classifier.save(model_path)
        return model_path
    
    def has_active_model(self) -> bool:
        """Check if there's an active trained model"""
        return self.active_model is not None and self.active_model.is_trained


# Global improved model manager instance
improved_model_manager = ImprovedModelManager()
