# Improved Training Pipeline with Ensemble Models
# Enhanced training with better evaluation and monitoring

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from mongodb_database import MongoDatabase
from ml_models_improved import EnsembleResumeClassifier, improved_model_manager
import config

# Initialize MongoDB database
db = MongoDatabase()


class ImprovedTrainingPipeline:
    """Enhanced training pipeline with better evaluation"""
    
    def __init__(self):
        """Initialize training pipeline"""
        self.classifier = None
        self.training_data = None
        self.metrics = None
    
    def check_training_readiness(self) -> Tuple[bool, str]:
        """Check if enough data is available for training"""
        sample_count = db.get_training_data_count()
        
        if sample_count < config.MIN_TRAINING_SAMPLES:
            return False, f"Need {config.MIN_TRAINING_SAMPLES - sample_count} more labeled samples (current: {sample_count})"
        
        return True, f"{sample_count} samples available for training ✓"
    
    def prepare_training_data(self) -> Tuple[List[str], List[Dict], List[int]]:
        """Fetch and prepare training data from database"""
        raw_data = db.get_training_data()
        
        if not raw_data:
            raise ValueError("No training data available")
        
        resume_texts = []
        analyses = []
        labels = []
        
        for row in raw_data:
            resume_text, jd_text, match_pct, score, quality_score, \
                skills_found, missing_skills, feedback_value = row
            
            # MongoDB returns native arrays, not JSON strings
            # Handle both cases for compatibility
            if isinstance(skills_found, str):
                skills_found_list = json.loads(skills_found) if skills_found else []
            else:
                skills_found_list = skills_found if skills_found else []
            
            if isinstance(missing_skills, str):
                missing_skills_list = json.loads(missing_skills) if missing_skills else []
            else:
                missing_skills_list = missing_skills if missing_skills else []
            
            # Create analysis dict
            analysis = {
                'mode': 'job_matching' if jd_text else 'standalone',
                'match_percentage': match_pct or 0,
                'score': score or 0,
                'quality': {'score': quality_score or 0, 'word_count': len(resume_text.split())},
                'required_skills_found': skills_found_list,
                'missing_skills': missing_skills_list,
                'preferred_skills_found': [],
                'has_education': True,
                'experience_years': self._extract_experience(resume_text),
                'contact_info': {},
                'certifications': {},
                'projects': {'estimated_count': 0},
                'total_skills': len(skills_found_list)
            }
            
            resume_texts.append(resume_text)
            analyses.append(analysis)
            labels.append(feedback_value)
        
        return resume_texts, analyses, labels
    
    def _extract_experience(self, text: str) -> int:
        """Quick experience extraction"""
        import re
        text_lower = text.lower()
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
            r'experience\s*[:–-]\s*(\d+)\+?\s*years?',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return max([int(m) for m in matches])
        return 0
    
    def train_model(self, use_smote: bool = True) -> Dict:
        """Train improved ensemble model"""
        print("\n" + "="*70)
        print("STARTING IMPROVED MODEL TRAINING")
        print("="*70)
        
        # Check readiness
        ready, message = self.check_training_readiness()
        if not ready:
            raise ValueError(f"Not ready for training: {message}")
        
        print(f"✓ {message}")
        
        # Prepare data
        print("\n📊 Preparing training data...")
        resume_texts, analyses, labels = self.prepare_training_data()
        
        # Create and train classifier
        print(f"\n🤖 Training ensemble classifier (XGBoost + LightGBM + Random Forest + Gradient Boosting)...")
        print(f"   SMOTE enabled: {use_smote}")
        
        self.classifier = EnsembleResumeClassifier(use_smote=use_smote)
        metrics = self.classifier.train(resume_texts, analyses, labels)
        
        # Generate evaluation plots
        print("\n📈 Generating evaluation plots...")
        self.generate_evaluation_plots(self.classifier, resume_texts, analyses, labels)
        
        # Save model
        model_path = improved_model_manager.save_model(self.classifier, config.MODEL_VERSION)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save metadata to database
        db.save_model_metadata(
            model_version=config.MODEL_VERSION,
            model_type='ensemble_classifier',
            training_samples=metrics['training_samples'],
            metrics=metrics,
            model_path=model_path
        )
        
        # Update active model in manager
        improved_model_manager.active_model = self.classifier
        
        self.metrics = metrics
        
        # Print final summary
        self.print_training_summary(metrics)
        
        return metrics
    
    def generate_evaluation_plots(self, classifier, resume_texts, analyses, labels):
        """Generate evaluation visualizations"""
        try:
            from sklearn.model_selection import train_test_split
            
            # Extract features
            X = classifier.feature_extractor.extract_features(resume_texts, analyses, fit=False)
            y = np.array([1 if val > 0 else 0 for val in labels])
            
            # Split for evaluation
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            if len(X_test) == 0:
                print("Not enough test data for plots")
                return
            
            # Predictions
            y_pred = classifier.ensemble.predict(X_test)
            y_pred_proba = classifier.ensemble.predict_proba(X_test)
            
            # Create plots directory
            plots_dir = os.path.join(config.MODELS_DIR, 'evaluation_plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. Confusion Matrix
            self._plot_confusion_matrix(y_test, y_pred, plots_dir)
            
            # 2. ROC Curve
            if len(np.unique(y_test)) > 1:
                self._plot_roc_curve(y_test, y_pred_proba[:, 1], plots_dir)
            
            # 3. Model Comparison
            self._plot_model_comparison(classifier.individual_metrics, plots_dir)
            
            print(f"   Plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"   Warning: Could not generate plots: {e}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, save_dir):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Recommend', 'Recommend'],
                   yticklabels=['Not Recommend', 'Recommend'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_scores, save_dir):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150)
        plt.close()
    
    def _plot_model_comparison(self, individual_metrics, save_dir):
        """Plot comparison of individual models"""
        if not individual_metrics:
            return
        
        models = list(individual_metrics.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        data = {metric: [individual_metrics[model][metric] for model in models] 
                for metric in metrics_names}
        
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics_names):
            offset = width * (i - 2)
            ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150)
        plt.close()
    
    def print_training_summary(self, metrics):
        """Print comprehensive training summary"""
        print("\n" + "="*70)
        print("TRAINING COMPLETE! 🎉")
        print("="*70)
        print(f"\n📊 ENSEMBLE MODEL METRICS:")
        print(f"   Accuracy:  {metrics['accuracy']:.1%}")
        print(f"   Precision: {metrics['precision']:.1%}")
        print(f"   Recall:    {metrics['recall']:.1%}")
        print(f"   F1 Score:  {metrics['f1_score']:.1%}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.1%}")
        
        if 'cv_mean' in metrics:
            print(f"\n🔄 CROSS-VALIDATION:")
            print(f"   Mean F1:   {metrics['cv_mean']:.1%} (+/- {metrics['cv_std']:.1%})")
        
        print(f"\n📈 DATASET INFO:")
        print(f"   Training samples: {metrics['training_samples']}")
        print(f"   Test samples:     {metrics['test_samples']}")
        print(f"   SMOTE applied:    {metrics.get('used_smote', False)}")
        
        print("\n" + "="*70)
        
        # Performance assessment
        f1 = metrics['f1_score']
        if f1 >= 0.85:
            print("🌟 EXCELLENT! Model performance is outstanding!")
        elif f1 >= 0.75:
            print("✅ GOOD! Model performance is solid.")
        elif f1 >= 0.65:
            print("⚠️  FAIR. Consider collecting more training data.")
        else:
            print("❌ NEEDS IMPROVEMENT. More training data required.")
        
        print("="*70 + "\n")
    
    def evaluate_model(self) -> Dict:
        """Evaluate the current model"""
        if self.classifier is None or not self.classifier.is_trained:
            raise ValueError("No trained model available")
        
        return self.classifier.training_metrics
    
    def auto_retrain_check(self) -> bool:
        """Check if automatic retraining should be triggered"""
        sample_count = db.get_training_data_count()
        
        stats = db.get_statistics()
        active_models = stats.get('active_models', 0)
        
        if active_models == 0:
            return sample_count >= config.MIN_TRAINING_SAMPLES
        
        return sample_count % config.RETRAIN_THRESHOLD == 0


# Global improved training pipeline instance
improved_training_pipeline = ImprovedTrainingPipeline()


def trigger_improved_training(use_smote: bool = True) -> Dict:
    """Trigger improved model training"""
    try:
        metrics = improved_training_pipeline.train_model(use_smote=use_smote)
        return {
            'success': True,
            'metrics': metrics,
            'message': 'Improved ensemble model trained successfully! 🎉'
        }
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'message': f'Training failed: {str(e)}'
        }


def get_improved_training_status() -> Dict:
    """Get current training status"""
    ready, message = improved_training_pipeline.check_training_readiness()
    stats = db.get_statistics()
    
    return {
        'ready_for_training': ready,
        'message': message,
        'statistics': stats,
        'has_active_model': improved_model_manager.has_active_model(),
        'min_samples_required': config.MIN_TRAINING_SAMPLES,
        'model_type': 'ensemble (XGBoost + LightGBM + RF + GB)'
    }
