"""
Batch Resume Processor
Feed multiple resumes to the ML model for training or analysis
"""

import os
import glob
from pathlib import Path
from app import extract_text_from_file, analyze_resume_standalone, analyze_resume_with_jd
from database import db
import json

class BatchResumeProcessor:
    """Process multiple resumes in batch"""
    
    def __init__(self, resumes_folder='resumes'):
        self.resumes_folder = resumes_folder
        self.processed_count = 0
        self.failed_count = 0
        
    def process_folder(self, job_description=None, save_to_db=True):
        """
        Process all resumes in a folder
        
        Args:
            job_description: Optional job description to match against
            save_to_db: Whether to save results to database for ML training
        
        Returns:
            List of analysis results
        """
        results = []
        
        # Get all resume files
        resume_files = []
        for ext in ['*.pdf', '*.docx', '*.doc', '*.txt']:
            resume_files.extend(glob.glob(os.path.join(self.resumes_folder, ext)))
        
        print(f"Found {len(resume_files)} resume files")
        
        for filepath in resume_files:
            try:
                filename = os.path.basename(filepath)
                print(f"Processing: {filename}")
                
                # Extract text
                resume_text = extract_text_from_file(filepath)
                
                if resume_text.startswith("Error") or resume_text.startswith("PDF parsing") or resume_text.startswith("DOCX parsing"):
                    print(f"  ❌ Failed to extract text: {resume_text}")
                    self.failed_count += 1
                    continue
                
                # Analyze resume
                if job_description:
                    analysis = analyze_resume_with_jd(resume_text, job_description)
                else:
                    analysis = analyze_resume_standalone(resume_text)
                
                # Add metadata
                analysis['filename'] = filename
                analysis['filepath'] = filepath
                analysis['resume_text'] = resume_text
                
                # Save to database if requested
                if save_to_db:
                    try:
                        # Save resume
                        resume_id = db.save_resume(
                            filename=filename,
                            resume_text=resume_text,
                            word_count=len(resume_text.split()),
                            has_education=analysis.get('has_education', False),
                            experience_years=analysis.get('experience_years', 0)
                        )
                        
                        # Save job description if provided
                        jd_id = None
                        if job_description:
                            from app import extract_keywords_from_jd
                            jd_requirements = extract_keywords_from_jd(job_description)
                            jd_id = db.save_job_description(
                                description_text=job_description,
                                required_skills=jd_requirements['required_skills'],
                                experience_years=jd_requirements['experience_years']
                            )
                        
                        # Save analysis - MongoDB version takes analysis dict
                        analysis_id = db.save_analysis_result(
                            resume_id=resume_id,
                            analysis=analysis,
                            job_description_id=jd_id
                        )
                        
                        analysis['resume_id'] = resume_id
                        analysis['analysis_id'] = analysis_id
                        
                        print(f"  ✓ Saved to database (ID: {analysis_id})")
                    except Exception as e:
                        print(f"  ⚠ Database save failed: {e}")
                
                results.append(analysis)
                self.processed_count += 1
                
                print(f"  ✓ Score: {analysis.get('match_percentage') or analysis.get('score', 0):.1f}%")
                print(f"  ✓ Recommendation: {analysis.get('recommendation')}")
                
            except Exception as e:
                print(f"  ❌ Error processing {filepath}: {e}")
                self.failed_count += 1
        
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete!")
        print(f"Successfully processed: {self.processed_count}")
        print(f"Failed: {self.failed_count}")
        print(f"{'='*60}\n")
        
        return results
    
    def process_from_kaggle_dataset(self, dataset_path, job_description=None, limit=None):
        """
        Process resumes from downloaded Kaggle dataset
        
        Args:
            dataset_path: Path to the Kaggle dataset folder
            job_description: Optional job description
            limit: Maximum number of resumes to process
        """
        import pandas as pd
        
        # Look for CSV files in the dataset
        csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {dataset_path}")
            return []
        
        results = []
        
        for csv_file in csv_files:
            print(f"Processing dataset: {csv_file}")
            
            try:
                df = pd.read_csv(csv_file)
                print(f"Found {len(df)} records in dataset")
                
                # Common column names for resume text
                text_columns = ['Resume', 'resume', 'Resume_str', 'resume_text', 'text', 'Text']
                resume_column = None
                
                for col in text_columns:
                    if col in df.columns:
                        resume_column = col
                        break
                
                if not resume_column:
                    print(f"Could not find resume text column. Available columns: {df.columns.tolist()}")
                    continue
                
                # Process each resume
                count = 0
                for idx, row in df.iterrows():
                    if limit and count >= limit:
                        break
                    
                    try:
                        resume_text = str(row[resume_column])
                        
                        if len(resume_text) < 50:  # Skip very short texts
                            continue
                        
                        # Analyze
                        if job_description:
                            analysis = analyze_resume_with_jd(resume_text, job_description)
                        else:
                            analysis = analyze_resume_standalone(resume_text)
                        
                        analysis['filename'] = f"kaggle_resume_{idx}.txt"
                        analysis['source'] = 'kaggle_dataset'
                        analysis['resume_text'] = resume_text
                        
                        # Save to database
                        try:
                            resume_id = db.save_resume(
                                filename=analysis['filename'],
                                resume_text=resume_text,
                                word_count=len(resume_text.split()),
                                has_education=analysis.get('has_education', False),
                                experience_years=analysis.get('experience_years', 0)
                            )
                            analysis['resume_id'] = resume_id
                        except Exception as e:
                            print(f"  ⚠ Database save failed for row {idx}: {e}")
                        
                        results.append(analysis)
                        count += 1
                        
                        if count % 10 == 0:
                            print(f"  Processed {count} resumes...")
                        
                    except Exception as e:
                        print(f"  Error processing row {idx}: {e}")
                        continue
                
                self.processed_count += count
                print(f"Processed {count} resumes from {csv_file}")
                
            except Exception as e:
                print(f"Error reading CSV {csv_file}: {e}")
        
        return results
    
    def export_results(self, results, output_file='batch_results.json'):
        """Export analysis results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results exported to {output_file}")


def main():
    """Example usage"""
    processor = BatchResumeProcessor()
    
    # Option 1: Process resumes from the resumes folder
    print("=" * 60)
    print("OPTION 1: Process resumes from folder")
    print("=" * 60)
    
    job_description = """
    We are looking for a Python Developer with 3+ years of experience.
    Required skills: Python, Django, Flask, REST API, SQL, Git
    Preferred: AWS, Docker, Machine Learning
    """
    
    # Uncomment to process folder
    # results = processor.process_folder(job_description=job_description, save_to_db=True)
    
    # Option 2: Process from Kaggle dataset
    print("\n" + "=" * 60)
    print("OPTION 2: Process resumes from Kaggle dataset")
    print("=" * 60)
    
    # First, download the dataset using resume_data.py
    try:
        import kagglehub
        path = kagglehub.dataset_download("palaksood97/resume-dataset")
        print(f"Dataset downloaded to: {path}")
        
        # Process the dataset
        results = processor.process_from_kaggle_dataset(
            dataset_path=path,
            job_description=job_description,
            limit=50  # Process first 50 resumes
        )
        
        # Export results
        processor.export_results(results)
        
    except ImportError:
        print("kagglehub not installed. Install with: pip install kagglehub")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
