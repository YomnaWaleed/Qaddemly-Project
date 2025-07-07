import json
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import pandas as pd

def load_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
  
  

def preprocess_text(text):
    """Preprocess text by removing special chars, lowercasing, etc."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
  

def extract_resume_data(applicant):
    """Extract and combine all relevant fields from a resume"""
    # Combine skills
    skills = ' '.join(applicant.get('skills', []))
    
    # Combine education
    education_parts = []
    for edu in applicant.get('job_application_education', []):
        education_parts.append(edu.get('university', ''))
        education_parts.append(edu.get('field_of_study', ''))
    education = ' '.join(education_parts)
    
    # Combine experience
    experience_parts = []
    for exp in applicant.get('job_application_experience', []):
        experience_parts.append(exp.get('job_title', ''))
        experience_parts.append(exp.get('description', ''))
        experience_parts.append(exp.get('company_name', ''))
    experience = ' '.join(experience_parts)
    
    # Combine question answers
    qa_parts = []
    for qa in applicant.get('questionAnswers', []):
        qa_parts.append(qa.get('answer', ''))
    qa_text = ' '.join(qa_parts)
    
    # Combine all parts
    full_resume = f"{skills} {education} {experience} {qa_text}"
    
    return full_resume


def extract_job_description(job):
    """Extract and combine job description fields"""
    title = job.get('title', '')
    description = job.get('description', '')
    skills = ' '.join(job.get('skills', []))
    keywords = ' '.join(job.get('keywords', []))
    
    full_job_desc = f"{title} {description} {skills} {keywords}"
    
    return full_job_desc

def summarize_text(text, sentence_count=3):
    """Summarize text using LexRank algorithm"""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


def calculate_similarity(resumes, job_desc):
    """Calculate similarity scores using n-grams and L1 distance"""
    # Create n-gram vectorizer (3-grams and 5-grams)
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    
    # Combine resumes and job description for consistent vectorization
    all_texts = resumes + [job_desc]
    X = vectorizer.fit_transform(all_texts)
    
    # Separate back into resumes and job
    resume_vectors = X[:len(resumes)]
    job_vector = X[len(resumes):]
    
    # Calculate L1 distance (Manhattan distance)
    distances = pairwise_distances(resume_vectors, job_vector, metric='manhattan')
    
    # Convert distances to similarity scores (lower distance = higher similarity)
    scores = 1 / (1 + distances.flatten())
    
    return scores


def process_data(input_data):
    """Process the input data and return ranked applicants"""
    # Extract and preprocess all resumes
    all_resumes = []
    resume_ids = []
    for applicant in input_data['jobApplication']:
        resume = extract_resume_data(applicant)
        preprocessed_resume = preprocess_text(resume)
        all_resumes.append(preprocessed_resume)
        resume_ids.append(applicant['id'])
    
    # Extract and preprocess job description
    job_desc = extract_job_description(input_data['job'])
    preprocessed_job_desc = preprocess_text(job_desc)
    
    # Summarize job description using LexRank
    summarized_job_desc = summarize_text(preprocessed_job_desc)
    
    # Calculate similarity scores
    scores = calculate_similarity(all_resumes, summarized_job_desc)
    
    # Create a DataFrame for results
    results = pd.DataFrame({
        'ID': resume_ids,
        'Score': scores,
        'Applicant': [f"{app['first_name']} {app['last_name']}" for app in input_data['jobApplication']]
    })
    
    # Sort by score
    results = results.sort_values('Score', ascending=False)
    
    return results.to_dict(orient='records')