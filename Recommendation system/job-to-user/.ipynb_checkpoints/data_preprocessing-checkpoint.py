import numpy as np
import pandas as pd
import json
import re
import nltk
import ast
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Cleaning Skills
def parse_skills(skills_list):
    """
    Parse a list of skill strings into a list of skill phrases based on capitalization,
    after removing content inside parentheses and English stop words, preserving case sensitivity.
    
    Args:
        skills_list (list): List of strings containing concatenated skills.
    
    Returns:
        list: List of lists, where each sublist contains parsed skill phrases.
    """
    def clean_string(s):
        # Remove content inside parentheses, including the parentheses
        return re.sub(r'\s*\(.*?\)', '', s)

    def standardize_skill(skill):
    # Lowercase and remove punctuation for consistent counting
        skill = skill.lower()
        skill = re.sub(r'[^\w\s]', '', skill)
        return skill.strip()
    
    def parse_single_skills(skills_string):
        # Split into words
        words = skills_string.split()
        # Filter out stop words (case-sensitive exact match)
        words = [word for word in words if word not in stop_words]
        
        skills = []
        current_skill = []
        prev_starts_with_capital = False  # Track previous word's capitalization
        
        for word in words:
            if word:  # Ensure word is not empty
                starts_with_capital = word[0].isupper()
                # Start a new skill if current word is capitalized and previous is not
                if starts_with_capital:
                    skill_phrase = ' '.join(current_skill)
                    if skill_phrase:  # Only append non-empty phrases
                        std_skill_phrase = standardize_skill(skill_phrase)
                        skills.append(std_skill_phrase)
                    current_skill = [word]
                else:
                    current_skill.append(word)
                prev_starts_with_capital = starts_with_capital
        
        # Append the last skill if it exists and is non-empty
        if current_skill:
            skill_phrase = ' '.join(current_skill)
            if skill_phrase:
                std_skill_phrase = standardize_skill(skill_phrase)
                skills.append(std_skill_phrase)
        
        return skills
    
    cleaned = clean_string(skills_list[0])
    parsed_skills = parse_single_skills(cleaned)
    
    return parsed_skills

# Cleaning String Objects
def preprocess_object(user_df, jobs_df):
    def preprocess_text(text):
        lemmatizer = WordNetLemmatizer()
    
        def preprocess_string(text_str):
            text_str = re.sub(r"[^a-zA-Z0-9 _-]", "", text_str)
            text_str = text_str.lower().strip()
            words = word_tokenize(text_str)
            filtered_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
            return " ".join(filtered_words)
    
        if isinstance(text, str):
            return preprocess_string(text)
    
        elif isinstance(text, list):
            return [preprocess_text(item) for item in text]
    
        elif isinstance(text, dict):
            return {preprocess_text(key): preprocess_text(value) if isinstance(value, str) else str(value) for
                    key, value in text.items()}
    
    for col in jobs_df.columns:
        if jobs_df[col].dtype == 'object':
            jobs_df[col] = jobs_df[col].apply(preprocess_text)
    
    for col in user_df.columns:
        if (user_df[col].dtype == 'object'):
            user_df[col] = user_df[col].apply(preprocess_text)
    
    def get_last_experience(df):
        start_date = ""
        last_exp = None
        for job in user_df['experiences'][0]:
            if job['start_date'] > start_date:
                last_date = job['start_date']
                last_exp = job
    
        return last_exp
    
    def preprocess_user_skills(skills):
        return [value for d in skills for value in d.values()]
    
    user_last_experience = get_last_experience(user_df)
    user_df['location_type'] = user_last_experience['location_type']
    user_df['employment_type'] = user_last_experience['employment_type']
    user_df['skills'] = user_df['skills'].apply(preprocess_user_skills)
    
    return user_df, jobs_df
    
user_df, jobs_df = preprocess(user_data, job_data)

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

def get_top_10_words(description):
    words = clean_and_tokenize(description)
    word_counts = Counter(words)
    top_10 = word_counts.most_common(10)
    return [word for word, count in top_10]
    # return [(word, count) for word, count in top_10]

# jobs_df['top_10_words'] = jobs_df['description'].apply(get_top_10_words)
# user_df['top_10_words'] = user_df['about_me'].apply(get_top_10_words)


# Preparing Skills embeddings

# # List to store embeddings
# job_embeddings = []

# for job in jobs_df.itertuples():
#     if job.parsed_skills:  # Check if the list isn’t empty
#         skill_embeddings = model.encode(job.parsed_skills)
#         job_embedding = np.mean(skill_embeddings, axis=0)  # Average embeddings
#     else:
#         job_embedding = np.zeros(model.get_sentence_embedding_dimension())
#     job_embeddings.append(job_embedding)

# user_embeddings = []
# for user in user_df.itertuples():
#     if user.skills:  # Check if the list isn’t empty
#         skill_embeddings = model.encode(user.skills)
#         user_embedding = np.mean(skill_embeddings, axis=0)  # Average embeddings
#     else:
#         user_embedding = np.zeros(model.get_sentence_embedding_dimension())
#     user_embeddings.append(user_embedding)

# Add embeddings to DataFrame
jobs_df['skill_embeddings'] = job_embeddings
user_df['skill_embeddings'] = user_embeddings

def precompute_skill_embeddings(user_df, jobs_df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_skills = set()
    for skills in jobs_df['parsed_skills']:
        all_skills.update(skills)
    for skills in user_df['skills']:
        all_skills.update(skills)
        
    all_skills = list(all_skills)
    skill_embeddings = model.encode(all_skills, batch_size=128, show_progress_bar=True)
    skill_to_embedding = dict(zip(all_skills, skill_embeddings))
    with open('skill_embeddings.pkl', 'wb') as f:
        pickle.dump(skill_to_embedding, f)
    return skill_to_embedding

# precompute_skill_embeddings(user_df, jobs_df)

# jobs_df.to_pickle('jobs_df.pkl')
# user_df.to_pickle('user_df.pkl')

def preprocess(input_json = 'inputJson (1).json'):
    # Reading Data
    input_json = 'inputJson (1).json'
    with open(input_json, 'r') as file:
        data = json.load(file)
    
    user_data = pd.DataFrame([data['user']])
    job_data = pd.DataFrame([job for job in data['jobs']])
    
    job_data = job_data.drop_duplicates(subset=['description'], keep='first').reset_index(drop=True)
    
    # Doownloading Stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    jobs_df['parsed_skills'] = jobs_df['skills'].apply(parse_skills)
    user_df, jobs_df = preprocess_object(user_df, jobs_df)

    jobs_df['top_10_words'] = jobs_df['description'].apply(get_top_10_words)
    user_df['top_10_words'] = user_df['about_me'].apply(get_top_10_words)

    precompute_skill_embeddings(user_df, jobs_df)

    jobs_df.to_pickle('jobs_df.pkl')
    user_df.to_pickle('user_df.pkl')

    return