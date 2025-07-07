import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

user_df = pd.read_pickle('user_df.pkl')
jobs_df = pd.read_pickle('jobs_df.pkl')

def recommend_for_user(user_df, jobs_df, title_w = 0.2, cosine_w = 0.2, jaccard_w = 0.2, emp_type_w = 0.2, loc_type_w = 0.2, top_n = 20):
    user_df = user_df.iloc[0]

    jobs_df['Title Equal'] = jobs_df['title'].apply(lambda x : 1 if x == user_df['subtitle'] else 0)
    jobs_df['Employee Equal'] = jobs_df['employee_type'].apply(lambda x : 1 if x == user_df['employment_type'] else 0)
    jobs_df['Location Equal'] = jobs_df['location_type'].apply(lambda x : 1 if x == user_df['location_type'] else 0)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    job_vectors = vectorizer.fit_transform(jobs_df['description'])
    user_vector = vectorizer.transform([user_df['about_me']])
    
    # Compute raw TF-IDF similarities
    raw_similarities = cosine_similarity(user_vector, job_vectors).flatten()
    max_score = raw_similarities.max()
    normalized_similarities = raw_similarities / max_score if max_score > 0 else raw_similarities
    jobs_df['description_similarity'] = normalized_similarities

    user_skills = set(user_df["skills"])

    def compute_similarity(user_emb, job_emb):
        """
        Compute cosine similarity between a user's skill embedding and a list of job skill embeddings.
        
        Args:
            user_emb (np.ndarray): User's skill embedding, shape (n_features,) or (1, n_features).
            job_emb (list or pd.Series): List or Series of job embeddings, each shape (n_features,).
        
        Returns:
            np.ndarray: Array of similarity scores for each job.
        """
        # Convert user_emb to 2D array if 1D
        if user_emb.ndim == 1:
            user_emb = user_emb.reshape(1, -1)
        
        # Convert job_emb to 2D array
        job_emb = np.vstack(job_emb)
        
        # Check for empty inputs
        if user_emb.shape[0] == 0 or job_emb.shape[0] == 0:
            return np.array([0] * len(job_emb))
    
        # Compute cosine similarity
        sim_matrix = cosine_similarity(user_emb, job_emb)
        return sim_matrix[0]  # Return 1D array of similarities for the single user
    
    # Compute hybrid similarity with bidirectional matching
    # def compute_hybrid_similarity(user_skills, job_skills, threshold=0.6):
    #     """
    #     Compute Jaccard similarity based on one-to-one skill matches identified via cosine similarity.
        
    #     Args:
    #         user_skills (list): List of user skills (strings).
    #         job_skills (list): List of job skills (strings).
    #         threshold (float): Cosine similarity threshold for considering skills matched.
        
    #     Returns:
    #         float: Jaccard similarity score based on one-to-one matches.
    #     """
    #     if not user_skills or not job_skills:
    #         return 0.0

    #     with open('skill_embeddings.pkl', 'rb') as f:
    #         skill_to_embedding = pickle.load(f)
        
    #     # Get embeddings
    #     user_embeddings = np.array([skill_to_embedding.get(skill, np.zeros(384)) for skill in user_skills])
    #     job_embeddings = np.array([skill_to_embedding.get(skill, np.zeros(384)) for skill in job_skills])
        
    #     # Compute cosine similarity matrix
    #     sim_matrix = cosine_similarity(user_embeddings, job_embeddings)
        
    #     # Find one-to-one matches (greedy approach)
    #     matches = []
    #     for i in range(len(user_skills)):
    #         for j in range(len(job_skills)):
    #             if sim_matrix[i, j] > threshold:
    #                 matches.append((sim_matrix[i, j], i, j))
        
    #     # Sort matches by similarity (highest first)
    #     matches.sort(reverse=True)
        
    #     # Assign matches, ensuring each skill is used only once
    #     used_user = set()
    #     used_job = set()
    #     common_skills = 0
    #     for sim, user_idx, job_idx in matches:
    #         if user_idx not in used_user and job_idx not in used_job:
    #             common_skills += 1
    #             used_user.add(user_idx)
    #             used_job.add(job_idx)

    #     return common_skills/20.0
        # # Compute Jaccard similarity
        # total_skills = len(user_skills) + len(job_skills) - common_skills
        # return common_skills / total_skills if total_skills > 0 else 0.0

##-----------------------------
    def compute_common_skills(user_skills, job_skills, threshold=0.6):
        """
        Count job skills that have at least one user skill with cosine similarity > threshold.
        
        Args:
            user_skills (list): List of user skills (strings).
            job_skills (list): List of job skills (strings).
            threshold (float): Cosine similarity threshold for a match.
        
        Returns:
            int: Number of common skills.
        """
        if not user_skills or not job_skills:
            return 0
        
        # Get embeddings for skills
        with open('skill_embeddings.pkl', 'rb') as f:
            skill_to_embedding = pickle.load(f)
        user_embeddings = np.array([skill_to_embedding.get(skill, np.zeros(384)) for skill in user_skills])
        job_embeddings = np.array([skill_to_embedding.get(skill, np.zeros(384)) for skill in job_skills])
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(user_embeddings, job_embeddings)
        
        # Count job skills with at least one match above threshold
        common_skills = sum(any(sim_matrix[i, j] > threshold for i in range(len(user_skills))) for j in range(len(job_skills)))
        return common_skills

    jobs_df['common_skills'] = jobs_df['parsed_skills'].apply(lambda job_skills: compute_common_skills(user_skills, job_skills, 0.5))
    
    # Normalize common skills (max gets 1)
    max_common = jobs_df['common_skills'].max()
    if max_common > 0:
        jobs_df['skill_similarity'] = jobs_df['common_skills'] / max_common
    else:
        jobs_df['skill_similarity'] = 0.0

    # jobs_df['skill_similarity'] = jobs_df['parsed_skills'].apply(
    #     lambda job_skills: compute_hybrid_similarity(user_df['skills'], job_skills, 0.4)
    # )
    
    final_scores = (
            title_w * jobs_df['Title Equal']
            + cosine_w * jobs_df['description_similarity']
            + emp_type_w * jobs_df['Employee Equal']
            + loc_type_w * jobs_df['Location Equal']
            + jaccard_w * jobs_df['skill_similarity']
                    )

    top_indices = final_scores.argsort()[::-1][:top_n]
    recommended_jobs = jobs_df.iloc[top_indices].copy()
    recommended_jobs["similarity_score"] = final_scores[top_indices]
    # print("FINAL SCORES: ")
    # print(final_scores)

    return recommended_jobs.to_dict(orient="records")