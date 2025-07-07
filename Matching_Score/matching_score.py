#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matching score module for job recommendation system.

This module provides a class to compute similarity scores between a user and a job
based on multiple criteria, including title, description, skills, employment type,
and location type, using embeddings and logarithmic scaling.
"""

import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import math


class MatchingScore:
    """
    A class to compute similarity scores between a user and a job for job recommendations.

    Scores are calculated using embedding-based similarities for titles and descriptions,
    skill overlap, and binary matching for employment and location types, with weights
    applied to combine the components. A logarithmic transformation is used to scale
    certain similarity scores.

    Parameters
    ----------
    model : SentenceTransformer
        Preloaded SentenceTransformer model for encoding text into embeddings.
    skill_embeddings : dict
        Dictionary mapping skills to their precomputed embeddings (numpy arrays).
    title_w : float, optional
        Weight for title similarity in the final score, default is 0.2.
    cosine_w : float, optional
        Weight for description similarity in the final score, default is 0.2.
    jaccard_w : float, optional
        Weight for skill similarity in the final score, default is 0.2.
    emp_type_w : float, optional
        Weight for employment type matching in the final score, default is 0.2.
    loc_type_w : float, optional
        Weight for location type matching in the final score, default is 0.2.
    top_n : int, optional
        Number of recommendations to return (not used in score computation), default is 20.
    k : int, optional
        Scaling factor for logarithmic transformation, default is 10.

    Attributes
    ----------
    model : SentenceTransformer
        The SentenceTransformer model used for encoding.
    skill_embeddings : dict
        The precomputed skill embeddings.
    title_weight : float
        Weight for title similarity.
    cosine_weight : float
        Weight for description similarity.
    jaccard_weight : float
        Weight for skill similarity.
    emp_type_weight : float
        Weight for employment type matching.
    loc_type_weight : float
        Weight for location type matching.
    top_n : int
        Number of recommendations (placeholder for future use).
    k : int
        Scaling factor for logarithmic transformation.
    """

    def __init__(self, model, skill_embeddings, title_w=0.2, cosine_w=0.2, jaccard_w=0.2,
                 emp_type_w=0.2, loc_type_w=0.2, top_n=20, k=10):
        self.model = model  # Preloaded SentenceTransformer
        self.skill_embeddings = skill_embeddings  # Preloaded skill embeddings
        self.title_weight = title_w
        self.cosine_weight = cosine_w
        self.jaccard_weight = jaccard_w
        self.emp_type_weight = emp_type_w
        self.loc_type_weight = loc_type_w
        self.top_n = top_n
        self.k = k  # Scaling factor for logarithmic transformation

    def logarithmic_transform(self, similarity):
        """
        Apply logarithmic transformation to a similarity score.

        Parameters
        ----------
        similarity : float
            The raw similarity score to transform, typically in [0, 1].

        Returns
        -------
        float
            The transformed similarity score, scaled logarithmically.
        """
        return math.log(1 + self.k * similarity) / math.log(1 + self.k)

    def compute_title_similarity(self, user_subtitle, job_title):
        """
        Compute embedding-based similarity between user subtitle and job title.

        Uses the SentenceTransformer model to encode both texts and computes their
        cosine similarity, followed by logarithmic scaling.

        Parameters
        ----------
        user_subtitle : str
            The user's subtitle or job title.
        job_title : str
            The job's title.

        Returns
        -------
        float
            The logarithmically scaled cosine similarity score.
        """
        user_emb = self.model.encode([user_subtitle])[0]
        job_emb = self.model.encode([job_title])[0]
        similarity = cosine_similarity([user_emb], [job_emb])[0][0]
        return self.logarithmic_transform(similarity)

    def compute_description_similarity(self, user_about_me, job_description):
        """
        Compute embedding-based similarity between user's 'about me' and job description.

        Encodes both texts using the SentenceTransformer model, computes their cosine
        similarity, and applies logarithmic scaling.

        Parameters
        ----------
        user_about_me : str
            The user's 'about me' description.
        job_description : str
            The job's description.

        Returns
        -------
        float
            The logarithmically scaled cosine similarity score.
        """
        user_emb = self.model.encode([user_about_me])[0]
        job_emb = self.model.encode([job_description])[0]
        similarity = cosine_similarity([user_emb], [job_emb])[0][0]
        return self.logarithmic_transform(similarity)

    def compute_skill_similarity(self, user_skills, job_skills):
        """
        Compute similarity between user and job skills based on embedding similarity.

        Counts the number of job skills that have at least one user skill with a cosine
        similarity above a threshold (0.5), normalized by the number of job skills.

        Parameters
        ----------
        user_skills : list
            List of user skill strings.
        job_skills : list
            List of job skill strings.

        Returns
        -------
        float
            The normalized skill similarity score, in [0, 1].
        """
        if not user_skills or not job_skills:
            return 0.0
        user_embeddings = np.array([self.skill_embeddings.get(skill, np.zeros(384)) for skill in user_skills])
        job_embeddings = np.array([self.skill_embeddings.get(skill, np.zeros(384)) for skill in job_skills])
        sim_matrix = cosine_similarity(user_embeddings, job_embeddings)
        common_skills = sum(any(sim_matrix[i, j] > 0.5 for i in range(len(user_skills)))
                            for j in range(len(job_skills)))
        return common_skills / len(job_skills) if len(job_skills) > 0 else 0.0

    def find_score(self, user, job):
        """
        Compute the overall similarity score between a user and a job.

        Combines weighted similarities for title, description, skills, employment type,
        and location type into a final score.

        Parameters
        ----------
        user : dict
            User data dictionary containing 'subtitle', 'about_me', 'skills',
            'employment_type', and 'location_type'.
        job : dict
            Job data dictionary containing 'title', 'description', 'skills',
            'employee_type', and 'location_type'.

        Returns
        -------
        float
            The final weighted similarity score.
        """
        # Title similarity (embedding-based with logarithmic scaling)
        title_similarity = self.compute_title_similarity(user['subtitle'], job['title'])

        # Description similarity (embedding-based with logarithmic scaling)
        description_similarity = self.compute_description_similarity(user['about_me'], job['description'])

        # Skill similarity
        skill_similarity = self.compute_skill_similarity(user['skills'], job['skills'])

        # Employment and location type matching (binary)
        emp_equal = 1 if user['employment_type'] == job['employee_type'] else 0
        loc_equal = 1 if user['location_type'] == job['location_type'] else 0

        # Final weighted score
        final_score = (
            self.title_weight * title_similarity +
            self.cosine_weight * description_similarity +
            self.jaccard_weight * skill_similarity +
            self.emp_type_weight * emp_equal +
            self.loc_type_weight * loc_equal
        )
        return final_score