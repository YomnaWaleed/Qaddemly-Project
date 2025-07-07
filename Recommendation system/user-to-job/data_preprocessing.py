#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing module for job recommendation system.

This module handles the preprocessing of user and job data for the recommendation system,
including text cleaning, skill parsing, and embedding generation.
"""

import numpy as np
import pandas as pd
import json
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

from psutil import users
from sentence_transformers import SentenceTransformer


class DataPreprocessor:
    """
    A class for preprocessing job and user data.

    Handles the cleaning, parsing, and vectorization of job and user data
    for use in recommendation algorithms.

    Attributes
    ----------
    stop_words : set
        Set of English stop words from NLTK
    """

    def __init__(self):
        """
        Initialize the DataPreprocessor with stop words.

        Loads stopwords from a local pickle file if available,
        otherwise downloads them from NLTK and saves for future use.
        """
        self.lemmatizer = WordNetLemmatizer()

        # Try to load stopwords from a local pickle file
        try:
            import os
            import pickle

            stopwords_file = 'nltk_stopwords.pkl'
            if os.path.exists(stopwords_file):
                with open(stopwords_file, 'rb') as f:
                    self.stop_words = pickle.load(f)
                print("Loaded stopwords from local file")
            else:
                # If file doesn't exist, download and save
                try:
                    self.stop_words = set(stopwords.words('english'))
                except LookupError:
                    nltk.download('stopwords')
                    self.stop_words = set(stopwords.words('english'))

                # Save for future use
                with open(stopwords_file, 'wb') as f:
                    pickle.dump(self.stop_words, f)
                print("Downloaded stopwords and saved to local file")
        except Exception as e:
            # Fallback to direct download if any error occurs
            print(f"Error handling stopwords file: {e}")
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords')
                self.stop_words = set(stopwords.words('english'))

    def get_user_skills(self, skill_dicts):
        skills = []
        for skill_dict in skill_dicts:
            skills.append(skill_dict['name'])
        return skills

    def parse_skills(self, skills_list):
        """
        Parse a list of skill strings into a list of skill phrases.

        Parameters
        ----------
        skills_list : list
            List of strings containing concatenated skills

        Returns
        -------
        list
            List of parsed skill phrases
        """

        def clean_string(skill_list):
            clean_list = []
            for s in skill_list:
                clean_list.append(re.sub(r'\s*\(.*?\)', '', s))
            return clean_list

        def standardize_skill(skill):
            skill = skill.lower()
            skill = re.sub(r'[^\w\s]', '', skill)
            return skill.strip()

        def parse_skill_list(skills_list):
            skills = []
            for skills_string in skills_list:
                words = skills_string.split()
                words = [word for word in words if word not in self.stop_words]

                current_skill = []
                prev_starts_with_capital = False

                for word in words:
                    if word:
                        starts_with_capital = word[0].isupper()
                        if starts_with_capital:
                            skill_phrase = ' '.join(current_skill)
                            if skill_phrase:
                                std_skill_phrase = standardize_skill(skill_phrase)
                                skills.append(std_skill_phrase)
                            current_skill = [word]
                        else:
                            current_skill.append(word)
                        prev_starts_with_capital = starts_with_capital

                if current_skill:
                    skill_phrase = ' '.join(current_skill)
                    if skill_phrase:
                        std_skill_phrase = standardize_skill(skill_phrase)
                        skills.append(std_skill_phrase)

            return skills

        cleaned = clean_string(skills_list)
        parsed_skills = parse_skill_list(cleaned)

        return parsed_skills

    def preprocess_text(self, text):
        """
        Preprocess text by removing special characters, lowercasing, and lemmatizing.

        Parameters
        ----------
        text : str, list, or dict
            Text to be preprocessed

        Returns
        -------
        str, list, or dict
            Preprocessed text
        """

        def preprocess_string(text_str):
            text_str = re.sub(r"[^a-zA-Z0-9 _-]", "", text_str)
            text_str = text_str.lower().strip()
            words = word_tokenize(text_str)
            filtered_words = [self.lemmatizer.lemmatize(word) for word in words if word.lower() not in self.stop_words]
            return " ".join(filtered_words)

        if isinstance(text, str):
            return preprocess_string(text)
        elif isinstance(text, list):
            return [self.preprocess_text(item) for item in text]
        elif isinstance(text, dict):
            return {self.preprocess_text(key) if isinstance(key, str) else key: self.preprocess_text(value) if isinstance(value, str) else value
                    for key, value in text.items()}
        return text

    def get_last_experience(self, experience_dict):
        """
        Get the most recent experience from user data.

        Parameters
        ----------
        df : pandas.DataFrame
            User data dataframe

        Returns
        -------
        dict
            The most recent experience
        """
        start_date = ""
        last_exp = None
        for job in experience_dict:
            if job['start_date'] > start_date:
                start_date = job['start_date']
                last_exp = job

        return last_exp

    def preprocess_object(self, users_df, job_df):
        """
        Preprocess object columns in dataframes.

        Parameters
        ----------
        users_df : pandas.DataFrame
            User data dataframe
        job_df : pandas.DataFrame
            Jobs data dataframe

        Returns
        -------
        tuple
            (preprocessed_users_df, preprocessed_job_df)
        """
        for col in job_df.columns:
            if job_df[col].dtype == 'object':
                job_df[col] = job_df[col].apply(self.preprocess_text)

        for col in users_df.columns:
            if users_df[col].dtype == 'object':
                users_df[col] = users_df[col].apply(self.preprocess_text)

        # user_last_experience = self.get_last_experience(users_df)
        users_df['last_experience'] = users_df['experiences'].apply(self.get_last_experience)
        # Extract location_type and employment_type from last_experience
        users_df['location_type'] = users_df['last_experience'].apply(
            lambda x: x['location_type'] if x and 'location_type' in x else ''
        )
        users_df['employment_type'] = users_df['last_experience'].apply(
            lambda x: x['employment_type'] if x and 'employment_type' in x else ''
        )

        return users_df, job_df

    def preprocess_user_skills(self, skills):
        """
        Extract skill values from user skills data.

        Parameters
        ----------
        skills : list
            List of skill dictionaries

        Returns
        -------
        list
            List of skill values
        """
        return [value for d in skills for value in d.values()]

    def clean_and_tokenize(self, text):
        """
        Clean and tokenize text.

        Parameters
        ----------
        text : str
            Text to be cleaned and tokenized

        Returns
        -------
        list
            List of cleaned tokens
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return words

    def get_top_10_words(self, description):
        """
        Get the top 10 most common words in a description.

        Parameters
        ----------
        description : str
            Text description

        Returns
        -------
        list
            List of top 10 words
        """
        words = self.clean_and_tokenize(description)
        word_counts = Counter(words)
        top_10 = word_counts.most_common(10)
        return [word for word, count in top_10]

    def precompute_skill_embeddings(self, users_df, job_df):
        """
        Precompute embeddings for all skills in the dataset.

        Parameters
        ----------
        users_df : pandas.DataFrame
            User data dataframe
        job_df : pandas.DataFrame
            Jobs data dataframe

        Returns
        -------
        dict
            Dictionary mapping skills to embeddings
        """
        model = SentenceTransformer('all-MiniLM-L6-v2')
        all_skills = set()
        for skills in job_df['skills']:
            all_skills.update(skills)
        for skills in users_df['skills']:
            all_skills.update(skills)

        all_skills = list(all_skills)
        skill_embeddings = model.encode(all_skills, batch_size=128, show_progress_bar=True)
        skill_to_embedding = dict(zip(all_skills, skill_embeddings))
        with open('skill_embeddings.pkl', 'wb') as f:
            pickle.dump(skill_to_embedding, f)
        return skill_to_embedding

    def preprocess(self, input_data = 'test.json'):
        """
        Main preprocessing function to process input JSON data.

        Parameters
        ----------
        input_json : str, optional
            Path to input JSON file, by default 'inputJson.json'

        Returns
        -------
        tuple
            (users_df, job_df) preprocessed dataframes
        """
        # with open(input_data, 'r') as f:
        #     input_data = json.load(f)

        users_df = pd.DataFrame([user for user in input_data['users']])
        job_df = pd.DataFrame([input_data['job']])

        users_df = users_df.drop_duplicates(subset=['about_me'], keep='first').reset_index(drop=True)

        users_df['skills'] = users_df['skills'].apply(self.get_user_skills)
        job_df['skills'] = job_df['skills'].apply(self.parse_skills)

        users_df, job_df = self.preprocess_object(users_df, job_df)

        job_df['top_10_words'] = job_df['description'].apply(self.get_top_10_words)
        users_df['top_10_words'] = users_df['about_me'].apply(self.get_top_10_words)

        self.precompute_skill_embeddings(users_df, job_df)

        job_df.to_pickle('job_df.pkl')
        users_df.to_pickle('users_df.pkl')

        return users_df, job_df


def preprocess(input_data):
    """
    Preprocess the input JSON data.

    Parameters
    ----------
    input_json : str, optional
        Path to input JSON file, by default 'inputJson.json'

    Returns
    -------
    tuple
        (users_df, job_df) Preprocessed dataframes
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(input_data)