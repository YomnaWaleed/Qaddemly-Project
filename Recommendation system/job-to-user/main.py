#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main module for the job recommendation system.

This module serves as the entry point to preprocess data and generate job recommendations.
"""

from data_preprocessing import DataPreprocessor
from recommendation_to_user import JobRecommender


def main(input_file='test.json'):
    """
    Main function to run the job recommendation system.

    Parameters
    ----------
    input_file : str, optional
        Path to input JSON file, default is 'test.json'.

    Returns
    -------
    list
        List of dictionaries with 'id' and 'similarity_score' for recommended jobs.
    """
    # Preprocess data
    preprocessor = DataPreprocessor()
    user_df, jobs_df, skill_embeddings = preprocessor.preprocess(input_file)

    # Generate recommendations
    recommender = JobRecommender(skill_embeddings)
    recommendations = recommender.recommend(user_df, jobs_df)

    # Print recommendations
    print(recommendations)
    return recommendations


if __name__ == "__main__":
    main()