from data_preprocessing import DataPreprocessor
from recommendation_to_job import UserRecommender

def main(input_file='test.json'):
    """
    Main function to run the job recommendation system.

    Parameters
    ----------
    input_file : str, optional
        Path to input JSON file, by default 'inputJson.json'

    Returns
    -------
    list
        List of recommended job dictionaries
    """
    # Preprocess data
    preprocessor = DataPreprocessor()
    users_df, job_df = preprocessor.preprocess(input_file)
    # Generate recommendations
    Recommender = UserRecommender()
    recommendations = Recommender.recommend(job_df, users_df)
    print(recommendations)

    return recommendations


if __name__ == "__main__":
    main()