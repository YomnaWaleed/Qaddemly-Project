from data_preprocessing import DataPreprocessor
from matching_score import MatchingScore
import pickle

# Create DataPreprocessor instance (loads model once)
preprocessor = DataPreprocessor()

# Preprocess data and generate skill embeddings
user, job = preprocessor.preprocess('very-good.json')

# Load precomputed skill embeddings from local file
with open('skill_embeddings.pkl', 'rb') as f:
    skill_embeddings = pickle.load(f)

# Create MatchingScore instance with preloaded model and embeddings
matcher = MatchingScore(preprocessor.model, skill_embeddings)

# Compute similarity score
score = matcher.find_score(user, job)
print("Similarity Score:", score)