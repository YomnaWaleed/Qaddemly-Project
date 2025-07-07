# Recommendation System for Companies

## Overview

The Recommendation System for Companies enables organizations to identify and notify suitable candidates for specific job openings. Given a job listing and a pool of user profiles, the system recommends users whose skills, experience, and preferences align with the job requirements. This content-based filtering approach uses NLP and similarity metrics to evaluate user skills, professional backgrounds, job titles, and work preferences, facilitating targeted candidate outreach.

## Components

The system is implemented across four Python modules:

- **`data_preprocessing.py`**: Preprocesses job and user data by cleaning text, parsing skills, extracting recent user experiences, and generating skill embeddings using a SentenceTransformer model. Produces structured Pandas DataFrames and embeddings.
- **`recommendation_to_user.py`**: Implements the `JobRecommender` class, adapted to recommend users for a given job by computing similarity scores based on skills, title alignment, description similarity, and employment/location preferences.
- **`app.py`**: Provides a FastAPI-based REST API to accept job and user data via HTTP requests and return recommended users, supporting integration with company systems.
- **`main.py`**: Offers a command-line interface to test the recommendation pipeline with a JSON input file.

## Setup

### Prerequisites

- **Python**: Version 3.8 or higher.
- **Dependencies**: Install required libraries using pip:
  ```bash
  pip install numpy pandas nltk sentence-transformers scikit-learn fastapi uvicorn pydantic
  ```
- **NLTK Data**: Download necessary NLTK resources:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('punkt')
  nltk.download('wordnet')
  ```

### Input Data

The system requires input in JSON format, containing a job listing and a list of user profiles. Save the input as `test.json` in the project directory. Example structure:

```json
{
  "job": {
    "id": 101,
    "title": "Machine Learning Engineer",
    "description": "Develop advanced AI models",
    "skills": ["Python", "TensorFlow"],
    "employee_type": "Full-time",
    "location_type": "Remote"
  },
  "users": [
    {
      "id": 1,
      "about_me": "Passionate software engineer with expertise in AI",
      "subtitle": "Software Engineer",
      "skills": [{"name": "Python"}, {"name": "Machine Learning"}],
      "experiences": [
        {
          "job_title": "Developer",
          "employment_type": "Full-time",
          "location_type": "Remote",
          "start_date": "2023-06-01"
        }
      ],
      "educations": []
    },
    {
      "id": 2,
      "about_me": "Creative writer with a knack for storytelling",
      "subtitle": "Content Writer",
      "skills": [{"name": "Writing"}, {"name": "SEO"}],
      "experiences": [
        {
          "job_title": "Writer",
          "employment_type": "Part-time",
          "location_type": "On-site",
          "start_date": "2022-01-01"
        }
      ],
      "educations": []
    }
  ]
}
```

## Usage

### Command-Line Interface

To generate user recommendations via the command line:
1. Ensure all Python files and `test.json` are in the same directory.
2. Modify `main.py` to process the job-to-user recommendation flow (e.g., swap user and job inputs in `recommendation_to_user.py`).
3. Run the main script:
   ```bash
   python main.py
   ```
4. The system will output a list of recommended users with their IDs and similarity scores.

### API Interface

To use the API:
1. Update `app.py` to handle job-to-user recommendation input (e.g., adjust the `InputData` model to accept `job` and `users`).
2. Start the FastAPI server:
   ```bash
   python app.py
   ```
3. The server will run on `http://localhost:8001`.
4. Send a POST request to the `/recommend` endpoint with the JSON input:
   ```bash
   curl -X POST "http://localhost:8001/recommend" -H "Content-Type: application/json" -d @test.json
   ```
5. The response will include a list of recommended users and a success message.

## Technical Details

The recommendation process involves the following steps:

1. **Preprocessing** (`data_preprocessing.py`):
   - Cleans text by removing special characters, lowercasing, and lemmatizing.
   - Parses user and job skills, extracting the most recent user experience.
   - Generates skill embeddings using the `all-MiniLM-L6-v2` SentenceTransformer model.
   - Outputs Pandas DataFrames and embeddings.

2. **Recommendation** (`recommendation_to_user.py`):
   - Adapted to recommend users for a job by computing:
     - **Title Similarity**: Binary match between job title and user subtitle.
     - **Description Similarity**: TF-IDF-based cosine similarity between job description and user’s “about me”.
     - **Skill Similarity**: Counts common skills using embedding-based cosine similarity (threshold: 0.5).
     - **Employment/Location**: Binary matching for employment and location types.
   - Combines similarities using weights (default: 0.2 each) to rank users.
   - Returns the top `top_n` users by similarity score.

3. **API and CLI** (`app.py`, `main.py`):
   - `app.py` (with modifications) exposes a `/recommend` endpoint for job-to-user recommendations.
   - `main.py` (with modifications) supports command-line testing.

## Troubleshooting

- **NLTK Errors**: Verify that `stopwords`, `punkt`, and `wordnet` are downloaded.
- **JSON Format Issues**: Ensure `test.json` includes `job` and `users` fields with the correct structure.
- **API Errors**: Confirm port `8001` is available and the JSON payload aligns with the modified `InputData` model.
- **Adaptation Needs**: Update `recommendation_to_user.py` and `app.py` to handle job-to-user logic if not already implemented.

## Potential Enhancements

- Implement notification integration for recommended users.
- Add filters for user availability or experience level.
- Optimize embedding generation for large user pools.
- Develop a dashboard for companies to review recommendations.

## License

This feature is licensed under the MIT License, permitting use, modification, and distribution with attribution.