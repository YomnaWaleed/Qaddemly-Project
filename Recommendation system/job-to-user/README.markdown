# Qaddemly Job Recommendation System

## Overview
The Qaddemly Job Recommendation System is a Python-based application designed to recommend users for a specific job based on similarity metrics such as skills, job title, description, employment type, and location type. The system processes user and job data, generates embeddings for skills, and computes similarity scores to identify the best-suited candidates for a job, enabling job owners to notify potential applicants. The application includes a FastAPI endpoint for integration into web services and a command-line interface for local execution.

## Features
- **Data Preprocessing**: Cleans and processes user and job data, including text normalization, skill parsing, and embedding generation using `sentence-transformers`.
- **User Recommendation**: Recommends users for a job based on weighted similarity metrics (title, description, skills, employment type, location type).
- **API Endpoint**: Provides a `/recommend-users` FastAPI endpoint to handle JSON requests with multiple users and a single job.
- **Command-Line Interface**: Allows local execution via `main.py` for testing and development.
- **ATS-Friendly**: Outputs minimal, structured data (user IDs and similarity scores) suitable for Applicant Tracking Systems.

## Project Structure
```
qaddemly/
├── app.py                  # FastAPI application with /recommend-users endpoint
├── data_preprocessing.py   # Data preprocessing and embedding generation
├── recommendation_to_job.py# User recommendation logic
├── main.py                # Command-line interface for recommendations
├── data/
│   └── test.json          # Sample input JSON file
├── .env                   # Environment variables (e.g., GROQ_API_KEY)
└── requirements.txt       # Python dependencies
```

## Setup Instructions
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd qaddemly
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv qaddemly_env
   source qaddemly_env/bin/activate  # Linux/Mac
   .\qaddemly_env\Scripts\activate   # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. **Set Environment Variables**:
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_valid_groq_api_key_here
   ```

6. **Prepare Input Data**:
   Create a `data/test.json` file with the following structure:
   ```json
   {
       "users": [
           {
               "id": 2,
               "country": "Egypt",
               "city": "Tanta",
               "about_me": "Passionate and goal-driven software engineer...",
               "subtitle": "Backend Developer | AI Enthusiast | Problem Solver",
               "skills": [{"name": "Node.js"}, {"name": "Backend"}, {"name": "postgres"}, {"name": "java"}, {"name": "software engineer"}, {"name": "TypeScript"}, {"name": "Spring boot"}],
               "educations": [...],
               "experiences": [...]
           },
           {...}
       ],
       "job": {
           "id": 28746,
           "title": "Digital Marketing Specialist",
           "description": "Social Media Managers oversee...",
           "skills": ["Social media platforms (e.g., Facebook, Twitter, Instagram)", ...],
           "location_type": "Onsite",
           "employee_type": "Internship",
           ...
       }
   }
   ```

## Usage
### Running the API
1. Start the FastAPI server:
   ```bash
   uvicorn app:app --host 0.0.0.1 --port 8005
   ```
2. Test the `/recommend-users` endpoint:
   ```bash
   curl -X POST "http://127.0.0.1:8005/recommend-users" \
        -H "Content-Type: application/json" \
        -d @data/test.json
   ```
   **Expected Output**:
   ```json
   [
       {"id": 2, "similarity_score": 0.3},
       {"id": 2, "similarity_score": 0.3}
   ]
   ```

### Running Locally
1. Execute the command-line interface:
   ```bash
   python main.py
   ```
   This processes `data/test.json` and prints recommendations to the console.

## File Descriptions
- **data_preprocessing.py**:
  - Contains the `DataPreprocessor` class for cleaning and processing user and job data.
  - Parses skills, preprocesses text (lowercasing, lemmatizing, removing stop words), and generates skill embeddings using `sentence-transformers`.
  - Saves preprocessed DataFrames (`users_df.pkl`, `job_df.pkl`) and embeddings (`skill_embeddings.pkl`).

- **recommendation_to_job.py**:
  - Contains the `UserRecommender` class for recommending users for a job.
  - Computes similarity scores based on title, description, skills, employment type, and location type, using TF-IDF for descriptions and cosine similarity for skills.

- **app.py**:
  - FastAPI application with the `/recommend-users` endpoint.
  - Accepts a JSON request with a list of users and a single job, returning recommended users with similarity scores.
  - Integrates with `DataPreprocessor` and `UserRecommender`.

- **main.py**:
  - Command-line interface for running recommendations locally.
  - Loads input from a JSON file, preprocesses data, and generates recommendations.

## Dependencies
See `requirements.txt` for a complete list of dependencies. Key libraries include:
- `fastapi`: For the API server.
- `uvicorn`: ASGI server for running FastAPI.
- `pydantic`: For request validation.
- `numpy`, `pandas`, `scikit-learn`: For data processing and similarity calculations.
- `nltk`: For text preprocessing.
- `sentence-transformers`: For skill embeddings.

## Notes
- The system expects a single job and multiple users in the input JSON.
- Similarity scores may be low if user skills (e.g., Node.js, Java) do not match job requirements (e.g., social media skills).
- The API runs on port 8005 by default, adjustable in `app.py`.
- Ensure the `GROQ_API_KEY` is set for any Groq-related features (e.g., resume generation integration).

## Troubleshooting
- **Dependency Errors**: If you encounter errors like `numpy.ufunc size changed`, ensure `requirements.txt` versions are used.
- **NLTK Data**: Run `nltk.download()` commands if preprocessing fails due to missing data.
- **Input Validation**: Ensure `test.json` matches the expected structure to avoid 400 errors.