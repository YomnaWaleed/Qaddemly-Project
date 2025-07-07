# Score Matching Feature

## Overview
The Score Matching feature calculates a similarity score between a user’s profile and a job posting on the JobApp website. It uses a SentenceTransformer model (`all-MiniLM-L6-v2`) to generate embeddings for skills, job titles, and descriptions, combining these with binary matching for employment and location types. The feature employs weighted scoring and logarithmic scaling to produce a final match score, accessible via a REST API or Python script.

## Dependencies
- Python 3.9+
- `sentence-transformers` 2.2.2
- `transformers` 4.35.0
- `fastapi` 0.95.0
- `uvicorn` 0.20.0
- `nltk` 3.8.1
- `numpy` 1.24.3
- `pandas` 2.0.3
- `scikit-learn` 1.2.2
- `psutil` 5.9.5
- `pydantic` 1.10.9
- `PyPDF2` 3.0.1 (optional, for resume parsing in related features)

Install dependencies:
```bash
pip install -r requirements.txt
```

### NLTK Data
The feature requires NLTK data for text preprocessing:
- `stopwords`
- `punkt`
- `wordnet`

Download NLTK data:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

## Setup Instructions
1. Navigate to the feature directory:
   ```bash
   cd ai-features/Score_Matching
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK data (if not already cached):
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
   ```
4. (Optional) Precompute skill embeddings for sample data:
   ```bash
   python main.py
   ```
   This generates `skill_embeddings.pkl` using the `very-good.json` sample data.
5. Start the FastAPI server for API access:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8003 --reload
   ```

## Usage
### API Endpoint
- **Endpoint**: `POST /match-score`
- **Input**: JSON payload with `user` and `job` objects (see schema below)
- **Output**: JSON with `similarity_score` (0-1) and `message`

**Input Schema**:
```json
{
  "success": true,
  "user": {
    "id": 1,
    "country": "USA",
    "city": "New York",
    "about_me": "Experienced software engineer with a passion for AI.",
    "subtitle": "Senior Software Engineer",
    "skills": [{"name": "Python"}, {"name": "Machine Learning"}],
    "educations": [{"university": "MIT", "field_of_study": "Computer Science", "gpa": 3.8}],
    "experiences": [
      {
        "job_title": "Software Engineer",
        "employment_type": "Full-time",
        "company_name": "Tech Corp",
        "location": "New York",
        "location_type": "On-site",
        "start_date": "2020-01-01"
      }
    ]
  },
  "job": {
    "id": 1,
    "title": "AI Engineer",
    "description": "Develop machine learning models for job matching.",
    "country": "USA",
    "city": "New York",
    "location_type": "On-site",
    "employee_type": "Full-time",
    "skills": ["Python", "Machine Learning", "Deep Learning"]
  }
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8003/match-score \
-H "Content-Type: application/json" \
-d @sample_input.json
```

**Example Response**:
```json
{
  "similarity_score": 0.85,
  "message": "Matching score calculated successfully"
}
```

### Python Usage
Run the matching score calculation directly:
```python
from data_preprocessing import DataPreprocessor
from matching_score import MatchingScore
import pickle

# Load sample data
input_data = {
    "user": {
        "id": 1,
        "about_me": "Experienced software engineer with a passion for AI.",
        "subtitle": "Senior Software Engineer",
        "skills": [{"name": "Python"}, {"name": "Machine Learning"}],
        "experiences": [{"employment_type": "Full-time", "location_type": "On-site", "start_date": "2020-01-01"}]
    },
    "job": {
        "id": 1,
        "title": "AI Engineer",
        "description": "Develop machine learning models for job matching.",
        "skills": ["Python", "Machine Learning", "Deep Learning"],
        "employee_type": "Full-time",
        "location_type": "On-site"
    }
}

# Preprocess data
preprocessor = DataPreprocessor()
user, job = preprocessor.preprocess(input_data)

# Load skill embeddings
with open('skill_embeddings.pkl', 'rb') as f:
    skill_embeddings = pickle.load(f)

# Calculate score
scorer = MatchingScore(preprocessor.model, skill_embeddings)
score = scorer.find_score(user, job)
print(f"Similarity Score: {score}")
```

## Model Details
- **Architecture**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Purpose**: Generates embeddings for job titles, descriptions, and skills to compute cosine similarity.
- **Scoring Components**:
  - **Title Similarity**: Cosine similarity between user subtitle and job title (log-scaled, weight: 0.2).
  - **Description Similarity**: Cosine similarity between user’s “about me” and job description (log-scaled, weight: 0.2).
  - **Skill Similarity**: Normalized count of job skills with cosine similarity > 0.5 to user skills (weight: 0.2).
  - **Employment Type Match**: Binary match (1 or 0, weight: 0.2).
  - **Location Type Match**: Binary match (1 or 0, weight: 0.2).
- **Output**: Weighted sum of components, range [0, 1].
- **Limitations**:
  - Supports English text only.
  - Skill matching may miss niche or synonymous skills due to embedding limitations.
  - Assumes preprocessed input data conforms to expected schema.

## Directory Structure
```
Score_Matching/
├── app.py                # FastAPI server for match score API
├── data_preprocessing.py # Data preprocessing and embedding generation
├── main.py               # Script to compute score for sample data
├── matching_score.py     # Matching score calculation logic
├── requirements.txt      # Python dependencies
├── skill_embeddings.pkl  # Precomputed skill embeddings (gitignored)
└── tests/                # Unit tests (not provided, placeholder)
    ├── test_preprocessing.py
    ├── test_matching.py
```

## Testing
Unit tests are located in the `tests/` directory (not provided). To run tests:
```bash
pytest tests/
```

Create tests to validate:
- Input preprocessing (e.g., skill parsing, text cleaning).
- Matching score accuracy for sample user-job pairs.

## Contributing
- Follow PEP 8 for Python code.
- Submit issues or PRs to [GitHub Issues](https://github.com/your-repo/jobapp/issues).
- Ensure tests pass before submitting PRs.

## License
MIT License (same as root project).

## Contact
Refer to the root `README.md` or contact the project maintainer at maintainer@jobapp.com.