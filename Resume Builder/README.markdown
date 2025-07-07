# Qaddemly Resume Section Generation Feature

This feature of the Qaddemly application provides FastAPI-based endpoints for generating and enhancing resume sections ("About Me" and "Skills") using the Groq API. It creates ATS-friendly resume content tailored to user profiles and job descriptions, enhancing the resume-building capabilities of Qaddemly.

## Feature Overview

This feature integrates two endpoints into the Qaddemly application to support resume creation:

- **Endpoints**:
  - `/generate-about-me`: Generates or enhances an "About Me" section as a single paragraph (\~100 words).
  - `/generate-skills`: Generates or enhances a "Skills" section as a comma-separated list (8–12 skills, 60% hard, 40% soft).
- **Input**: Accepts user data (skills, education, experiences), optional existing sections, and job descriptions for ATS optimization.
- **Output**: Produces ATS-friendly, minimal outputs tailored to job descriptions.
- **Mode Switching**: Automatically switches to "generate" mode if no existing section is provided in "enhance" mode.

## Prerequisites

- Python 3.8+
- A valid Groq API key (sign up at xAI to obtain one)

## Setup

1. **Navigate to the Qaddemly Directory**: Ensure the feature files (`app.py`, `generator.py`) are in the Qaddemly application directory:

   ```bash
   cd qaddemly
   ```

2. **Install Dependencies**: Create a virtual environment and install the required libraries:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**: Create a `.env` file in the Qaddemly root with your Groq API key:

   ```
   GROQ_API_KEY=your_valid_groq_api_key_here
   ```

4. **Prepare Sample Data**: Save sample user data as `data/test.json` for testing. Example:

   ```json
   {
       "user": {
           "id": 2,
           "first_name": "Abdo",
           "last_name": "Khattab",
           "skills": [{"id": 1, "account_id": 2, "name": "Node.js"}, ...],
           "educations": [{"id": 1, "university": "Tanta", "field_of_study": "Computer Engineering", ...}],
           "experiences": [{"id": 1, "job_title": "Java Sprint Boot Internship", ...}],
           ...
       }
   }
   ```

## Running the Feature

Start the FastAPI server to enable the resume section generation endpoints:

```bash
uvicorn app:app --host 0.0.0.1 --port 8004
```

The endpoints will be available at `http://127.0.0.1:8004`.

## Endpoints

### 1. Generate or Enhance "About Me" Section

- **Endpoint**: `POST /generate-about-me`
- **Request Body**:

  ```json
  {
      "user": {...},              // User data (required)
      "aboutMe": "Existing text", // Optional, for enhancement
      "jobDescription": "Job desc" // Optional, for ATS tailoring
  }
  ```
- **Response**:

  ```json
  {
      "about_me": "I specialize in backend development, building scalable APIs with Node.js, TypeScript, and PostgreSQL..."
  }
  ```
- **Example**:

  ```bash
  curl -X POST "http://127.0.0.1:8004/generate-about-me" \
       -H "Content-Type: application/json" \
       -d '{
             "user": '"$(cat data/test.json | jq .user)"',
             "aboutMe": "Social Media Managers oversee an organizations social media presence.",
             "jobDescription": "Seeking a Backend Developer to build APIs with Node.js and FastAPI"
           }'
  ```

### 2. Generate or Enhance "Skills" Section

- **Endpoint**: `POST /generate-skills`
- **Request Body**:

  ```json
  {
      "user": {...},              // User data (required)
      "skills": ["python", "backend"], // Optional, for enhancement
      "jobDescription": "Job desc" // Optional, for ATS tailoring
  }
  ```
- **Response**:

  ```json
  {
      "skills": "Node.js, TypeScript, PostgreSQL, Java, Spring Boot, Python, FastAPI, Backend Development, Problem-Solving, Teamwork, Communication, API Development"
  }
  ```
- **Example**:

  ```bash
  curl -X POST "http://127.0.0.1:8004/generate-skills" \
       -H "Content-Type: application/json" \
       -d '{
             "user": '"$(cat data/test.json | jq .user)"',
             "skills": ["python", "backend", "fastAPI"],
             "jobDescription": "Seeking a Backend Developer to build APIs with Node.js and FastAPI"
           }'
  ```

## Testing

1. **Ensure** `data/test.json` **exists** with valid user data.
2. **Run the API** and test endpoints using `curl` or tools like Postman.
3. **Verify outputs**:
   - "About Me": A single paragraph (\~100 words, 7 lines).
   - "Skills": A comma-separated list (8–12 skills, no extra text).

## Integration with DataPreprocessor

If Qaddemly uses a `DataPreprocessor` class, integrate the feature as follows:

```python
from data_preprocessing import DataPreprocessor
from generator import AboutMeGenerator, SkillsGenerator

preprocessor = DataPreprocessor()
user_data = {"user": {...}}  # From database or test.json
job_description = preprocessor.extract_job_description(job_data)
existing_skills = preprocessor.extract_existing_skills(user_data)
existing_about_me = preprocessor.extract_existing_about_me(user_data)

# Generate/Enhance About Me
about_me_gen = AboutMeGenerator()
about_me = about_me_gen.generate_section(user_data, "enhance", existing_about_me, job_description)

# Generate/Enhance Skills
skills_gen = SkillsGenerator()
skills = skills_gen.generate_section(user_data, "enhance", ", ".join(existing_skills) if existing_skills else None, job_description)
```

## Notes

- Ensure a valid `GROQ_API_KEY` is set in `.env`.
- The feature runs on port 8004 by default; adjust as needed in Qaddemly’s configuration.
- Outputs are ATS-friendly, tailored to job descriptions, and minimal (no extra text).

## Contributing

Submit feedback or enhancements to the Qaddemly development team to improve this feature.

## License

MIT License