
# Resume Optimization API

This FastAPI-based application provides an AI-powered resume optimization service using Groq LLMs.

## 💡 Features

- Upload resumes as PDF or structured JSON
- Analyze and enhance resume content based on a job description
- Provide keyword analysis (missing and underrepresented)
- Recommend additions, modifications, or removals
- Full JSON-based response for structured usage

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

You must also create a `.env` file with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Running the Server

```bash
uvicorn main:app --reload --port 8006
```

Access the API docs at: [http://localhost:8006/docs](http://localhost:8006/docs)

## 📮 API Endpoints

### `POST /optimize-resume/`

Optimize resume using structured JSON input.

**Body Example:**

```json
{
  "userData": { "name": "John Doe", "email": "john@example.com" },
  "resumeData": { "raw_text": "Experienced software engineer..." },
  "jobDescription": "We are hiring a backend Python developer..."
}
```

---

### `POST /optimize-resume-pdf`

Upload a PDF resume along with a job description and user data.

**Form Fields:**

- `resume_pdf`: The uploaded PDF file
- `job_description`: The job description text
- `user_data`: A JSON string with user data

---

## 📊 Response Format

All endpoints return structured JSON:

```json
{
  "summary": "...",
  "recommendations": {
    "add": [...],
    "modify": [...],
    "remove": [...]
  },
  "keyword_analysis": {
    "missing_keywords": [...],
    "underrepresented_keywords": [...]
  }
}
```

---

## 🔐 Environment Variables

Ensure your `.env` file contains:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📄 License

This project is for educational/demo purposes.
