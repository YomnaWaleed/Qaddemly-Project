# Job Post Enhancement API

This FastAPI-based application provides endpoints to enhance and structure job posts using the Groq LLM service. It can:

- Enhance job descriptions  
- Generate or enhance job-related skills  
- Generate or enhance SEO keywords  
- Generate full job posts from prompts

## 🧠 Features

- Uses Groq LLM (e.g., LLaMA) to enhance or structure job content  
- JSON-based structured output  
- Accepts optional fields like skills and keywords  
- Easily extendable with new endpoints

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

You must also create a `.env` file with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Running the Server

```bash
uvicorn main:app --reload --port 8002
```

Access the API docs at: [http://localhost:8002/docs](http://localhost:8002/docs)

## 📮 API Endpoints

### `POST /enhance-description`

Enhances a job description using a given title, optional skills and keywords.

#### Body Example:

```json
{
  "title": "Software Engineer",
  "description": "We are looking for a software engineer...",
  "skills": ["Python", "FastAPI"],
  "keywords": ["backend", "developer"]
}
```

---

### `POST /enhance-or-generate-skills`

Returns enhanced or generated skills based on the job post.

---

### `POST /enhance-or-generate-keywords`

Returns SEO keywords based on the job post.

---

### `POST /generate-job-from-prompt`

Generates a structured job post from a free-form prompt.

#### Body Example:

```json
{
  "prompt": "Looking for a backend engineer to build scalable APIs using Python."
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
