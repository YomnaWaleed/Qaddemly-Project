import json
import re
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file!")

client = Groq(api_key=GROQ_API_KEY)

def enhance_cover_letter(user_job_json: dict) -> str:
    user = user_job_json["user"]
    user_skills = user.get("skills", [])
    user_certifications = user.get("certificates", [])
    user_education = user.get("educations", [])
    user_experience = user.get("experiences", [])
    user_projects = user.get("projects", [])
    job_desc = user_job_json.get("jobDescription", "")
    #print(f"Job Description: {job_desc}")
    cover_letter = user_job_json.get("existingBody", "")
    #print(f"Existing Cover Letter: {cover_letter}")

    prompt = f"""
You are a professional career advisor specializing in technical roles. Your task is to enhance — not rewrite — an existing backend developer cover letter by making it more concise, polished, and compelling. The applicant wants minimal but meaningful improvements.

 **Instructions**:
1. Maintain the original tone, structure, and voice of the candidate.
2. Only refine phrasing, clarity, and flow to improve readability and professionalism.
3. Ensure 3 clear paragraphs (opening, body, closing), each between 40–60 words.
4. Emphasize backend-related achievements and strong technical impact where appropriate.

 **Enhance using (but do not inject unnaturally)**:
- **Skills**: {', '.join([s['name'] for s in user_skills]) or 'Not provided'}
- **Certifications**: {', '.join([c['name'] for c in user_certifications]) or 'None'}
- **Education**: {f"{user_education[0]['field_of_study']} at {user_education[0]['university']} (GPA: {user_education[0]['gpa']})" if user_education else 'Not provided'}
- **Experience**: {', '.join([f"{e['job_title']} at {e['company_name']}" for e in user_experience]) or 'None'}
- **Projects**: {', '.join([p['name'] for p in user_projects if p.get('is_featured')]) or 'None'}
- **Job Description Focus**: {job_desc[:150]}...

 **Current Cover Letter Body**:
{cover_letter}

 **Output Requirements**:
Return ONLY the optimized version of the existing cover letter as plain text. Preserve paragraph breaks (single newlines). Do not generate new content from scratch — just improve what’s already there.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()



if __name__ == "__main__":
    # Define the path to the JSON input file
    input_file_path = "data/testwithBody.json"

    # Check if file exists
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file '{input_file_path}' not found!")

    # Load the JSON data
    with open(input_file_path, "r", encoding="utf-8") as file:
        user_job_json = json.load(file)

    try:
        # Generate the cover letter
        result = enhance_cover_letter(user_job_json)

        # Print the enhanced cover letter
        print("\enhanced Cover Letter:\n")
        print(result)

    except Exception as e:
        print(f"Error: {e}")
