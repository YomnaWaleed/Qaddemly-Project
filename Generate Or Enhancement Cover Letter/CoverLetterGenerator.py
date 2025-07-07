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




def generate_cover_letter_parts(user_job_json: dict) -> tuple:
    user = user_job_json["user"]
    user_skills = user_job_json["user"]["skills"]
    user_certifications = user_job_json["user"]["certificates"]
    user_education = user_job_json["user"]["educations"]
    user_experience = user_job_json["user"]["experiences"]
    user_projects = user_job_json["user"]["projects"]
    job_desc = user_job_json["jobDescription"]
    prompt = f"""
You are an expert career advisor and professional cover letter writer.

Your task is to generate a concise, tailored cover letter for a backend developer role based on the structured JSON data provided below. Create three focused and polished paragraphs:

---

1. **Opening Paragraph (40–60 words)**:
- Mention how the applicant found the job (assume via Qaddemly website).
- Express enthusiasm for the position and company.
- Reference the specific role, such as Backend Developer.

2. **Body Paragraph (40–60 words)**:
- Summarize relevant skills: {', '.join([skill['name'] for skill in user_skills])}
- Mention notable projects: {', '.join([proj['name'] for proj in user_projects if proj.get('is_featured', False)]) or 'None'}
- Highlight impactful experience: {', '.join([f"{exp['job_title']} at {exp['company_name']}" for exp in user_experience])}
- Include certifications if relevant: {', '.join([cert['name'] for cert in user_certifications]) or 'None'}
- Incorporate quantifiable or outcome-driven achievements if possible.

3. **Closing Paragraph (40–60 words)**:
- Do **not** include any email address, phone number, or contact details.
- Do **not** include any email address, phone number, or contact details.
- Express availability and eagerness to contribute.
- Politely invite the reader to discuss next steps.
- Do **not** include any email address, phone number, or contact details.

---

**Job Description Summary**:
{job_desc}

**User Education**: {user_education[0]['field_of_study']} at {user_education[0]['university']} (GPA: {user_education[0]['gpa']}) if available.

---

**Guidelines**:
- Maintain a natural, confident tone.
- Avoid buzzwords, exaggeration, or clichés.
- Keep paragraphs under 60 words each.
- Do not include section labels like “Opening Paragraph.”
- Do **not** mention contact details like phone number or email under any circumstance.

Output only the final cover letter text with three paragraphs separated by single newlines.
"""

    # Request to Groq API to generate the cover letter
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=600,
    )

    content = response.choices[0].message.content.strip()

    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # Remove the first line if it's an intro like "Here's a cover letter..."
    if "cover letter" in paragraphs[0].lower():
        paragraphs = paragraphs[1:]

    # Remove any bolded headers like "**Opening Paragraph**"
    cleaned_paragraphs = [re.sub(r"\*\*.*?\*\*", "", p).strip() for p in paragraphs]

    # Ensure we still have exactly 3 paragraphs
    if len(cleaned_paragraphs) != 3:
        raise ValueError(
            "The cleaned cover letter doesn't contain all required sections."
        )

    # Merge them into one string
    final_text = "\n".join(cleaned_paragraphs)

    return cleaned_paragraphs, final_text


if __name__ == "__main__":
    # Define the path to the JSON input file
    input_file_path = "data/testwithoutBody.json"

    # Check if file exists
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file '{input_file_path}' not found!")

    # Load the JSON data
    with open(input_file_path, "r", encoding="utf-8") as file:
        user_job_json = json.load(file)

    try:
        # Generate the cover letter
        result, text = generate_cover_letter_parts(user_job_json)

        # Print the generated cover letter
        #print("\nGenerated Cover Letter:\n")
        #print(result)
        print("text : \n")
        print(text)

    except Exception as e:
        print(f"Error: {e}")