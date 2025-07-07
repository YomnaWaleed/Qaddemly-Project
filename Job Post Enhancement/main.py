from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Any
from groq import Groq
import os, json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

def get_env_variable(var_name):
    """Get the environment variable or raise an exception."""
    try:
        return os.environ[var_name]
    except KeyError:
        raise EnvironmentError(f"Set the {var_name} environment variable.")

def parse_json_response(response: str) -> Any:
    """
    Parse the JSON response from the LLM.
    """
    try:
        return json.loads(response, strict=False)  # Allow trailing commas and other non-strict JSON features
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")


# Initialize the Groq client (uses GROQ_API_KEY from environment)
api_key = get_env_variable("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Pydantic input models
class JobPostInput(BaseModel):
    title: str
    description: str
    skills: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

class PromptInput(BaseModel):
    prompt: str

class FieldItem(BaseModel):
    field: str
    value: Union[str, List[str]]

def call_llm(sys_prompt,user_prompt: str,model: str = "meta-llama/llama-4-scout-17b-16e-instruct" ):

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role":"system", "content":sys_prompt},{"role": "user", "content": user_prompt}],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

@app.post("/enhance-description")
def enhance_description(input: JobPostInput):
    """
    Enhances the job description.
    Return a JSON object with a single key "description" containing the enhanced job description.
    Value of "description" can be formatted with markdown, but it should be a single string.
    """


    title = input.title
    desc = input.description
    skills = input.skills or []
    keywords = input.keywords or []

    sys_prompt = """
    - You are an expert career assistant that helps companies improve their job postings.
    - Your task is to take a job title and its current description, and rewrite the description to be:
        1. Clearer and more engaging
        2. Professionally worded and free of redundancy
        3. Well-structured and relevant to the role
    - Do not change the job title.
    - You may use the optional list of skills and keywords (if provided) to strengthen the content. Keep the output strictly limited to the enhanced job description. Do not modify or include the title. Do not return explanations or comments—just return the improved job description as plain text.
    - Your response format must be a JSON single object with a single key "description" containing the enhanced job description as a string.
    - like this: {"description": "Enhanced job description here..."}
    - The value of the "description" can be formatting with markdown, but it should be a single string.
    - Do not include the title, skills, or keywords in the output. 
    - Do not include explanations, comments, or additional fields.
    - The output should be a valid UTF8 encoded JSON.
    """

    user_promot = f"Job Title: {title}\nJob Description: {desc}\n"
    if skills:
        user_promot += f"Existing Skills: {', '.join(skills)}\n"
    if keywords:
        user_promot += f"Existing Keywords: {', '.join(keywords)}\n"
    user_promot += "Enhance the job description above to make it more detailed and engaging."

    enhanced_desc = call_llm(sys_prompt,user_promot)

    print(f"Enhanced Description: {enhanced_desc}")

    return parse_json_response(enhanced_desc)

@app.post("/enhance-or-generate-skills")
def enhance_or_generate_skills(input: JobPostInput):
    """
    Enhances or generates the list of skills for the job.
    """
    title = input.title
    desc = input.description
    skills = input.skills or []
    keywords = input.keywords or []

    sys_prompt = """
    - You are an AI assistant that helps companies identify the most relevant and competitive skills for job postings.
    - Given a job title and description, optionally with existing skills and keywords, your task is to return a list of job-related skills.
    - If a list of skills is provided, enhance and expand it with more precise or in-demand skills.  
    - If no skills are provided, generate a new list based on the title and description.
    - Return a clean JSON array of strings, each being a single skill. Do not include any explanations or extra text.
    - The output should be a JSON array of strings, like this: {"skills": ["skill1", "skill2", ...]}
    - Do not include the title, description, or keywords in the output.
    - Output should be UTF8 encoded JSON.
    """
    
    user_promot = f"Job Title: {title}\nJob Description: {desc}\n"
    if skills:
        user_promot += f"Existing Skills: {', '.join(skills)}\n"
    if keywords:
        user_promot += f"Keywords: {', '.join(keywords)}\n"
    user_promot += "Provide an enhanced or complete list of relevant skills for this job."

    skills_list = call_llm(sys_prompt,user_promot)

    return parse_json_response(skills_list)

@app.post("/enhance-or-generate-keywords")
def enhance_or_generate_keywords(input: JobPostInput):
    """
    Enhances or generates the list of keywords for the job.
    """
    title = input.title
    desc = input.description
    skills = input.skills or []
    keywords = input.keywords or []

    sys_prompt = """
    - You are a professional recruiter assistant. Your job is to extract or enhance relevant keywords for a job post.
    - Given a job title and description, optionally with a skills list or existing keywords, return a concise list of search-optimized keywords that would help candidates or job boards find this job.
    - If keywords are provided, enhance them by making them more relevant or adding related terms.  
    - If not provided, generate new keywords based on the job context.
    - make keywords Search engine-friendly terms or tags that describe this job.
    - Return a clean JSON array of strings, each being a single keyword. Do not include any explanations or extra text.
    - The output should be a JSON array of strings, like this: {"keywords": ["keyword1", "keyword2", ...]}
    - Do not include the title, description, or skills in the output.
    - Output should be UTF8 encoded JSON.
    """

    user_prompt = f"Job Title: {title}\nJob Description: {desc}\n"
    if skills:
        user_prompt += f"Skills: {', '.join(skills)}\n"
    if keywords:
        user_prompt += f"Existing Keywords: {', '.join(keywords)}\n"
    user_prompt += "Provide an enhanced or complete list of relevant keywords for this job."


    keywords_list = call_llm(sys_prompt,user_prompt)

    return parse_json_response(keywords_list)

@app.post("/generate-job-from-prompt")
def generate_job_from_prompt(input: PromptInput):
    """
    Generates a full job post (title, description, skills, keywords) from the prompt.
    """

    sys_prompt = """
    - You are a professional job structuring assistant that transforms raw job descriptions into structured fields suitable for a job posting system.
    - You will receive a free-form job prompt, which may be informal or incomplete. Your task is to extract and generate the following structured fields from the prompt:
    1. title: A clear, concise, and professional job title based on the role.
    2. description: A well-written, grammatically correct job description in full sentences. It should highlight responsibilities, qualifications, and any benefits or perks.
    3. skills: A list of 5 to 10 relevant technical and soft skills as strings.
    4. keywords: A list of keywords or tags (used for search engine optimization and job board matching) that summarize the role.
    - The output should be :
            {
            "title": "string",
            "description": "string",
            "skills": ["string", "string", ...],
            "keywords": ["string", "string", ...]
            }
    - Do not include any additional text, explanations, or comments in the output.
    - Ensure that the output is a valid JSON in UTF8 with all fields filled in.
    """

    user_prompt = input.prompt


    output = call_llm(sys_prompt,user_prompt)
    
    print("Generated Job Post:", output)

    return parse_json_response(output)


# uvicorn main:app --reload --port 8002