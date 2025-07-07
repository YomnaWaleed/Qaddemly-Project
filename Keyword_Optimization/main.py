import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
from pydantic import BaseModel
from typing import Any
from dotenv import load_dotenv
from groq import Groq
import os, json


# Load .env file
load_dotenv()

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

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extract text content from a PDF file."""
    try:
        # Read the PDF file
        contents = pdf_file.file.read()
        
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        
        # Extract text from each page
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    finally:
        pdf_file.file.close()

# Initialize Groq client
api_key = get_env_variable("GROQ_API_KEY")
client = Groq(api_key=api_key)

# FastAPI instance
app = FastAPI()

# Add CORS middleware to handle file uploads from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# -------------------- Input model --------------------

class KeywordOptimizationRequest(BaseModel):
    userData:dict
    resumeData: dict
    jobDescription: str

# -------------------- LLM call helper --------------------

def call_llm(sys_prompt, user_prompt: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

# -------------------- POST endpoint --------------------
@app.post("/optimize-resume/")
async def optimize_resume(request: KeywordOptimizationRequest):
    # Powerful system prompt
    sys_prompt = """
    You are an expert resume optimizer with 15+ years of experience in HR and recruitment. 
    Your task is to analyze the provided resume and job description, then provide specific, actionable recommendations 
    to optimize the resume by:
    1. Identifying and incorporating missing keywords from the job description
    2. Suggesting modifications to existing content to better align with the job requirements
    3. Highlighting transferable skills and relevant experiences
    4. Recommending sections or content that should be removed as they're not relevant
    5. Suggesting structural improvements for better readability and impact
    
    Your recommendations should be:
    - Highly specific to the job description
    - Actionable with clear examples
    - Focused on measurable achievements
    - Prioritized by importance
    - Professional and concise
    
    CRITICAL RULES:
    - NEVER suggest adding skills/experiences the candidate doesn't already have
    - Only work with the existing resume content
    - Focus on rephrasing and reorganizing, not inventing new content

    Provide your response in JSON format with the following structure:
    {
        "summary": "Brief overview of the key optimization opportunities",
        "recommendations": {
            "add": [
                {
                    "section": "section_name",
                    "content": "specific content to add",
                    "reason": "why this addition improves the resume"
                }
            ],
            "modify": [
                {
                    "section": "section_name",
                    "current_content": "existing content",
                    "suggested_change": "optimized content",
                    "reason": "why this change improves the resume"
                }
            ],
            "remove": [
                {
                    "section": "section_name",
                    "content": "content to remove",
                    "reason": "why removing this improves the resume"
                }
            ]
        },
        "keyword_analysis": {
            "missing_keywords": ["list", "of", "important", "missing", "keywords"],
            "underrepresented_keywords": ["keywords", "that", "should", "be", "emphasized", "more"]
        }
    }
    """
    
    # Powerful user prompt
    user_prompt = f"""
    RESUME DATA:
    {request.resumeData}
    
    JOB DESCRIPTION:
    {request.jobDescription}
    
    USER DATA:
    {request.userData}

    Based on the above information, please provide detailed recommendations to optimize the resume for this specific job opportunity.
    Focus particularly on:
    - Matching the technical skills and qualifications mentioned in the job description
    - Aligning with the required experience and responsibilities
    - Incorporating industry-specific terminology
    - Highlighting measurable achievements that demonstrate required competencies
    - Removing irrelevant information that doesn't support the candidate's fit for this role
    
    Provide your recommendations in the specified JSON format.
    """
    
    try:
        optimization_result = call_llm(sys_prompt, user_prompt)
        return parse_json_response(optimization_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/optimize-resume-pdf")
async def optimize_resume_pdf(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
    user_data: str = Form(...)
):
    """
    Optimize a resume PDF based on job description and user data.
    
    Parameters:
    - resume_pdf: PDF file containing the resume
    - job_description: String with the job description
    - user_data: JSON string with additional user data
    """
    try:
        # Extract text from PDF
        resume_text = extract_text_from_pdf(resume_pdf)
        
        # Parse user_data JSON string to dict
        try:
            user_data_dict = json.loads(user_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in user_data")
        
        # Create a minimal resumeData structure from the extracted text
        resume_data = {
            "raw_text": resume_text,
            # You might want to add more structure here if needed
        }
        
        # Create the request object that matches your existing endpoint
        request_data = KeywordOptimizationRequest(
            userData=user_data_dict,
            resumeData=resume_data,
            jobDescription=job_description
        )
        
        # Reuse your existing optimization logic
        return await optimize_resume(request_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# uvicorn main:app --reload --port 8006