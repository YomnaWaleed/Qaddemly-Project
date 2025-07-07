from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from generator import AboutMeGenerator, SkillsGenerator

app = FastAPI()

class ResumeSectionRequest(BaseModel):
    """
    Request model for generating or enhancing resume sections ('About Me' or 'Skills').

    Attributes
    ----------
    user : dict
        User data containing profile information (skills, educations, experiences, etc.).
    aboutMe : str or None
        Existing 'About Me' section to enhance (optional, for /generate-about-me).
    skills : list or None
        Existing skills list to enhance (optional, for /generate-skills).
    jobDescription : str or None
        Job description to tailor the section for ATS compatibility (optional).
    """
    user: dict
    aboutMe: str | None = None
    skills: list | None = None
    jobDescription: str | None = None

@app.post("/generate-about-me")
async def generate_about_me(request: ResumeSectionRequest):
    """
    Generate or enhance an 'About Me' resume section.

    Parameters
    ----------
    request : ResumeSectionRequest
        Request body containing user data, optional existing 'About Me' section,
        and optional job description.

    Returns
    -------
    dict
        A dictionary containing the generated or enhanced 'About Me' section
        (e.g., {"about_me": "Generated paragraph"}).

    Raises
    ------
    HTTPException
        400 if input validation fails (e.g., missing user data).
        500 if Groq API call fails.
    """
    try:
        generator = AboutMeGenerator()
        mode = "enhance" if request.aboutMe else "generate"
        result = generator.generate_section(
            user_json={"user": request.user},
            mode=mode,
            existing_section=request.aboutMe,
            job_description=request.jobDescription
        )
        return {"about_me": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate-skills")
async def generate_skills(request: ResumeSectionRequest):
    """
    Generate or enhance a 'Skills' resume section.

    Parameters
    ----------
    request : ResumeSectionRequest
        Request body containing user data, optional existing skills list,
        and optional job description.

    Returns
    -------
    dict
        A dictionary containing the generated or enhanced 'Skills' section
        (e.g., {"skills": "Node.js, TypeScript, ..."}).

    Raises
    ------
    HTTPException
        400 if input validation fails (e.g., missing user data).
        500 if Groq API call fails.
    """
    try:
        generator = SkillsGenerator()
        existing_section = ", ".join(request.skills) if request.skills else None
        mode = "enhance" if existing_section else "generate"
        result = generator.generate_section(
            user_json={"user": request.user},
            mode=mode,
            existing_section=existing_section,
            job_description=request.jobDescription
        ).split(', ')
        if result[-1][-1] == '.':
            result[-1] = result[-1][:-1]
        return {"skills": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)