from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from CoverLetterGenerator import generate_cover_letter_parts
from CoverLetterEnhance import enhance_cover_letter

app = FastAPI()


class UserJobRequest(BaseModel):
    user: Dict
    jobDescription: str
    existingBody: str = ""  # Optional field for existing cover letter body

@app.post("/generate-enhance-cover-letter")
def generate_or_enhancement_cover_letter(request: UserJobRequest):
    try:
        if request.existingBody.strip() == "":
            cleaned_paragraphs, final_text = generate_cover_letter_parts(request.dict())
            return {"result": final_text}
        else:
            enhanced_cover_letter = enhance_cover_letter(request.dict())
            return {"result": enhanced_cover_letter}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ =="__main__":
    import uvicorn
    from pyngrok import ngrok
    
    public_url = ngrok.connect(8000)
    print(f" * public ngrok URL : { public_url}" )
    
    uvicorn.run(app, host="0.0.0.0", port = 8000) 