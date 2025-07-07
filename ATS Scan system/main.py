from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import process_data
import uvicorn
from pyngrok import ngrok
import nest_asyncio

app = FastAPI()

class InputData(BaseModel):
    success: bool
    job_application_state: dict
    job: dict
    jobApplication: list

@app.post("/process")
async def process_json(input_data: InputData):
    try:
        # Convert Pydantic model to dict
        input_dict = input_data.dict()
        
        # Process the data
        results = process_data(input_dict)
        
        return {
            "success": True,
            "results": results,
            "job_application_id": input_dict['job_application_state']['job_application_id']
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Setup ngrok
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    
    # Allow nested asyncio for running in notebook environments
    nest_asyncio.apply()
    
    # Run FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)