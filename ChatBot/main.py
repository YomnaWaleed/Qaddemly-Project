from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import List, Dict
from pyngrok import ngrok
from bot_runner import run_qaddemly_bot

# Start ngrok tunnel
#public_url = ngrok.connect(8000)
#print(f"FastAPI ngrok public URL: {public_url}")

app = FastAPI()


class BotRequest(BaseModel):
    question: str
    user_type: str
    user_data: Dict  


@app.post("/qaddemly-bot")
def ask_bot(request: BotRequest):
    return run_qaddemly_bot(
        question=request.question,
        user_type=request.user_type,
        user_data=request.user_data,  
    )
