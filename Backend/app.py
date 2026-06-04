import os
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginInput(BaseModel):
  groq_api_key: str
  username: str
  hf_token: str

@app.post("/login")
async def login(data: LoginInput):
  
    groq_api_key = data.groq_api_key
    username = data.username
    hf_token = data.hf_token

    # Store as environment variable
    os.environ["HF_TOKEN"] = hf_token

    if not groq_api_key:
        raise HTTPException(
            status_code=400,
            detail="API Key cannot be empty, Please provide API Key!!"
        )

    if not username:
        raise HTTPException(
            status_code=400,
            detail="Username cannot be empty, Please enter username!!"
        )
    
    return JSONResponse(
        status_code=200,
        content = {
        "message": f"Welcome, {username}",
        "is_new_user": True
        }   
    )

