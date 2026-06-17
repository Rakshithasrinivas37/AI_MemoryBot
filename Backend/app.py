import os
import sys
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chatbot.bot import Chatbot
from Evaluation.evaluate import Evaluate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_sessions = {}

class LoginInput(BaseModel):
  groq_api_key: str
  username: str
  hf_token: str

class chatInput(BaseModel):
  username: str
  user_input: str

class MemoryToDelete(BaseModel):
   username: str
   memory: str

class LogoutInput(BaseModel):
   username: str

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
    
    user_sessions[username] = Chatbot(groq_api_key, username)
    
    return JSONResponse(
        status_code=200,
        content = {
        "message": f"Welcome, {username}",
        "is_new_user": True
        }   
    )

@app.post("/chat")
async def chat(data: chatInput):
    username = data.username
    user_query = data.user_input

    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="User query cannot be empty, Please enter the query!!"
    )

    chatbot = user_sessions[username]

    response = chatbot.bot(user_query)

    # ── Auto evaluate every response ─────────────────
    evaluation_instance = Evaluate(
        api_key      = chatbot.groq_api_key,
        user_query   = user_query,
        response = response
    )

    scores = evaluation_instance.auto_evaluate()
    if scores:
        print(f"\n📊 Auto Eval → {user_query[:50]}")
        print(f"   Correctness : {scores['correctness']}/5")
        print(f"   Relevance   : {scores['relevance']}/5")
        print(f"   Completeness: {scores['completeness']}/5")
        print(f"   Reason      : {scores['reason']}")

    return JSONResponse(
        status_code=200,
        content={
        "message": response
        }
    )

@app.get("/memories")
async def memories(username: str):
   if username not in user_sessions:
        return JSONResponse(
            status_code = 404,
            content     = {
                "status" : "error",
                "message": "User not found"
            }
        )
   
   memories = user_sessions[username].get_all_memories()

   documents = [
        memory["summary"]
        for memory in memories
    ]

   return JSONResponse(
      status_code=200,
      content={
         "memories": documents
      }
   )

@app.delete("/delete_memory")
async def delete_memory(data: MemoryToDelete):
   username = data.username
   memory_to_delete = data.memory

   if username not in user_sessions:
        return JSONResponse(
            status_code = 404,
            content     = {
                "status" : "error",
                "message": "User not found"
            }
        )
   
   user_sessions[username].delete_memory(memory_to_delete)

   return JSONResponse(
      status_code = 200,
      content = {
         "status": "success",
         "message": "Memory deleted successfully"
      }
   )

@app.post("/logout")
async def logout(data: LogoutInput):
    username = data.username

    if username not in user_sessions:
        return JSONResponse(
            status_code = 404,
            content     = {
                "status" : "error",
                "message": "User not found"
            }
        )

    user_sessions[username].save_on_exit()
    del user_sessions[username]

    return JSONResponse(
        status_code = 200,
        content     = {
            "status" : "success",
            "message": f"Goodbye, {username}!"
        }
    )