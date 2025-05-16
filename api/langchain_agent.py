# api/langchain_agent.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.agent import chat_with_agent

app = FastAPI(title="LangChain Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    input: str

@app.get("/")
def root():
    return {"status": "ok", "agent": "LangChain Football Agent"}

@app.post("/predict")
def predict(req: AgentRequest):
    # Llamamos a tu función que invoca al agente
    try:
        reply = chat_with_agent(req.input)
        return {"response": reply}
    except Exception as e:
        return {"error": str(e)}