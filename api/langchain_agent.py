# api/langchain_agent.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.agent import football_agent, chat_with_agent

class ChatRequest(BaseModel):
    input: str

app = FastAPI(title="Football LangChain Agent")

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        resp = chat_with_agent(req.input, agent_executor=football_agent)
        return {"response": resp}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
