# api/model1_api.py

from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequestModel1, MatchRequest
from core.inference_model1 import (
    load_model as load_model1,
    predict_match as predict1_internal,
)
from core.data_processing_model1 import preprocess, make_sequence
from pathlib import Path
import torch, pandas as pd

app = FastAPI(title="Modelo 1 – API")

# 1) Carga artefactos al arrancar
CFG1_PATH   = Path("model/model1_config.json")
WEIGHTS1    = Path("model/model1.pth")
DATA1_CSV   = Path("data/calendario_futbol_completo.csv")

model1, cfg1 = load_model1(CFG1_PATH, WEIGHTS1)
df1, history1, team2idx = preprocess(str(DATA1_CSV))
STEPS = cfg1.get("time_steps")

# 2) Endpoints
@app.get("/")
def root():
    return {"status": "ok", "model": "Modelo 1", "device": str(model1.device)}

@app.post("/match_predict")
def match_predict1(req: MatchRequest):
    try:
        res = predict1_internal(
            model=model1,
            local_team=req.local,
            visitor_team=req.visitor,
            date=req.date,
            history=history1,
            team2idx=team2idx,
            steps=STEPS 
        )
        winner = "local" if res[f"{req.local.lower()}_win"] > res[f"{req.visitor.lower()}_win"] else "visitor"
        return {
            "winner":       winner,
            "prob_local":   res[f"{req.local.lower()}_win"],
            "prob_visitor": res[f"{req.visitor.lower()}_win"],
            "confidence":   max(res.values())
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
