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

@app.post("/predict")
def predict1(req: PredictRequestModel1):
    try:
        # convertir listas a tensores
        seq_l = torch.tensor(req.seq_loc, dtype=torch.float32).unsqueeze(0).to(model1.device)
        seq_v = torch.tensor(req.seq_vis, dtype=torch.float32).unsqueeze(0).to(model1.device)
        idx_l = torch.tensor([req.idx_loc], dtype=torch.long).to(model1.device)
        idx_v = torch.tensor([req.idx_vis], dtype=torch.long).to(model1.device)

        with torch.no_grad():
            logits = model1(seq_l.unsqueeze(-1).unsqueeze(-1),
                            seq_v.unsqueeze(-1).unsqueeze(-1),
                            idx_l, idx_v)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        return {
            "visitor": float(probs[0]),
            "local":   float(probs[1]),
            "confidence": float(probs.max())
        }
    except Exception as e:
        raise HTTPException(400, str(e))

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
