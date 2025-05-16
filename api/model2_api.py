# api/model2_api.py
# --------------------------------------------------------
# FastAPI para el “Modelo 2” (LSTM + attention + embeddings + cuotas)
# --------------------------------------------------------
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException

from api.schemas import MatchRequest, PredictRequestModel2
from core.inference_model2 import (
    load_runtime_objects,       # carga pesos, cfg y artefactos en un solo paso
    predict_match as predict2,  # wrapper amigable
)

# ------------------------------------------------------------------
# 1)  Carga de pesos + artefactos
# ------------------------------------------------------------------
MODEL_DIR       = Path("model")            #  model/
ARTEFACTS_PKL   = "artefacts.pkl"
HISTORY_PARQUET = "history_df.parquet"

runtime: Dict[str, Any] = load_runtime_objects(
    model_dir       = MODEL_DIR,
    artefacts_pkl   = ARTEFACTS_PKL,
    history_parquet = HISTORY_PARQUET,
)

model2            = runtime["model"]
cfg2              = runtime["cfg"]
history_df        = runtime["history_df"]
equipos_dict      = runtime["equipos_dict"]
arb_dict          = runtime["arb_dict"]
comp_dict         = runtime["comp_dict"]
players_name2idx  = runtime["players_name2idx"]
PAD_IDX           = runtime["PAD_PLAYER_IDX"]
impute_means      = runtime["impute_means"]
DEVICE            = runtime["device"]

MAX_PLAYERS = cfg2.get("max_players", 11)  # por si no estuviera en el JSON

# ------------------------------------------------------------------
# 2)  Definición de la API
# ------------------------------------------------------------------
app = FastAPI(title="Modelo 2 – API", version="1.0")

@app.get("/")
def root():
    return {"status": "ok", "model": "Modelo 2", "device": str(DEVICE)}

# ---------- endpoint bajo nivel: tensores explícitos ----------------
@app.post("/predict")
def predict(req: PredictRequestModel2):
    try:
        # a Tensores y al mismo dispositivo que el modelo
        seq_l = torch.tensor(req.seq_loc, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        seq_v = torch.tensor(req.seq_vis, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        idx_l = torch.tensor([req.idx_loc], dtype=torch.long).to(DEVICE)
        idx_v = torch.tensor([req.idx_vis], dtype=torch.long).to(DEVICE)
        idx_r = torch.tensor([req.idx_ref], dtype=torch.long).to(DEVICE)
        idx_c = torch.tensor([req.idx_cmp], dtype=torch.long).to(DEVICE)
        lp    = torch.tensor([req.lineup_loc], dtype=torch.long).to(DEVICE)
        vp    = torch.tensor([req.lineup_vis], dtype=torch.long).to(DEVICE)
        odds  = torch.tensor([req.odds], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model2(
                seq_l, seq_v,
                idx_l, idx_v,
                idx_r, idx_c,
                lp, vp,
                odds,
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        return {
            "visitor":   float(probs[0]),
            "local":     float(probs[1]),
            "draw":      float(probs[2]),
            "confidence": float(probs.max()),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- endpoint alto nivel: nombres / cuotas / fecha -----------
@app.post("/match_predict")
def match_predict(req: MatchRequest):
    try:
        probs = predict2(
            # --- modelo + cfg + histórico ---
            model             = model2,
            cfg               = cfg2,
            history_df        = history_df,
            # --- artefactos ---
            equipos_dict      = equipos_dict,
            players_name2idx  = players_name2idx,
            PAD_PLAYER_IDX    = PAD_IDX,
            max_players       = MAX_PLAYERS,
            impute_means      = impute_means,
            # --- datos del partido ---
            local             = req.local,
            visitor           = req.visitor,
            date              = req.date,
            local_players     = req.local_players,
            visitor_players   = req.visitor_players,
            referee           = req.referee,
            competition       = req.competition,
            cuota_1           = req.cuota_1,
            cuota_x           = req.cuota_x,
            cuota_2           = req.cuota_2,
        )

        winner = max(probs, key=probs.get)
        return {
            "winner":       winner,
            "prob_local":   probs.get(req.local.title(), 0.0),
            "prob_visitor": probs.get(req.visitor.title(), 0.0),
            "prob_draw":    probs.get("Empate", 0.0),
            "confidence":   float(probs[winner]),
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
