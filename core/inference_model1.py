# core/inference_model1.py
import json
from pathlib import Path
from typing import Dict

import torch
import pandas as pd

from .model1_arch import FootballPredictorSimple
from .data_processing_model1 import make_sequence

# ------------------------------------------------------------
def load_model(cfg_path: Path, weights_path: Path, device=None):
    """
    Carga el Modelo 1 en modo inferencia + devuelve el cfg dict.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = FootballPredictorSimple(
        num_teams      = cfg["num_teams"],
        feature_dim    = cfg["feature_dim"],
        team_embed_dim = cfg["team_embed_dim"],
        convlstm_hid   = cfg["convlstm_hid"],
        convlstm_kernel= tuple(cfg["convlstm_kernel"]),
        fc_hid         = cfg["fc_hid"],
        dropout_p      = cfg["drop"],
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.device = device           # comodín útil
    return model, cfg


# ------------------------------------------------------------
def predict_match(
    *,
    model: torch.nn.Module,
    local_team: str,
    visitor_team: str,
    date: str,
    history: pd.DataFrame,
    team2idx: Dict[str, int],
    steps: int,
) -> Dict[str, float]:
    """
    Wrapper de inferencia que devuelve:
      { '<local>_win': prob,  '<visitor>_win': prob }
    """
    local = local_team.lower().strip()
    visitor = visitor_team.lower().strip()
    if local not in team2idx or visitor not in team2idx:
        raise ValueError("Equipo desconocido")

    dt = pd.to_datetime(date)
    seq_l = make_sequence(history, team2idx[local], dt, steps)
    seq_v = make_sequence(history, team2idx[visitor], dt, steps)

    with torch.no_grad():
        logits = model(
            torch.tensor(seq_l).unsqueeze(0).to(model.device),
            torch.tensor(seq_v).unsqueeze(0).to(model.device),
            torch.tensor([team2idx[local]]).to(model.device),
            torch.tensor([team2idx[visitor]]).to(model.device),
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    return {f"{local}_win": float(probs[1]), f"{visitor}_win": float(probs[0])}
