# core/data_processing_model1.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple

# ------------------------------------------------------------------
# 1) pre-proceso global (solo equipo, goal_diff, is_home)
# ------------------------------------------------------------------
def preprocess(csv_path: str):
    """
    Lee el CSV completo de partidos y construye:
      • df  ............. DataFrame original depurado
      • history_df ...... tabla de historial por equipo (ordenada)
      • team2idx ........ dict nombre→idx
    """
    df = pd.read_csv(csv_path)
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df.dropna(subset=["fecha"], inplace=True)

    df["equipo_local"]    = df["equipo_local"].str.lower().str.strip()
    df["equipo_visitante"] = df["equipo_visitante"].str.lower().str.strip()

    teams = sorted(set(df["equipo_local"]) | set(df["equipo_visitante"]))
    team2idx = {t: i for i, t in enumerate(teams)}
    df["local_idx"]   = df["equipo_local"].map(team2idx)
    df["visitor_idx"] = df["equipo_visitante"].map(team2idx)

    # historial “flat” 1 fila ↔ 1 equipo-partido
    rows = []
    for _, r in df.iterrows():
        rows.append(
            dict(
                fecha=r["fecha"],
                equipo_idx=r["local_idx"],
                goal_diff=r["goles_local"] - r["goles_visitante"],
                is_home=1,
            )
        )
        rows.append(
            dict(
                fecha=r["fecha"],
                equipo_idx=r["visitor_idx"],
                goal_diff=r["goles_visitante"] - r["goles_local"],
                is_home=0,
            )
        )
    history_df = pd.DataFrame(rows).sort_values(["equipo_idx", "fecha"])

    return df, history_df, team2idx


# ------------------------------------------------------------------
# 2) genera la secuencia de un equipo
# ------------------------------------------------------------------
def make_sequence(
    history_df: pd.DataFrame,
    equipo_idx: int,
    fecha,
    steps: int,
    feature_dim: int = 2,
):
    """
    Devuelve un np.ndarray shape=(steps, feature_dim) con
    0-padding al principio si faltan partidos.
    """
    filt = history_df[
        (history_df.equipo_idx == equipo_idx) & (history_df.fecha < fecha)
    ]
    seq = (
        filt.sort_values("fecha", ascending=False)
        .head(steps)[["goal_diff", "is_home"]]
        .values
    )
    if seq.shape[0] < steps:
        pad = np.zeros((steps - seq.shape[0], feature_dim), dtype=np.float32)
        seq = np.vstack([pad, seq])
    return seq.astype(np.float32)
