"""
utils/preprocessing.py

Funciones para cargar el histórico de partidos y generar secuencias
de entradas para el modelo.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def preprocess(csv_path: str):
    """
    Carga CSV con columnas [fecha, equipo_local, equipo_visitante,
    goles_local, goles_visitante], normaliza nombres y crea un DataFrame
    histórico con una fila por equipo y partido.
    """
    df = pd.read_csv(csv_path)
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
    df.dropna(subset=['fecha'], inplace=True)

    df['equipo_local']     = df['equipo_local'].str.lower().str.strip()
    df['equipo_visitante'] = df['equipo_visitante'].str.lower().str.strip()

    teams   = sorted(set(df['equipo_local']) | set(df['equipo_visitante']))
    team2idx= {t:i for i,t in enumerate(teams)}

    rows = []
    for _, r in df.iterrows():
        diff = r['goles_local'] - r['goles_visitante']
        rows.append({
            'fecha'      : r['fecha'],
            'equipo_idx' : team2idx[r['equipo_local']],
            'goal_diff'  : diff,
            'is_home'    : 1
        })
        rows.append({
            'fecha'      : r['fecha'],
            'equipo_idx' : team2idx[r['equipo_visitante']],
            'goal_diff'  : -diff,
            'is_home'    : 0
        })

    history = pd.DataFrame(rows)
    history.sort_values(['equipo_idx','fecha'], inplace=True)
    return history, team2idx

def make_sequence(history: pd.DataFrame, idx: int, date: datetime,
                  steps: int = 10, feature_dim: int = 2) -> np.ndarray:
    """
    Construye una secuencia de longitud `steps` con [goal_diff, is_home]
    para `idx` antes de `date`. Rellena con ceros si faltan partidos.
    """
    past = history[(history.equipo_idx == idx) & (history.fecha < date)]
    seq  = past.sort_values('fecha', ascending=False) \
                .head(steps)[['goal_diff','is_home']].values
    pad  = steps - seq.shape[0]
    if pad > 0:
        zeros = np.zeros((pad, feature_dim), dtype=np.float32)
        seq = np.vstack([zeros, seq.astype(np.float32)])
    return seq.astype(np.float32)
