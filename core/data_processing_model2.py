# core/data_processing_model2.py
# ---------------------------------------------------------------------
# Pre-procesado y utilidades para el Modelo 2 (con cuotas + alineaciones)
# ---------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Si tienes tqdm instalado mostrará barras de progreso; si no, no pasa nada
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, *args, **kw):  # type: ignore
        return x


# ---------------------------------------------------------------------
# 1)  Carga y pre-procesado de los datos
# ---------------------------------------------------------------------
def load_and_preprocess_data(
    main_fp: str,
    players_fp: str,
    max_players: int = 11,
) -> Tuple[
    pd.DataFrame,                # df ya procesado
    Dict[str, int],              # equipos_dict
    Dict[str, int] | None,       # arb_dict (o None si no hay árbitros)
    Dict[str, int] | None,       # comp_dict (o None si no hay competiciones)
    Dict[str, int],              # players_id2idx
    Dict[str, int],              # players_name2idx
    int,                         # PAD_PLAYER_IDX
    int,                         # NUM_PLAYERS (sin el PAD extra)
    Dict[str, float],            # medias para imputar cuotas
]:
    """
    Lee los dos CSV, genera diccionarios y columnas auxiliares y devuelve
    todos los artefactos que después necesita la API / inference_model2.
    """
    # ---------- 1.1  jugadores.csv → diccionarios ----------
    df_players = pd.read_csv(players_fp)
    df_players["jugador"] = df_players["jugador"].str.lower().str.strip()
    df_players["id"] = df_players["id"].astype(str).str.strip()

    players_id2idx: Dict[str, int] = {
        pid: i for i, pid in enumerate(df_players["id"].unique())
    }
    players_name2idx: Dict[str, int] = {}
    for _, row in df_players.iterrows():
        players_name2idx.setdefault(row["jugador"], players_id2idx[row["id"]])

    num_total_players = len(players_id2idx)
    PAD_PLAYER_IDX = num_total_players  # índice reservado para el PAD

    # ---------- 1.2  dataset de partidos ----------
    df = pd.read_csv(main_fp)
    df["fecha"] = pd.to_datetime(df["fecha"])

    # etiqueta 1-X-2
    cond = [
        df["goles_local"] > df["goles_visitante"],
        df["goles_local"] < df["goles_visitante"],
        df["goles_local"] == df["goles_visitante"],
    ]
    df["target"] = np.select(cond, [1, 0, 2]).astype(int)

    # diccionario equipos
    teams = pd.concat([df["equipo_local"], df["equipo_visitante"]]).str.lower().unique()
    equipos_dict = {t: i for i, t in enumerate(teams)}
    df["local_idx"] = df["equipo_local"].str.lower().map(equipos_dict)
    df["visitor_idx"] = df["equipo_visitante"].str.lower().map(equipos_dict)

    # árbitros (si existe la columna)
    if "arbitro" in df.columns:
        df["arbitro"] = df["arbitro"].str.lower().str.strip()
        arb_dict = {a: i for i, a in enumerate(df["arbitro"].unique())}
        df["arbitro_idx"] = df["arbitro"].map(arb_dict)
    else:
        arb_dict = None
        df["arbitro_idx"] = 0

    # competiciones (si existe la columna)
    if "competicion" in df.columns:
        df["competicion"] = df["competicion"].str.lower().str.strip()
        comp_dict = {c: i for i, c in enumerate(df["competicion"].unique())}
        df["competicion_idx"] = df["competicion"].map(comp_dict)
    else:
        comp_dict = None
        df["competicion_idx"] = 0

    # ---------- 1.3  Cuotas → probabilidades implícitas ----------
    cuota_cols = ["Cuota 1", "Cuota X", "Cuota 2"]
    prob_cols = ["prob_cuota_1", "prob_cuota_x", "prob_cuota_2"]

    for c_src, c_dst in zip(cuota_cols, prob_cols):
        df[c_dst] = 1 / pd.to_numeric(df.get(c_src, np.nan), errors="coerce")
        df.loc[(df[c_dst] <= 0) | (df[c_dst] > 1) | pd.isna(df[c_dst]), c_dst] = np.nan

    impute_means = {c: df[c].mean() for c in prob_cols}
    for c in prob_cols:
        df[c] = df[c].fillna(impute_means[c])

    # ---------- 1.4  Alineaciones → listas fijas de índices ----------
    def _parse_ids(s: str | float) -> List[str]:
        if pd.isna(s):
            return []
        return [x.strip() for x in str(s).split(",")]

    # pre-crear columnas para velocidad
    df["local_lineup_indices"] = [[]] * len(df)
    df["visitor_lineup_indices"] = [[]] * len(df)

    for i, r in tqdm(df.iterrows(), total=len(df), desc="Alineaciones"):
        loc_ids = [
            players_id2idx[x] for x in _parse_ids(r.get("titulares_local_ids", ""))
            if x in players_id2idx
        ]
        vis_ids = [
            players_id2idx[x] for x in _parse_ids(r.get("titulares_visitante_ids", ""))
            if x in players_id2idx
        ]

        df.at[i, "local_lineup_indices"] = (
            loc_ids[:max_players] + [PAD_PLAYER_IDX] * max_players
        )[:max_players]
        df.at[i, "visitor_lineup_indices"] = (
            vis_ids[:max_players] + [PAD_PLAYER_IDX] * max_players
        )[:max_players]

    return (
        df,
        equipos_dict,
        arb_dict,
        comp_dict,
        players_id2idx,
        players_name2idx,
        PAD_PLAYER_IDX,
        num_total_players,
        impute_means,
    )


# ---------------------------------------------------------------------
# 2)  Historial temporal (por equipo)
# ---------------------------------------------------------------------
def create_team_history_features(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con un registro por *equipo-partido* que luego se
    usa para construir las secuencias temporales en inferencia.
    """
    records: List[dict] = []

    for _, r in tqdm(df_matches.iterrows(), total=len(df_matches), desc="Historial"):
        # local
        records.append(
            {
                "match_id": r["match_id"],
                "fecha": r["fecha"],
                "team_idx": r["local_idx"],
                "opponent_idx": r["visitor_idx"],
                "is_home": 1,
                "goles_marcados": r["goles_local"],
                "goles_concedidos": r["goles_visitante"],
                "tarjetas_amarillas": r.get("cantidad_tarjetas_amarillas_local", 0),
            }
        )
        # visitante
        records.append(
            {
                "match_id": r["match_id"],
                "fecha": r["fecha"],
                "team_idx": r["visitor_idx"],
                "opponent_idx": r["local_idx"],
                "is_home": 0,
                "goles_marcados": r["goles_visitante"],
                "goles_concedidos": r["goles_local"],
                "tarjetas_amarillas": r.get("cantidad_tarjetas_amarillas_visita", 0),
            }
        )

    h = pd.DataFrame(records)
    h["goal_diff"] = h["goles_marcados"] - h["goles_concedidos"]
    return h.sort_values(["team_idx", "fecha"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# 3)  Helper para obtener la secuencia de un equipo (lo usa inference)
# ---------------------------------------------------------------------
def get_sequence_for_team(
    hist_df: pd.DataFrame,
    team_idx: int,
    match_date,
    time_steps: int,
    feat_dim: int,
) -> np.ndarray:
    """
    Devuelve una matriz (time_steps, feat_dim) con padding al principio si
    el equipo tiene menos de *time_steps* partidos antes de *match_date*.
    """
    filt = hist_df[(hist_df.team_idx == team_idx) & (hist_df.fecha < match_date)]
    seq = (
        filt.sort_values("fecha", ascending=False)
        .head(time_steps)[
            [
                "goal_diff",
                "is_home",
                "goles_marcados",
                "goles_concedidos",
                "tarjetas_amarillas",
            ]
        ]
        .values
    )

    if seq.shape[0] < time_steps:
        pad = np.zeros((time_steps - seq.shape[0], feat_dim), dtype=np.float32)
        seq = np.vstack([pad, seq])

    return seq.astype(np.float32)

# -----------------------------------------------------------------
# Alias retro-compatible para inference_model2
# -----------------------------------------------------------------
def make_sequence(
    history_df,
    team_idx: int,
    date,
    steps: int,
    feat_dim: int = 5,
):
    """wrapper requerido por inference_model2 (simple alias)."""
    return get_sequence_for_team(
        hist_df=history_df,
        team_idx=team_idx,
        match_date=date,
        time_steps=steps,
        feat_dim=feat_dim,
    )