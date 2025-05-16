# ────────────────────────────────────────────────────────────────
#  core/inference_model2.py
# ────────────────────────────────────────────────────────────────
"""
Inference wrapper para el *Modelo 2*.

Requisitos en disco (carpeta «model/» o la que uses):

• best_model_odds.pth      –  pesos entrenados (state-dict de PyTorch)
• model_config.json        –  hiper-parámetros del modelo
• artefacts.pkl            –  diccionarios, medias de imputación, etc.
• history_df.parquet       –  historial procesado de partidos

Todo es cargado por `load_runtime_objects()` y después consumido
por `predict_match()`, que es la única función que tu FastAPI debe
importar.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch import Tensor

# ────────────────────────────
#  Rutas por defecto
# ────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _THIS_DIR / ".." / "model"          # <- ajusta si procede

# ────────────────────────────
#  ── 1. Arquitectura ────────────────────────────────────────────
# ────────────────────────────
#  (idéntica a la usada en entrenamiento; no toques nada aquí)
class FootballPredictionModel(torch.nn.Module):
    def __init__(
        self,
        num_teams: int,
        team_dim: int,
        num_arb: int,
        arb_dim: int,
        num_comp: int,
        comp_dim: int,
        num_players: int,
        player_dim: int,
        pad_idx: int,
        seq_in: int,
        lstm_hid: int,
        num_odds: int,
        fc_hid: int,
        drop: float,
    ):
        super().__init__()

        self.team_emb   = torch.nn.Embedding(num_teams, team_dim)
        self.arb_emb    = torch.nn.Embedding(num_arb or 1, arb_dim)  # si 0 ⇒ dim 1 “dummy”
        self.comp_emb   = torch.nn.Embedding(num_comp or 1, comp_dim)
        self.player_emb = torch.nn.Embedding(num_players + 1, player_dim, padding_idx=pad_idx)
        self.pad_idx = pad_idx

        self.lstm_loc = torch.nn.LSTM(
            seq_in, lstm_hid, num_layers=2,
            batch_first=True, bidirectional=True, dropout=drop
        )
        self.lstm_vis = torch.nn.LSTM(
            seq_in, lstm_hid, num_layers=2,
            batch_first=True, bidirectional=True, dropout=drop
        )
        self.att_loc = torch.nn.Sequential(torch.nn.Linear(2 * lstm_hid, 1), torch.nn.Softmax(dim=1))
        self.att_vis = torch.nn.Sequential(torch.nn.Linear(2 * lstm_hid, 1), torch.nn.Softmax(dim=1))

        comb_dim = (
            2 * (2 * lstm_hid)      # contextos LSTM
            + 2 * team_dim
            + (arb_dim if num_arb else 0)
            + (comp_dim if num_comp else 0)
            + 2 * player_dim
            + num_odds
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(comb_dim, fc_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop),
            torch.nn.Linear(fc_hid, fc_hid // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop),
            torch.nn.Linear(fc_hid // 2, 3),        # 3 salidas → [Visitor, Local, Draw]
        )

    # ---------- forward ----------
    def forward(
        self,
        loc_seq: Tensor,
        vis_seq: Tensor,
        loc_i: Tensor,
        vis_i: Tensor,
        arb_i: Tensor,
        comp_i: Tensor,
        lp: Tensor,
        vp: Tensor,
        odds: Tensor,
    ) -> Tensor:
        # -- embeddings
        te_loc, te_vis = self.team_emb(loc_i), self.team_emb(vis_i)
        arb_e = self.arb_emb(arb_i) if arb_i.numel() else torch.zeros_like(te_loc)
        comp_e = self.comp_emb(comp_i) if comp_i.numel() else torch.zeros_like(te_loc)

        # -- LSTM + atención
        lo, _ = self.lstm_loc(loc_seq)
        vo, _ = self.lstm_vis(vis_seq)
        ctx_lo = (lo * self.att_loc(lo)).sum(1)
        ctx_vo = (vo * self.att_vis(vo)).sum(1)

        # -- jugadores (mask padding)
        ple, vpe = self.player_emb(lp), self.player_emb(vp)
        mask_lp = (lp != self.pad_idx).unsqueeze(-1).float()
        mask_vp = (vp != self.pad_idx).unsqueeze(-1).float()
        agg_lp = (ple * mask_lp).sum(1) / mask_lp.sum(1).clamp(min=1)
        agg_vp = (vpe * mask_vp).sum(1) / mask_vp.sum(1).clamp(min=1)

        # -- concatenación final
        feat = torch.cat([ctx_lo, ctx_vo, te_loc, te_vis, arb_e, comp_e, agg_lp, agg_vp, odds], dim=1)
        return self.fc(feat)


# ────────────────────────────
#  ── 2. Helper interno ──────────────────────────────────────────
# ────────────────────────────
def _internal_predict(
    *,
    model: FootballPredictionModel,
    history_df: pd.DataFrame,
    local_idx: int,
    visitor_idx: int,
    match_date_ts: pd.Timestamp,
    arbitro_idx: int,
    competicion_idx: int,
    local_player_idx: List[int],
    visitor_player_idx: List[int],
    odds_probs: List[float],
    feature_dim: int,
    time_steps: int,
    device: torch.device,
) -> List[float]:
    """
    Hace la forward-pass real y devuelve [p_visitante, p_local, p_empate].
    """
    from .data_processing_model2 import get_sequence_for_team  # evita ciclo de import

    # 1) secuencias históricas
    lseq = get_sequence_for_team(history_df, local_idx, match_date_ts, time_steps, feature_dim)
    vseq = get_sequence_for_team(history_df, visitor_idx, match_date_ts, time_steps, feature_dim)

    # 2) empaquetar tensores
    batch = [
        torch.tensor(lseq).unsqueeze(0).float(),
        torch.tensor(vseq).unsqueeze(0).float(),
        torch.tensor([local_idx]).long(),
        torch.tensor([visitor_idx]).long(),
        torch.tensor([arbitro_idx]).long(),
        torch.tensor([competicion_idx]).long(),
        torch.tensor([local_player_idx]).long(),
        torch.tensor([visitor_player_idx]).long(),
        torch.tensor([odds_probs]).float(),
    ]
    batch = [t.to(device) for t in batch]

    # 3) inferencia
    model.eval()
    with torch.no_grad():
        logits = model(*batch).squeeze()
        probs = torch.softmax(logits, dim=0).cpu().tolist()

    return probs  # [visit, local, draw]


# ────────────────────────────
#  ── 3.  Carga de runtime ───────────────────────────────────────
# ────────────────────────────
def _load_model(cfg_path: Path, weights_path: Path, device: torch.device) -> Tuple[FootballPredictionModel, Dict[str, Any]]:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    model = FootballPredictionModel(
        num_teams=cfg["num_teams"],
        team_dim=cfg["team_dim"],
        num_arb=cfg["num_arb"],
        arb_dim=cfg["arb_dim"],
        num_comp=cfg["num_comp"],
        comp_dim=cfg["comp_dim"],
        num_players=cfg["num_players"],
        player_dim=cfg["player_dim"],
        pad_idx=cfg["pad_idx"],
        seq_in=cfg["seq_in"],
        lstm_hid=cfg["lstm_hid"],
        num_odds=cfg["num_odds"],
        fc_hid=cfg["fc_hid"],
        drop=cfg["drop"],
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def load_runtime_objects(
    model_dir: Path | None = None,
    artefacts_pkl: str = "artefacts.pkl",
    history_parquet: str = "history_df.parquet",
) -> Dict[str, Any]:
    """
    Carga pesos, config y todos los *artefacts* (diccionarios, medias, historial…)
    y los devuelve en un solo dict para inyección de dependencias en la API.
    """
    model_dir = model_dir or _MODEL_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- modelo
    model, cfg = _load_model(
        cfg_path=model_dir / "model5_config.json",
        weights_path=model_dir / "model5.pth",
        device=device,
    )

    # -- artefacts
    with open(model_dir / artefacts_pkl, "rb") as f:
        art: Dict[str, Any] = pickle.load(f)

    history_df = pd.read_parquet(model_dir / history_parquet)

    return dict(
        model=model,
        cfg=cfg,
        history_df=history_df,
        device=device,
        **art,  # equipos_dict, arb_dict, comp_dict, players_name2idx, PAD_PLAYER_IDX, impute_means, …
    )


# ────────────────────────────
#  ── 4.  Wrapper público ────────────────────────────────────────
# ────────────────────────────
def predict_match(
    *,
    # --- obligatorios ------------------
    model: FootballPredictionModel,
    cfg: Dict[str, Any],
    history_df: pd.DataFrame,
    equipos_dict: Dict[str, int],
    players_name2idx: Dict[str, int],
    PAD_PLAYER_IDX: int,
    max_players: int,
    impute_means: Dict[str, float],
    # --- info del partido --------------
    local: str,
    visitor: str,
    date: str | pd.Timestamp,
    local_players: List[str] | None = None,
    visitor_players: List[str] | None = None,
    referee: str = "",
    competition: str = "",
    cuota_1: float | None = None,
    cuota_x: float | None = None,
    cuota_2: float | None = None,
    # --- opcional, si no se usa CUDA ---
    device: torch.device | None = None,
) -> Dict[str, float]:
    """
    Interfaz ÚNICA que consume tu FastAPI.

    Retorna un dict:
        {<Local>: prob, <Visitor>: prob, "Empate": prob}
    """
    device = device or torch.device("cpu")
    local_players = local_players or []
    visitor_players = visitor_players or []

    # --- 1) índices de equipos ----------
    li = equipos_dict.get(local.lower().strip())
    vi = equipos_dict.get(visitor.lower().strip())
    if li is None or vi is None:
        raise ValueError("Equipo no encontrado en el diccionario")

    # --- 2) árbitro / competición -------
    ai = cfg["num_arb"] and equipos_dict.get(referee.lower().strip(), 0) or 0
    ci = cfg["num_comp"] and equipos_dict.get(competition.lower().strip(), 0) or 0

    # --- 3) jugadores → índices + padding
    def _names2idx(names: List[str]) -> List[int]:
        idxs = [players_name2idx.get(n.lower().strip(), PAD_PLAYER_IDX) for n in names]
        return (idxs + [PAD_PLAYER_IDX] * max_players)[:max_players]

    lpl = _names2idx(local_players)
    vpl = _names2idx(visitor_players)

    # --- 4) cuotas → prob. implícitas ---
    def _to_prob(cuota: float | None, key: str) -> float:
        try:
            v = float(cuota)
            return 1 / v if v > 1 else impute_means[key]
        except Exception:
            return impute_means[key]

    p1 = _to_prob(cuota_1, "prob_cuota_1")
    px = _to_prob(cuota_x, "prob_cuota_x")
    p2 = _to_prob(cuota_2, "prob_cuota_2")

    # --- 5) llamada interna -------------
    probs = _internal_predict(
        model=model,
        history_df=history_df,
        local_idx=li,
        visitor_idx=vi,
        match_date_ts=pd.Timestamp(date),
        arbitro_idx=ai,
        competicion_idx=ci,
        local_player_idx=lpl,
        visitor_player_idx=vpl,
        odds_probs=[p1, px, p2],
        feature_dim=cfg["seq_in"],
        time_steps=cfg.get("time_steps", 15),
        device=device,
    )

    return {local.title(): probs[1], visitor.title(): probs[0], "Empate": probs[2]}


# ────────────────────────────
#  ── 5.  Exports ────────────────────────────────────────────────
# ────────────────────────────
__all__ = [
    "load_runtime_objects",
    "predict_match",
]
# ────────────────────────────────────────────────────────────────
