# api/schemas.py

from typing import List, Optional
from pydantic import BaseModel, Field

MAX_PLAYERS = 11   # igual que en tu configuración
TIME_STEPS_1 = 10  # pasos para Modelo 1
TIME_STEPS_2 = 15  # pasos para Modelo 2

class MatchRequest(BaseModel):
    """
    Petición “alta” para /match_predict (idéntica en ambos modelos).
    """
    local           : str
    visitor         : str
    date            : str   = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Fecha en formato YYYY-MM-DD"
    )
    referee         : Optional[str]   = ""
    competition     : Optional[str]   = ""
    local_players   : List[str]       = Field(
        default_factory=list,
        max_items=MAX_PLAYERS
    )
    visitor_players : List[str]       = Field(
        default_factory=list,
        max_items=MAX_PLAYERS
    )
    cuota_1         : Optional[float] = None
    cuota_x         : Optional[float] = None
    cuota_2         : Optional[float] = None


class PredictRequestModel1(BaseModel):
    """
    Petición “baja” para el endpoint /predict del Modelo 1.
    El usuario debe enviar directamente las secuencias y los índices.
    """
    seq_loc : List[List[float]] = Field(
        ..., min_length=TIME_STEPS_1, max_length=TIME_STEPS_1,
        description="Secuencia histórica local (10 pasos × 2 features)"
    )
    seq_vis : List[List[float]] = Field(
        ..., min_length=TIME_STEPS_1, max_length=TIME_STEPS_1,
        description="Secuencia histórica visitante (10 pasos × 2 features)"
    )
    idx_loc : int = Field(..., ge=0, description="Índice del equipo local")
    idx_vis : int = Field(..., ge=0, description="Índice del equipo visitante")


class PredictRequestModel2(BaseModel):
    """
    Petición “baja” para el endpoint /predict del Modelo 2.
    Acepta secuencias, índices de entidades, alineaciones y cuotas.
    """
    seq_loc    : List[List[float]] = Field(
        ..., min_length=TIME_STEPS_2, max_length=TIME_STEPS_2,
        description="Secuencia histórica local (15 pasos × 5 features)"
    )
    seq_vis    : List[List[float]] = Field(
        ..., min_length=TIME_STEPS_2, max_length=TIME_STEPS_2,
        description="Secuencia histórica visitante (15 pasos × 5 features)"
    )
    idx_loc    : int           = Field(..., ge=0, description="Índice del equipo local")
    idx_vis    : int           = Field(..., ge=0, description="Índice del equipo visitante")
    idx_ref    : int           = Field(..., ge=0, description="Índice del árbitro")
    idx_cmp    : int           = Field(..., ge=0, description="Índice de la competición")
    lineup_loc : List[int]     = Field(
        ..., min_length=MAX_PLAYERS, max_length=MAX_PLAYERS,
        description="Índices de los jugadores titulares (local)"
    )
    lineup_vis : List[int]     = Field(
        ..., min_length=MAX_PLAYERS, max_length=MAX_PLAYERS,
        description="Índices de los jugadores titulares (visitante)"
    )
    odds       : List[float]   = Field(
        ..., min_length=3, max_length=3,
        description="Probabilidades implícitas de las cuotas [P1, PX, P2]"
    )
