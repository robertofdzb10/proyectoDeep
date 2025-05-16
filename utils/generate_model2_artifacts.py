# utils/generate_model2_artifacts.py

import sys, os
from pathlib import Path
import pickle
import pandas as pd

# ————— Ajuste del path para que 'core' sea importable —————
# Inserta la carpeta raíz del proyecto en sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ahora sí importamos tus funciones
from core.data_processing_model2 import load_and_preprocess_data, create_team_history_features

# ————— Configuración de rutas —————
MODEL_DIR    = ROOT / "model"
DF_CSV       = ROOT / "data" / "df_final_con_cuotas.csv"
PLAYERS_CSV  = ROOT / "data" / "jugadores.csv"
MAX_PLAYERS  = 11

MODEL_DIR.mkdir(exist_ok=True)

# 1) Preprocesado
df, equipos_dict, arb_dict, comp_dict, \
players_id2idx, players_name2idx, PAD_PLAYER_IDX, \
NUM_PLAYERS, impute_means = load_and_preprocess_data(
    str(DF_CSV),
    str(PLAYERS_CSV),
    MAX_PLAYERS
)

# 2) Historial
history_df = create_team_history_features(df)

# 3) Serializa artefacts
artefacts = {
    "equipos_dict":      equipos_dict,
    "arb_dict":          arb_dict,
    "comp_dict":         comp_dict,
    "players_name2idx":  players_name2idx,
    "PAD_PLAYER_IDX":    PAD_PLAYER_IDX,
    "max_players":       MAX_PLAYERS,
    "impute_means":      impute_means
}

with open(MODEL_DIR / "artefacts.pkl", "wb") as f:
    pickle.dump(artefacts, f)

# 4) Guarda el historial
history_df.to_parquet(MODEL_DIR / "history_df.parquet")

print("✅ artefacts.pkl y history_df.parquet generados en", MODEL_DIR)
