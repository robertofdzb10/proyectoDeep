import torch, json
from pathlib import Path

CKPT = Path("../models/model5.pth")          # ← tu ruta real
state = torch.load(CKPT, map_location="cpu")

# --- Embeddings ----------------------------------------------------
num_teams,   team_dim   = state["team_emb.weight"].shape
num_players1, player_dim = state["player_emb.weight"].shape   # incluye +1 para el PAD
num_players              = num_players1 - 1
pad_idx                  = num_players                       # el idx_pad es el último

if "arb_emb.weight" in state:
    num_arb,  arb_dim  = state["arb_emb.weight"].shape
else:                             # por si tu modelo no usó árbitro/competición
    num_arb,  arb_dim  = 0, 0

if "comp_emb.weight" in state:
    num_comp, comp_dim = state["comp_emb.weight"].shape
else:
    num_comp, comp_dim = 0, 0

# --- LSTM ----------------------------------------------------------
# weight_ih_l0 = (4*hidden,  input_dim)
lstm_hid = state["lstm_loc.weight_ih_l0"].shape[0] // 4
seq_in   = state["lstm_loc.weight_ih_l0"].shape[1]

# --- Primera capa FC ----------------------------------------------
fc_hid = state["fc.0.weight"].shape[0]            # unidades ocultas FC

# --- Otros hiperparámetros que NO están en los pesos --------------
num_odds = 3      # porque entrenaste con 3 cuotas implícitas
drop     = 0.2    # valor que pusiste en tu script; en inferencia no afecta

config = {
    "model_path": str(CKPT),
    "num_teams":   num_teams,
    "team_dim":    team_dim,
    "num_arb":     num_arb,
    "arb_dim":     arb_dim,
    "num_comp":    num_comp,
    "comp_dim":    comp_dim,
    "num_players": num_players,
    "player_dim":  player_dim,
    "pad_idx":     pad_idx,
    "seq_in":      seq_in,
    "lstm_hid":    lstm_hid,
    "num_odds":    num_odds,
    "fc_hid":      fc_hid,
    "drop":        drop
}

# ruta al JSON
cfg_path = Path("model/model5_config.json")

# 1) crea la carpeta (y subcarpetas) si no existen
cfg_path.parent.mkdir(parents=True, exist_ok=True)

# 2) vuelca el JSON
with cfg_path.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("✅  model2_config.json generado:")
print(json.dumps(config, indent=2, ensure_ascii=False))

