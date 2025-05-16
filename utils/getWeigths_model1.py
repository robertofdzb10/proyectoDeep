#!/usr/bin/env python3
import torch, json
from pathlib import Path

# 1) Ruta real a tus pesos
CKPT = Path("../models/model1.pth")

# 2) Carga estado
state = torch.load(CKPT, map_location="cpu")

# 3) Extrae embeddings de equipos
num_teams, team_embed_dim = state["embed.weight"].shape

# 4) Extrae parámetros de ConvLSTM (celda 0)
#   la conv.weight en la primera celda tiene forma (4*hid, in_ch+hid, kH, kW)
conv_w = state["convlstm.cells.0.conv.weight"]
out_ch, in_ch_plus_hid, kH, kW = conv_w.shape
conv_lstm_hid = out_ch // 4
feature_dim   = in_ch_plus_hid - conv_lstm_hid
conv_lstm_kernel = [kH, kW]

# 5) Extrae FC hidden dim de la primera capa lineal del MLP
#    asumiendo que la secuencia es: Linear(fc_in, fc_hid), ReLU...
fc_hid = state["mlp.0.weight"].shape[0]

# 6) Valores “externos” que no hay en los pesos
time_steps = 10    # <-- ajusta al valor que usaste
drop       = 0.2   # <-- idem

# 7) Construye el dict de config
config = {
    "model_path":      str(CKPT),
    "num_teams":       num_teams,
    "feature_dim":     feature_dim,
    "team_embed_dim":  team_embed_dim,
    "convlstm_hid":    conv_lstm_hid,
    "convlstm_kernel": conv_lstm_kernel,
    "fc_hid":          fc_hid,
    "drop":            drop,
    "time_steps":      time_steps
}

# 8) Vuelca a JSON en model/model1_config.json
cfg_path = Path("model/model1_config.json")
cfg_path.parent.mkdir(parents=True, exist_ok=True)
with cfg_path.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("✅  model1_config.json generado en", cfg_path)
print(json.dumps(config, indent=2, ensure_ascii=False))
