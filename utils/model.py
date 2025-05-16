"""
utils/model.py

Definición de la arquitectura ConvLSTM y del modelo de predicción.
"""

import torch
import torch.nn as nn
from typing import Tuple

# -----------------------------------------------------------------------------
# Celda ConvLSTM (single-step)
# -----------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int,
                 k_size: Tuple[int,int], bias: bool = True):
        super().__init__()
        pad = (k_size[0]//2, k_size[1]//2)
        self.hid_ch = hid_ch
        self.conv   = nn.Conv2d(in_ch + hid_ch, 4*hid_ch,
                                kernel_size=k_size, padding=pad, bias=bias)

    def forward(self, x, h_c):
        h, c = h_c
        combined = torch.cat([x,h], dim=1)
        i,f,o,g = torch.chunk(self.conv(combined), 4, dim=1)
        i,f,o   = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g       = torch.tanh(g)
        c_next  = f*c + i*g
        h_next  = o*torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch: int, hw: Tuple[int,int], device):
        h,w = hw
        zeros = torch.zeros(batch, self.hid_ch, h, w, device=device)
        return zeros.clone(), zeros.clone()

# -----------------------------------------------------------------------------
# Módulo ConvLSTM completo (varias capas si se desea)
# -----------------------------------------------------------------------------
class ConvLSTM(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int,
                 k_size: Tuple[int,int], layers: int = 1,
                 batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        self.cells = nn.ModuleList([
            ConvLSTMCell(in_ch if i==0 else hid_ch, hid_ch, k_size)
            for i in range(layers)
        ])

    def forward(self, x):
        # entrada x: (B,T,C,H,W) si batch_first=True
        if not self.batch_first:
            x = x.permute(1,0,2,3,4)
        B,T,_,H,W = x.shape
        h_c_list = [cell.init_state(B, (H,W), x.device) for cell in self.cells]
        seq = x

        for layer_idx, cell in enumerate(self.cells):
            h, c = h_c_list[layer_idx]
            outputs = []
            for t in range(T):
                h, c = cell(seq[:,t], (h,c))
                outputs.append(h)
            seq = torch.stack(outputs, dim=1)

        # devolvemos solo el último estado temporal de la última capa
        return seq[:,-1]

# -----------------------------------------------------------------------------
# Modelo final: embebido de equipos + ConvLSTM + MLP
# -----------------------------------------------------------------------------
class FootballPredictorSimple(nn.Module):
    def __init__(self, num_teams: int, feature_dim: int = 2,
                 team_embed_dim: int = 12,
                 conv_hidden: int = 24, conv_kernel=(3,3),
                 fc_hidden: int = 48, dropout: float = 0.2,
                 num_classes: int = 2):
        super().__init__()
        # Embedding de equipos
        self.embed    = nn.Embedding(num_teams, team_embed_dim)
        # ConvLSTM unicapas para secuencias
        self.convlstm = ConvLSTM(feature_dim, conv_hidden, conv_kernel,
                                 layers=1, batch_first=True)
        # Pooling espacial para aplanar salida ConvLSTM
        self.avg      = nn.AdaptiveAvgPool2d((1,1))
        # Capa densa final
        in_features = 2*conv_hidden + 2*team_embed_dim
        self.mlp     = nn.Sequential(
            nn.Linear(in_features, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, seq_l, seq_v, idx_l, idx_v):
        # seq_*: (B,T,feature_dim) → añadimos dos dims para Conv2D
        hl = self.convlstm(seq_l.unsqueeze(-1).unsqueeze(-1))
        hv = self.convlstm(seq_v.unsqueeze(-1).unsqueeze(-1))

        # pooling + flatten
        hl = self.avg(hl).flatten(1)
        hv = self.avg(hv).flatten(1)

        # embeddings de índices
        el = self.embed(idx_l)
        ev = self.embed(idx_v)

        # concatenar y pasar por MLP
        x  = torch.cat([hl,hv,el,ev], dim=1)
        return self.mlp(x)
