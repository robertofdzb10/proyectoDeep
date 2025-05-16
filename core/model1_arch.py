# core/model1_arch.py
import torch
import torch.nn as nn
from .convlstm import ConvLSTM

__all__ = ["FootballPredictorSimple"]


class FootballPredictorSimple(nn.Module):
    """
    Arquitectura ‘modelo 1’:
      • Dos secuencias (local / visitante) → ConvLSTM
      • Embedding básico de equipo
      • MLP final para dos clases (Visitor / Local)
    """

    def __init__(
        self,
        num_teams: int,
        feature_dim: int = 2,
        team_embed_dim: int = 12,
        convlstm_hid: int = 24,
        convlstm_kernel: tuple = (3, 3),
        fc_hid: int = 48,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.embed = nn.Embedding(num_teams, team_embed_dim)

        self.convlstm = ConvLSTM(
            feature_dim,              # canales de entrada
            convlstm_hid,
            convlstm_kernel,
            layers=1,
            batch_first=True,
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        fc_in = 2 * convlstm_hid + 2 * team_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(fc_in, fc_hid),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(fc_hid, 2),     # [visitor_win, local_win]
        )

    # -------------------------------------------------------------
    def forward(
        self,
        seq_l: torch.Tensor,  # (B,T,F)
        seq_v: torch.Tensor,  # (B,T,F)
        idx_l: torch.Tensor,  # (B,)
        idx_v: torch.Tensor,  # (B,)
    ) -> torch.Tensor:

        # (B,T,F) → (B,T,F,1,1) para ConvLSTM
        hl = self.convlstm(seq_l.unsqueeze(-1).unsqueeze(-1))
        hv = self.convlstm(seq_v.unsqueeze(-1).unsqueeze(-1))

        hl = self.avg(hl).flatten(1)
        hv = self.avg(hv).flatten(1)

        el, ev = self.embed(idx_l), self.embed(idx_v)
        x = torch.cat([hl, hv, el, ev], dim=1)
        return self.mlp(x)
