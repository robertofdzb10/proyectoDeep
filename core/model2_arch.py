# core/model2_arch.py
import torch
import torch.nn as nn

__all__ = ["FootballPredictionModel"]


class FootballPredictionModel(nn.Module):
    """
    Arquitectura ‘modelo 2’ híbrida:
      • Dos LSTM bidireccionales + atención
      • Embeddings de equipo, árbitro, competición y jugadores
      • Cuotas implícitas como features numéricas
    """

    def __init__(
        self,
        *,
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

        # -------- embeddings --------
        self.team_emb = nn.Embedding(num_teams, team_dim)
        self.arb_emb = (
            nn.Embedding(num_arb, arb_dim) if num_arb > 0 else None
        )
        self.comp_emb = (
            nn.Embedding(num_comp, comp_dim) if num_comp > 0 else None
        )
        self.player_emb = nn.Embedding(
            num_players + 1, player_dim, padding_idx=pad_idx
        )
        self.pad_idx = pad_idx

        # -------- LSTM + atención --------
        self.lstm_loc = nn.LSTM(
            seq_in, lstm_hid, 2, batch_first=True, bidirectional=True, dropout=drop
        )
        self.lstm_vis = nn.LSTM(
            seq_in, lstm_hid, 2, batch_first=True, bidirectional=True, dropout=drop
        )
        self.att_loc = nn.Sequential(nn.Linear(2 * lstm_hid, 1), nn.Softmax(dim=1))
        self.att_vis = nn.Sequential(nn.Linear(2 * lstm_hid, 1), nn.Softmax(dim=1))

        # -------- FC final --------
        combined = (
            2 * (2 * lstm_hid)  # contextos loc + vis
            + 2 * team_dim
            + (arb_dim if num_arb > 0 else 0)
            + (comp_dim if num_comp > 0 else 0)
            + 2 * player_dim
            + num_odds
        )
        self.fc = nn.Sequential(
            nn.Linear(combined, fc_hid),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(fc_hid, fc_hid // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(fc_hid // 2, 3),  # [visitor, local, draw]
        )

    # -------------------------------------------------------------
    def forward(
        self,
        loc_seq,
        vis_seq,
        loc_i,
        vis_i,
        arb_i,
        comp_i,
        lp,
        vp,
        odds,
    ):
        # Embeddings
        te_loc = self.team_emb(loc_i)
        te_vis = self.team_emb(vis_i)
        arb_e = self.arb_emb(arb_i) if self.arb_emb else torch.zeros_like(te_loc)
        comp_e = self.comp_emb(comp_i) if self.comp_emb else torch.zeros_like(te_loc)

        # LSTM + atención
        lo, _ = self.lstm_loc(loc_seq)
        vo, _ = self.lstm_vis(vis_seq)
        wlo = self.att_loc(lo)
        wvo = self.att_vis(vo)
        ctx_lo = (lo * wlo).sum(1)
        ctx_vo = (vo * wvo).sum(1)

        # Jugadores
        ple = self.player_emb(lp)
        vpe = self.player_emb(vp)
        mask_lp = (lp != self.pad_idx).unsqueeze(-1).float()
        mask_vp = (vp != self.pad_idx).unsqueeze(-1).float()
        agg_lp = (ple * mask_lp).sum(1) / mask_lp.sum(1).clamp(min=1)
        agg_vp = (vpe * mask_vp).sum(1) / mask_vp.sum(1).clamp(min=1)

        # Concatenar todo
        feat = torch.cat(
            [ctx_lo, ctx_vo, te_loc, te_vis, arb_e, comp_e, agg_lp, agg_vp, odds], dim=1
        )
        return self.fc(feat)
