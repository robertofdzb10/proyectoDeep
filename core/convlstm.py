# core/convlstm.py
import torch
import torch.nn as nn
from typing import Tuple

__all__ = ["ConvLSTMCell", "ConvLSTM"]


class ConvLSTMCell(nn.Module):
    """
    Célula Conv-LSTM con padding 'same' (mantiene H×W).
    """

    def __init__(
        self,
        in_ch: int,
        hid_ch: int,
        k_size: Tuple[int, int] = (3, 3),
        bias: bool = True,
    ):
        super().__init__()
        pad = k_size[0] // 2, k_size[1] // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k_size, padding=pad, bias=bias)

    def forward(self, x: torch.Tensor, hc):
        h, c = hc
        y = torch.cat([x, h], dim=1)       # (B, in_ch+hid_ch, H, W)
        i, f, o, g = torch.chunk(self.conv(y), 4, dim=1)
        i, f, o = map(torch.sigmoid, (i, f, o))
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    # helpers ---------------------------------------------------------
    def init_state(self, b: int, hw: Tuple[int, int], device):
        h, w = hw
        zeros = torch.zeros(b, self.hid_ch, h, w, device=device)
        return zeros.clone(), zeros.clone()


class ConvLSTM(nn.Module):
    """
    Módulo multi-capa ConvLSTM.  Devuelve el *hidden state* de la última capa
    en el último paso temporal.

    Entrada esperada: (B, T, C, H, W)  si `batch_first=True`
    """

    def __init__(
        self,
        in_ch: int,
        hid_ch: int,
        k_size: Tuple[int, int] = (3, 3),
        layers: int = 1,
        batch_first: bool = True,
    ):
        super().__init__()
        self.batch_first = batch_first

        self.cells = nn.ModuleList()
        for i in range(layers):
            cur_in = in_ch if i == 0 else hid_ch
            self.cells.append(ConvLSTMCell(cur_in, hid_ch, k_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.batch_first:           # (T,B,C,H,W)  →  (B,T,C,H,W)
            x = x.permute(1, 0, 2, 3, 4)

        b, t, _, h, w = x.shape
        h_c = [cell.init_state(b, (h, w), x.device) for cell in self.cells]

        seq = x
        for l, cell in enumerate(self.cells):
            outs = []
            h, c = h_c[l]
            for ti in range(t):
                h, c = cell(seq[:, ti], (h, c))
                outs.append(h)
            seq = torch.stack(outs, dim=1)    # salida (B,T,H,W)
        return seq[:, -1]                     # último paso temporal
