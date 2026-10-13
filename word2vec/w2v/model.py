"""Word2vec with a fixed, closed-form convolutional context average."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from w2v.data import PAD_ID
from w2v.losses import jepa_loss, ns_loss

MODES = ("bidir", "causal")


def context_kernel(window: int, mode: str) -> torch.Tensor:
    """bidir: ([1]*k + [0] + [1]*k) / 2k;  causal: ([1]*k + [0] + [0]*k) / k."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if window < 1:
        raise ValueError("window must be >= 1")
    right = torch.ones(window) if mode == "bidir" else torch.zeros(window)
    w = torch.cat([torch.ones(window), torch.zeros(1), right])
    return w / w.sum()


class W2V(nn.Module):
    def __init__(self, vocab_size: int, dim: int, window: int, mode: str, jepa: bool):
        super().__init__()
        self.dim, self.window, self.mode, self.jepa = dim, window, mode, jepa
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        nn.init.uniform_(self.emb.weight, -0.5 / dim, 0.5 / dim)
        with torch.no_grad():
            self.emb.weight[PAD_ID].zero_()
        kernel = context_kernel(window, mode)
        self.register_buffer("kernel", kernel, persistent=False)
        self.register_buffer("support", (kernel > 0).float(), persistent=False)
        if jepa:
            self.predictor = nn.Linear(dim, dim, bias=False)
            self.emb_ema = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
            self.emb_ema.weight.data.copy_(self.emb.weight.data)
            self.emb_ema.weight.requires_grad_(False)
        else:
            self.out = nn.Linear(dim, vocab_size, bias=False)
            nn.init.zeros_(self.out.weight)

    def context(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B,T) ids, m (B,T) bool -> ctx (B,T,d) closed-form window average, has_ctx (B,T) bool."""
        e = self.emb(x) * m.unsqueeze(-1).to(self.emb.weight.dtype)
        w = self.kernel.view(1, 1, -1).expand(self.dim, 1, -1)
        ctx = F.conv1d(e.transpose(1, 2), w, padding=self.window, groups=self.dim).transpose(1, 2)
        n = F.conv1d(m.float().unsqueeze(1), self.support.view(1, 1, -1), padding=self.window).squeeze(1)
        return ctx, n > 0.5

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        self.emb_ema.weight.lerp_(self.emb.weight, 1.0 - tau)

    def loss(self, x, m, negatives=None, lambda_var: float = 1.0, lambda_cov: float = 0.04, n_vicreg: int = 2048):
        """Returns (scalar loss, dict of detached parts). Loss positions: real target with >= 1 context token."""
        ctx, has_ctx = self.context(x, m)
        valid = m & has_ctx
        if self.jepa:
            with torch.no_grad():
                tgt = self.emb_ema(x)
            return jepa_loss(self.predictor(ctx), tgt, valid, lambda_var, lambda_cov, n_vicreg)
        return ns_loss(ctx, x, valid, self.out.weight, negatives)
