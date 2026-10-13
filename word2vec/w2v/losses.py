"""Training objectives: word2vec negative sampling and JEPA latent prediction (+ VICReg)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

NS_NEGATIVES = 5  # word2vec's negatives per position; the shared-K term is rescaled to match


def masked_mean(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(v.dtype)
    return (v * mask).sum() / mask.sum().clamp(min=1)


def ns_loss(ctx, targets, valid, out_weight, negatives):
    """ctx (B,T,d), targets (B,T), valid (B,T) bool, out_weight (V,d), negatives (K,) shared by the batch."""
    pos = (ctx * out_weight[targets]).sum(-1).float()
    neg = (ctx @ out_weight[negatives].T).float()
    k = negatives.numel()
    per = -(F.logsigmoid(pos) + (NS_NEGATIVES / k) * F.logsigmoid(-neg).sum(-1))
    return masked_mean(per, valid), {"loss_pos": masked_mean(-F.logsigmoid(pos), valid).detach()}


def vicreg_terms(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.float()
    n, d = z.shape
    with torch.autocast(device_type=z.device.type, enabled=False):  # keep matmul in fp32 under autocast
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        zc = z - z.mean(dim=0)
        cov = zc.T @ zc / max(n - 1, 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        cov_loss = off_diag.pow(2).sum() / d
    return var_loss, cov_loss


def jepa_loss(pred, tgt, valid, lambda_var: float, lambda_cov: float, n_vicreg: int):
    """pred (B,T,d) predictor output, tgt (B,T,d) EMA target embeddings (no grad)."""
    loss_cos = masked_mean(1.0 - F.cosine_similarity(pred.float(), tgt.float(), dim=-1), valid)
    weights = valid.flatten().float() + 1e-12  # never all-zero, keeps shapes static
    idx = torch.multinomial(weights, n_vicreg, replacement=True)
    var_l, cov_l = vicreg_terms(pred.flatten(0, 1)[idx])
    loss = loss_cos + lambda_var * var_l + lambda_cov * cov_l
    return loss, {"loss_cos": loss_cos.detach(), "loss_var": var_l.detach(), "loss_cov": cov_l.detach()}


def normalized_std(z: torch.Tensor) -> torch.Tensor:
    """Mean per-dimension std of L2-normalized rows; ~0 means collapse, ~1/sqrt(d) is healthy."""
    return F.normalize(z.float(), dim=-1).std(dim=0).mean()
