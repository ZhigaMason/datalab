import pytest
import torch
import torch.nn.functional as F

from w2v.losses import jepa_loss, normalized_std, ns_loss, vicreg_terms
from w2v.model import W2V


def test_ns_loss_matches_reference():
    torch.manual_seed(0)
    B, T, d, V, K = 1, 4, 3, 10, 6
    ctx, W = torch.randn(B, T, d), torch.randn(V, d)
    tgt = torch.randint(2, V, (B, T))
    negs = torch.randint(2, V, (K,))
    valid = torch.tensor([[True, False, True, True]])
    loss, parts = ns_loss(ctx, tgt, valid, W, negs)
    ref = []
    for t in (0, 2, 3):
        h = ctx[0, t]
        pos = F.logsigmoid(h @ W[tgt[0, t]])
        neg = sum(F.logsigmoid(-(h @ W[n])) for n in negs)
        ref.append(-(pos + 5 / K * neg))
    torch.testing.assert_close(loss, torch.stack(ref).mean())
    assert "loss_pos" in parts


def test_ns_loss_all_invalid_is_zero_not_nan():
    loss, _ = ns_loss(torch.randn(1, 3, 2), torch.ones(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.bool),
                      torch.randn(4, 2), torch.tensor([2, 3]))
    assert loss.item() == 0.0


def test_vicreg_terms():
    torch.manual_seed(0)
    var_l, _ = vicreg_terms(torch.randn(512, 8) * 3)
    assert var_l.item() == 0.0
    var_l, _ = vicreg_terms(torch.ones(64, 8))
    assert var_l.item() == pytest.approx(0.99, abs=1e-3)
    _, cov_l = vicreg_terms(torch.randn(4096, 4))
    assert cov_l.item() < 0.01
    z = torch.randn(4096, 4)
    z[:, 1] = z[:, 0]
    assert vicreg_terms(z)[1].item() > 0.3


def test_jepa_loss_cosine_part():
    torch.manual_seed(0)
    tgt = torch.randn(1, 5, 4)
    valid = torch.ones(1, 5, dtype=torch.bool)
    loss, parts = jepa_loss(2 * tgt, tgt, valid, 0.0, 0.0, 8)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    loss, parts = jepa_loss(-tgt, tgt, valid, 0.0, 0.0, 8)
    assert parts["loss_cos"].item() == pytest.approx(2.0, abs=1e-5)


def test_normalized_std():
    assert normalized_std(torch.ones(10, 4)).item() == pytest.approx(0.0, abs=1e-6)
    assert normalized_std(torch.randn(1000, 16)).item() > 0.1


def test_model_ns_loss_backprop():
    torch.manual_seed(0)
    model = W2V(30, 8, window=2, mode="bidir", jepa=False)
    x = torch.randint(2, 30, (2, 12))
    m = torch.rand(2, 12) > 0.2
    loss, parts = model.loss(x, m, torch.randint(2, 30, (16,)))
    loss.backward()
    assert torch.isfinite(loss) and model.out.weight.grad.abs().sum() > 0


def test_masked_target_does_not_contribute():
    torch.manual_seed(0)
    model = W2V(30, 8, window=2, mode="causal", jepa=False)
    torch.nn.init.normal_(model.out.weight)
    x = torch.randint(2, 30, (1, 12))
    m = torch.ones(1, 12, dtype=torch.bool)
    m[0, 5] = False
    negs = torch.randint(2, 30, (16,))
    x2 = x.clone()
    x2[0, 5] = (x[0, 5] - 2 + 1) % 28 + 2
    torch.testing.assert_close(model.loss(x, m, negs)[0], model.loss(x2, m, negs)[0])


def test_model_jepa_loss_grads_skip_ema():
    torch.manual_seed(0)
    model = W2V(30, 8, window=2, mode="causal", jepa=True)
    x = torch.randint(2, 30, (2, 12))
    m = torch.ones(2, 12, dtype=torch.bool)
    loss, parts = model.loss(x, m, None, n_vicreg=16)
    loss.backward()
    assert set(parts) == {"loss_cos", "loss_var", "loss_cov", "loss_var_emb", "loss_cov_emb"}  # lambda_emb 0.3
    assert model.emb_ema.weight.grad is None
    assert model.emb.weight.grad is not None and model.predictor.weight.grad is not None


def test_vicreg_terms_fp32_under_autocast():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        var_l, cov_l = vicreg_terms(torch.randn(256, 8))
    assert var_l.dtype == torch.float32
    assert cov_l.dtype == torch.float32


def test_jepa_loss_regularizes_embeddings_when_asked():
    torch.manual_seed(0)
    pred, tgt = torch.randn(1, 64, 8), torch.randn(1, 64, 8)
    valid = torch.ones(1, 64, dtype=torch.bool)
    emb = torch.ones(1, 64, 8, requires_grad=True)  # collapsed rows: var hinge is maximal
    torch.manual_seed(0)
    base, parts0 = jepa_loss(pred, tgt, valid, 1.0, 0.04, 32)
    torch.manual_seed(0)
    base2, _ = jepa_loss(pred, tgt, valid, 1.0, 0.04, 32, emb=emb, lambda_emb=0.0)
    assert set(parts0) == {"loss_cos", "loss_var", "loss_cov"}
    torch.testing.assert_close(base, base2)
    torch.manual_seed(0)
    loss, parts = jepa_loss(pred, tgt, valid, 1.0, 0.04, 32, emb=emb, lambda_emb=0.5)
    assert parts["loss_var_emb"].item() == pytest.approx(1.0 - 1e-2, abs=1e-3)  # sqrt(0 + 1e-4) = 0.01
    torch.testing.assert_close(loss, base + 0.5 * (parts["loss_var_emb"] + 0.04 * parts["loss_cov_emb"]))
    e = torch.randn(1, 64, 8, requires_grad=True)
    jepa_loss(pred, tgt, valid, 1.0, 0.04, 32, emb=e * 0.01, lambda_emb=1.0)[0].backward()
    assert e.grad.abs().sum() > 0


def test_model_jepa_lambda_emb_sends_vicreg_grads_to_embeddings():
    torch.manual_seed(0)
    model = W2V(30, 8, window=2, mode="bidir", jepa=True)
    x = torch.randint(2, 30, (2, 12))
    m = torch.ones(2, 12, dtype=torch.bool)
    loss, parts = model.loss(x, m, None, n_vicreg=16, lambda_emb=1.0)
    assert {"loss_var_emb", "loss_cov_emb"} <= set(parts)
    loss.backward()
    assert model.emb.weight.grad[x.unique()].abs().sum() > 0
