import pytest
import torch

from w2v.model import W2V, context_kernel


def naive_context(e, m, window, mode):
    T, d = e.shape
    out, has = torch.zeros(T, d), torch.zeros(T, dtype=torch.bool)
    offsets = [o for o in range(-window, window + 1) if o != 0 and (mode == "bidir" or o < 0)]
    norm = 2 * window if mode == "bidir" else window
    for t in range(T):
        for o in offsets:
            j = t + o
            if 0 <= j < T and m[j]:
                out[t] += e[j]
                has[t] = True
        out[t] /= norm
    return out, has


def test_context_kernel_closed_form():
    assert context_kernel(2, "bidir").tolist() == [0.25, 0.25, 0.0, 0.25, 0.25]
    assert context_kernel(2, "causal").tolist() == [0.5, 0.5, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        context_kernel(2, "sideways")


@pytest.mark.parametrize("mode", ["bidir", "causal"])
def test_context_matches_naive_loop(mode):
    torch.manual_seed(0)
    model = W2V(20, 6, window=3, mode=mode, jepa=False)
    x = torch.randint(2, 20, (2, 15))
    m = torch.rand(2, 15) > 0.3
    x[0, :2] = 0
    m[0, :2] = False
    ctx, has = model.context(x, m)
    for b in range(2):
        exp, exp_has = naive_context(model.emb(x[b]).detach(), m[b], 3, mode)
        torch.testing.assert_close(ctx[b].detach(), exp)
        assert torch.equal(has[b], exp_has)


def test_causal_context_ignores_current_and_future_tokens():
    torch.manual_seed(0)
    model = W2V(20, 6, window=4, mode="causal", jepa=False)
    x = torch.randint(2, 20, (1, 20))
    m = torch.ones(1, 20, dtype=torch.bool)
    x2 = x.clone()
    x2[0, 10:] = (x[0, 10:] - 2 + 1) % 18 + 2  # different non-special ids
    ctx, _ = model.context(x, m)
    ctx2, _ = model.context(x2, m)
    torch.testing.assert_close(ctx[0, :11], ctx2[0, :11])
    assert not torch.allclose(ctx[0, 11:], ctx2[0, 11:])


def test_bidir_context_ignores_center_token():
    torch.manual_seed(0)
    model = W2V(20, 6, window=2, mode="bidir", jepa=False)
    x = torch.randint(2, 20, (1, 12))
    m = torch.ones(1, 12, dtype=torch.bool)
    x2 = x.clone()
    x2[0, 6] = (x[0, 6] - 2 + 1) % 18 + 2
    torch.testing.assert_close(model.context(x, m)[0][0, 6], model.context(x2, m)[0][0, 6])


def test_has_ctx_false_when_window_fully_masked():
    model = W2V(10, 4, window=2, mode="bidir", jepa=False)
    x = torch.full((1, 9), 3)
    m = torch.zeros(1, 9, dtype=torch.bool)
    m[0, 0] = m[0, 8] = True
    ctx, has = model.context(x, m)
    assert has[0].tolist() == [False, True, True, False, False, False, True, True, False]
    assert torch.count_nonzero(ctx[0, 4]) == 0
    causal = W2V(10, 4, window=2, mode="causal", jepa=False)
    assert not causal.context(x, torch.ones(1, 9, dtype=torch.bool))[1][0, 0]


def test_jepa_model_has_ema_copy_without_grad():
    model = W2V(30, 8, window=2, mode="bidir", jepa=True)
    assert torch.equal(model.emb_ema.weight, model.emb.weight)
    assert not model.emb_ema.weight.requires_grad
    assert not hasattr(model, "out")
    with torch.no_grad():
        model.emb.weight.add_(1.0)
    before = model.emb_ema.weight.clone()
    model.update_ema(0.9)
    torch.testing.assert_close(model.emb_ema.weight, 0.9 * before + 0.1 * model.emb.weight)


def test_non_jepa_out_is_zero_initialized():
    model = W2V(30, 8, window=2, mode="causal", jepa=False)
    assert model.out.weight.shape == (30, 8) and torch.count_nonzero(model.out.weight) == 0
    assert torch.count_nonzero(model.emb.weight[0]) == 0
    assert model.emb.weight.abs().max() <= 0.5 / 8


def test_jepa_predictor_has_no_bias():
    model = W2V(30, 8, 3, "bidir", True)
    assert model.predictor.bias is None
