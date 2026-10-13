import numpy as np
import pytest
import torch

from w2v.data import TokenStream, keep_probs, noise_dist, sample_mask


def _corpus(tmp_path, n):
    p = tmp_path / "tokens.bin"
    np.arange(n, dtype=np.uint32).tofile(p)
    return p


def test_token_stream_deterministic_contiguous_chunks(tmp_path):
    p = _corpus(tmp_path, 100)
    s = TokenStream(p, seq_len=10, batch_size=4, seed=0)
    b = s.batch(3)
    assert b.shape == (4, 10) and b.dtype == torch.int64
    assert torch.equal(b, s.batch(3))
    assert torch.equal(b, TokenStream(p, 10, 4, seed=0).batch(3))
    assert not torch.equal(b, s.batch(4))
    assert not torch.equal(b, TokenStream(p, 10, 4, seed=1).batch(3))
    assert torch.equal(b[:, 1:] - b[:, :-1], torch.ones(4, 9, dtype=torch.long))
    assert int(b.max()) < 100
    assert s.steps_per_epoch() == 2 and s.num_tokens == 100


def test_token_stream_shorter_than_seq_len_errors(tmp_path):
    with pytest.raises(ValueError, match="seq_len"):
        TokenStream(_corpus(tmp_path, 5), seq_len=10, batch_size=2, seed=0)


def test_keep_probs():
    p = keep_probs([0, 0, 900, 100], t=0.01)
    assert p[0] == 0 and p[1] == 0
    assert p[2].item() == pytest.approx((0.01 / 0.9) ** 0.5, rel=1e-5)
    assert p[3].item() == pytest.approx((0.01 / 0.1) ** 0.5, rel=1e-5)
    assert keep_probs([0, 0, 1, 10**9], t=1e-4)[2].item() == 1.0
    assert keep_probs([0, 0, 900, 100], t=0).tolist() == [0.0, 0.0, 1.0, 1.0]


def test_sample_mask_follows_keep_probs():
    keep = torch.tensor([0.0, 0.0, 1.0, 0.5])
    g = torch.Generator().manual_seed(0)
    assert not sample_mask(torch.zeros(1, 50, dtype=torch.long), keep, g).any()
    assert sample_mask(torch.full((1, 50), 2), keep, g).all()
    assert sample_mask(torch.full((1, 20000), 3), keep, g).float().mean().item() == pytest.approx(0.5, abs=0.02)


def test_noise_dist():
    q = noise_dist([0, 0, 16, 1])
    assert q.tolist() == pytest.approx([0.0, 0.0, 8 / 9, 1 / 9])
