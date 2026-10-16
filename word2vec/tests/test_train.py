import json

import pytest
import torch

from w2v.train import ema_tau, lr_lambda, parse_args, run_name, train


def test_run_name():
    assert run_name("cs", "causal", True, 300, 5, 0) == "cs-causal-jepa-d300-w5-s0"
    assert run_name("en", "bidir", False, 100, 20, 2) == "en-bidir-d100-w20-s2"


def test_schedules():
    assert lr_lambda(0, 100, 10) == pytest.approx(0.1)
    assert lr_lambda(9, 100, 10) == pytest.approx(1.0)
    assert lr_lambda(100, 100, 10) == pytest.approx(0.0)
    assert ema_tau(0, 100, 0.996) == pytest.approx(0.996)
    assert ema_tau(100, 100, 0.996) == pytest.approx(1.0)


def _args(root, *extra, mode="bidir", jepa=False, steps=6):
    argv = ["--lang", "en", "--mode", mode, "--dim", "16", "--batch", "4", "--seq-len", "64", "--neg", "32",
            "--max-steps", str(steps), "--log-every", "2", "--eval-every", "3", "--ckpt-every", "3",
            "--n-vicreg", "64", "--subsample-t", "0", "--device", "cpu", "--no-compile",
            "--data-dir", str(root / "data"), "--runs-dir", str(root / "runs"), *extra]
    if jepa:
        argv.append("--jepa")
    return parse_args(argv)


@pytest.mark.parametrize("mode", ["bidir", "causal"])
@pytest.mark.parametrize("jepa", [False, True])
def test_train_end_to_end(tiny_data, mode, jepa):
    run_dir = train(_args(tiny_data, "--write-vec", mode=mode, jepa=jepa))
    assert run_dir.name == run_name("en", mode, jepa, 16, 5, 0)
    for f in ("config.json", "metrics.jsonl", "model.onnx", "vocab.json", "final_metrics.json", "vectors.vec"):
        assert (run_dir / f).exists(), f
    assert not (run_dir / "ckpt.pt").exists()  # finished runs drop their checkpoint unless --keep-ckpt
    final = json.loads((run_dir / "final_metrics.json").read_text())
    assert final["onnx_max_abs_diff"] <= 1e-5 and "sim_toy/spearman" in final and "collapse_warning" in final
    records = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_recs = [r for r in records if "loss" in r]
    assert [r["step"] for r in train_recs] == [2, 4, 6]
    assert all(r["loss"] == r["loss"] for r in train_recs)  # not NaN
    assert any("sim_toy/coverage" in r for r in records)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["dim"] == 16 and cfg["jepa"] == jepa and cfg["vocab_size"] == 52


def test_resume_matches_uninterrupted_run(tiny_data):
    straight = train(_args(tiny_data, "--run-dir", str(tiny_data / "runs/straight"), "--keep-ckpt"))
    interrupted = tiny_data / "runs/interrupted"
    train(_args(tiny_data, "--run-dir", str(interrupted), "--stop-after", "3"))
    assert torch.load(interrupted / "ckpt.pt", weights_only=False)["step"] == 3
    assert not (interrupted / "model.onnx").exists()
    train(_args(tiny_data, "--run-dir", str(interrupted), "--keep-ckpt"))
    a = torch.load(straight / "ckpt.pt", weights_only=False)
    b = torch.load(interrupted / "ckpt.pt", weights_only=False)
    assert a["step"] == b["step"] == 6
    torch.testing.assert_close(a["model"]["emb.weight"], b["model"]["emb.weight"])


def test_resume_refuses_changed_config(tiny_data):
    run_dir = str(tiny_data / "runs/x")
    train(_args(tiny_data, "--run-dir", run_dir, "--stop-after", "1"))
    with pytest.raises(ValueError, match="different config"):
        train(_args(tiny_data, "--run-dir", run_dir, "--lr", "0.1"))


def test_window_wider_than_encoded_pad_is_rejected(tiny_data):
    with pytest.raises(ValueError, match="pad"):
        train(_args(tiny_data, "--window", "6"))


def test_jepa_resume_matches_uninterrupted_run(tiny_data):
    straight = train(_args(tiny_data, "--run-dir", str(tiny_data / "runs/straight"), "--keep-ckpt", jepa=True))
    interrupted = tiny_data / "runs/interrupted"
    train(_args(tiny_data, "--run-dir", str(interrupted), "--stop-after", "3", jepa=True))
    train(_args(tiny_data, "--run-dir", str(interrupted), "--keep-ckpt", jepa=True))
    a = torch.load(straight / "ckpt.pt", weights_only=False)["model"]
    b = torch.load(interrupted / "ckpt.pt", weights_only=False)["model"]
    for k in ("emb.weight", "emb_ema.weight", "predictor.weight"):
        torch.testing.assert_close(a[k], b[k])


def test_resume_refuses_re_encoded_corpus(tiny_data):
    run_dir = str(tiny_data / "runs/x")
    train(_args(tiny_data, "--run-dir", run_dir, "--stop-after", "1"))
    p = tiny_data / "data/en/stats.json"
    stats = json.loads(p.read_text())
    stats["sentences"] += 1
    p.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="different config"):
        train(_args(tiny_data, "--run-dir", run_dir))


def test_missing_benchmarks_fail_fast(tiny_data):
    import shutil

    shutil.rmtree(tiny_data / "data/benchmarks")
    run_dir = tiny_data / "runs/nb"
    with pytest.raises(FileNotFoundError, match="fetch_benchmarks"), pytest.warns(UserWarning):
        train(_args(tiny_data, "--run-dir", str(run_dir)))
    assert not (run_dir / "config.json").exists()
    with pytest.warns(UserWarning):
        out = train(_args(tiny_data, "--run-dir", str(run_dir), "--allow-no-benchmarks"))
    assert (out / "model.onnx").exists()


def test_jepa_identity_predictor_and_lambda_emb_train(tiny_data):
    run_dir = train(_args(tiny_data, "--predictor", "identity", "--lambda-emb", "1.0", jepa=True))
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["predictor"] == "identity" and cfg["lambda_emb"] == 1.0
    recs = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any("loss_var_emb" in r for r in recs)
    assert (run_dir / "final_metrics.json").exists()


def test_jepa_regularizes_the_table_by_default(tmp_path):
    assert _args(tmp_path, jepa=True).lambda_emb == 0.3
