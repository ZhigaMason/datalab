"""Train one word2vec variant and export its ONNX embedding artifact.

    python -m w2v.train --lang cs --mode causal --jepa --dim 300 --seed 0 --data-dir data
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from w2v.data import TokenStream, Vocab, keep_probs, noise_dist, sample_mask
from w2v.evaluate import evaluate_all, load_benchmarks
from w2v.export import export_onnx, load_onnx_embeddings, verify_onnx, write_vec
from w2v.losses import normalized_std
from w2v.model import MODES, PREDICTORS, W2V

CONFIG_KEYS = ("lang", "mode", "jepa", "dim", "window", "neg", "subsample_t", "lr", "warmup_frac", "batch",
               "seq_len", "epochs", "max_steps", "ema_start", "lambda_var", "lambda_cov", "n_vicreg", "seed",
               "lambda_emb", "predictor")
COLLAPSE_STD = 1e-3


def run_name(lang: str, mode: str, jepa: bool, dim: int, window: int, seed: int) -> str:
    return f"{lang}-{mode}{'-jepa' if jepa else ''}-d{dim}-w{window}-s{seed}"


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_tau(step: int, total: int, tau0: float) -> float:
    return 1.0 - (1.0 - tau0) * (math.cos(math.pi * min(1.0, step / max(1, total))) + 1.0) / 2.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["en", "cs", "uk"])
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--jepa", action="store_true")
    p.add_argument("--dim", type=int, default=300)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--neg", type=int, default=1024, help="shared negatives per batch (non-JEPA)")
    p.add_argument("--subsample-t", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None, help="override the epoch-derived step count")
    p.add_argument("--ema-start", type=float, default=0.996)
    p.add_argument("--lambda-var", type=float, default=1.0)
    p.add_argument("--lambda-cov", type=float, default=0.04)
    p.add_argument("--n-vicreg", type=int, default=2048)
    p.add_argument("--predictor", choices=PREDICTORS, default="linear", help="JEPA predictor")
    p.add_argument("--lambda-emb", type=float, default=0.3,
                   help="JEPA: also apply the VICReg terms (x this weight) to the online target embeddings; "
                        "0 = the original objective, whose embedding table collapses toward one direction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000, help="0 disables periodic evaluation")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--stop-after", type=int, default=None, help="stop after N steps this invocation, no export")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-dir", default=None, help="explicit run directory (default: runs-dir/<run name>)")
    p.add_argument("--device", default="auto")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile the loss (default: on for CUDA)")
    p.add_argument("--allow-no-benchmarks", action="store_true", help="train even if no benchmark files exist")
    p.add_argument("--write-vec", action="store_true", help="also write vectors.vec")
    p.add_argument("--keep-ckpt", action="store_true",
                   help="keep ckpt.pt after a finished run (default: delete it; it holds weights + Adam state)")
    return p.parse_args(argv)


def _save_ckpt(path: Path, model, opt, sched, step: int, collapse: bool) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "collapse": collapse}, tmp)
    os.replace(tmp, path)


def _prepare_run_dir(args, run_dir: Path, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = sorted(k for k in (*CONFIG_KEYS, "vocab_size", "total_steps", "data_stats") if old.get(k) != config.get(k))
        if diff:
            raise ValueError(f"run dir {run_dir} has a different config ({', '.join(diff)}); use another --run-dir")
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def train(args) -> Path:
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    use_compile = device.type == "cuda" if args.compile is None else args.compile
    lang_dir = Path(args.data_dir) / args.lang
    vocab = Vocab.load(lang_dir / "vocab.json")
    stats = json.loads((lang_dir / "stats.json").read_text())
    if stats["pad"] < args.window:
        raise ValueError(f"corpus was encoded with pad={stats['pad']} < window={args.window}; "
                         f"re-run scripts.encode with --pad >= {args.window}")
    stream = TokenStream(lang_dir / "tokens.bin", args.seq_len, args.batch, args.seed)
    total = args.max_steps or max(1, int(stream.steps_per_epoch() * args.epochs))
    warmup = max(1, int(total * args.warmup_frac))

    bench_dir = Path(args.data_dir) / "benchmarks" / args.lang
    benchmarks = load_benchmarks(bench_dir)
    if not (benchmarks["sim"] or benchmarks["analogy"]) and not args.allow_no_benchmarks:
        raise FileNotFoundError(f"no benchmarks in {bench_dir}; run scripts.fetch_benchmarks "
                                "(or pass --allow-no-benchmarks)")

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.runs_dir) / run_name(
        args.lang, args.mode, args.jepa, args.dim, args.window, args.seed)
    config = {k: getattr(args, k) for k in CONFIG_KEYS}
    config.update(vocab_size=len(vocab), total_steps=total, data_stats=stats)
    _prepare_run_dir(args, run_dir, config)

    torch.manual_seed(args.seed)
    model = W2V(len(vocab), args.dim, args.window, args.mode, args.jepa, args.predictor).to(device)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("predictor.")]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("predictor.")]
    opt = torch.optim.AdamW([{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": 0.01}],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, warmup))
    keep = keep_probs(vocab.counts, args.subsample_t).to(device)
    noise = noise_dist(vocab.counts).to(device)

    ckpt_path = run_dir / "ckpt.pt"
    start, collapse = 0, False
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, collapse = state["step"], state["collapse"]
        print(f"resumed {run_dir} at step {start}/{total}")

    loss_fn = torch.compile(model.loss) if use_compile else model.loss
    writer = SummaryWriter(str(run_dir / "tb"))
    end = total if args.stop_after is None else min(total, start + args.stop_after)
    t0, tokens_seen = time.time(), 0

    with open(run_dir / "metrics.jsonl", "a") as metrics_f:
        def log(record: dict) -> None:
            metrics_f.write(json.dumps(record) + "\n")
            metrics_f.flush()
            for k, v in record.items():
                if k != "step" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    writer.add_scalar(k, v, record["step"])

        for step in range(start, end):
            model.train()
            x = stream.batch(step).to(device, non_blocking=True)
            torch.manual_seed(args.seed * 1_000_003 + step)  # mask, negatives, VICReg sample: resumable
            m = sample_mask(x, keep)
            negs = None if args.jepa else torch.multinomial(noise, args.neg, replacement=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, parts = loss_fn(x, m, negs, args.lambda_var, args.lambda_cov, args.n_vicreg,
                                        args.lambda_emb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if args.jepa:
                model.update_ema(ema_tau(step + 1, total, args.ema_start))
            tokens_seen += x.numel()
            done = step + 1

            if done % args.log_every == 0 or done == total:
                loss_v = loss.item()
                if not math.isfinite(loss_v):
                    raise FloatingPointError(f"non-finite loss {loss_v} at step {done}")
                record = {"step": done, "loss": loss_v, "lr": sched.get_last_lr()[0],
                          "tokens_per_s": tokens_seen / max(time.time() - t0, 1e-9)}
                record.update({k: v.item() for k, v in parts.items()})
                with torch.no_grad():
                    ctx, has = model.context(x, m)
                    sel = m & has
                    record["emb_std"] = normalized_std(model.emb(x[sel][:4096])).item()
                    if args.jepa:
                        record["pred_std"] = normalized_std(model.predictor(ctx[sel][:4096])).item()
                if args.jepa and done > warmup:
                    now_collapsed = min(record["emb_std"], record["pred_std"]) < COLLAPSE_STD
                    if now_collapsed and not collapse:
                        warnings.warn(f"representation collapse suspected at step {done}")
                    collapse = now_collapsed
                    if collapse:
                        record["collapse_warning"] = True
                log(record)
            if args.eval_every and done % args.eval_every == 0:
                ev = evaluate_all(benchmarks, vocab.index, model.emb.weight.detach().float())
                log({"step": done, **ev})
            if done % args.ckpt_every == 0:
                _save_ckpt(ckpt_path, model, opt, sched, done, collapse)
            if done % args.log_every == 0 or done == total:
                t0, tokens_seen = time.time(), 0

        _save_ckpt(ckpt_path, model, opt, sched, end, collapse)
    if end < total:
        print(f"stopped at step {end}/{total}; re-run the same command to resume")
        writer.close()
        return run_dir

    weight = model.emb.weight.detach().float().cpu()
    onnx_path = export_onnx(weight, run_dir / "model.onnx")
    diff = verify_onnx(onnx_path, weight)
    vocab.save(run_dir / "vocab.json")
    words, matrix = load_onnx_embeddings(run_dir)
    final = {"onnx_max_abs_diff": diff, "collapse_warning": collapse, "steps": total,
             **evaluate_all(benchmarks, {w: i for i, w in enumerate(words)}, torch.from_numpy(matrix))}
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    if args.write_vec:
        write_vec(run_dir / "vectors.vec", words, matrix)
    if not args.keep_ckpt:
        ckpt_path.unlink(missing_ok=True)
    writer.close()
    print(json.dumps(final, indent=2))
    return run_dir


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
