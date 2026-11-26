#!/usr/bin/env python3
"""Train a small Inkling2Loop model from scratch on WikiText-103.

The point is a controlled reproduction of the trained-loop depth curve, not a
frontier model: 8 physical layers, a 4-layer core (2 blocks of 2), 3 passes,
8 experts, hidden 256.  A `--passes 1` control is trained with the same data
and seed so inference-time looping can be compared with trained looping.

Example:
  python scripts/train_tiny_loop.py \
    --train "$DATA_DIR/tokenized/train.u16" \
    --val   "$DATA_DIR/tokenized/validation.u16" \
    --manifest "$DATA_DIR/tokenized/manifest.json" \
    --out "$RUN_DIR/tiny-loop-p3" \
    --passes 3 --tokens 120000000
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers.models.inkling2_loop import Inkling2LoopForCausalLM, Inkling2LoopTextConfig


def build_config(vocab_size: int, passes: int) -> Inkling2LoopTextConfig:
    return Inkling2LoopTextConfig(
        vocab_size=vocab_size,
        unpadded_vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        swa_num_attention_heads=8,
        swa_num_key_value_heads=4,
        swa_head_dim=32,
        sliding_window_size=128,
        d_rel=8,
        rel_extent=128,
        log_scaling_n_floor=256,
        log_scaling_alpha=0.1,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        conv_kernel_size=4,
        moe_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        route_scale=8.0,
        n_layers_per_attn_res_block=2,
        loop_layer_index_start=2,
        loop_layer_index_end=6,
        loop_num_passes=passes,
        loop_gate_rank=8,
        eos_token_id=2,
        pad_token_id=0,
    )


def sample_batch(data: np.ndarray, batch: int, ctx: int, rng: np.random.Generator) -> np.ndarray:
    offsets = rng.integers(0, len(data) - ctx - 1, size=batch)
    return np.stack([data[o:o + ctx] for o in offsets]).astype(np.int64)


def fixed_batches(data: np.ndarray, batch: int, ctx: int, n_batches: int) -> list[np.ndarray]:
    offsets = np.linspace(0, len(data) - ctx - 1, num=batch * n_batches, dtype=np.int64)
    return [np.stack([data[o:o + ctx] for o in offsets[i * batch:(i + 1) * batch]]).astype(np.int64)
            for i in range(n_batches)]


@torch.no_grad()
def evaluate(model, batches, passes: int) -> float:
    model.eval()
    old = model.config.loop_num_passes
    model.config.loop_num_passes = passes
    total, count = 0.0, 0
    for arr in batches:
        ids = torch.from_numpy(arr).cuda()
        out = model(input_ids=ids, labels=ids, use_cache=False)
        total += float(out.loss) * ids.numel()
        count += ids.numel()
    model.config.loop_num_passes = old
    model.train()
    return math.exp(total / count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=120_000_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    vocab_size = int(manifest["vocab_size"])
    train = np.fromfile(args.train, dtype=np.uint16)
    val = np.fromfile(args.val, dtype=np.uint16)
    steps = args.tokens // (args.batch * args.ctx)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    print(f"train tokens={len(train):,} val tokens={len(val):,} steps={steps} passes={args.passes}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = Inkling2LoopForCausalLM(build_config(vocab_size, args.passes)).cuda().train()
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    warmup = min(args.warmup, max(1, steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        frac = (step - warmup) / max(1, steps - warmup)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * frac))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    val_batches = fixed_batches(val, args.eval_batch, args.ctx, args.eval_batches)
    log_path = out / "train.jsonl"
    results = {"args": vars(args), "vocab_size": vocab_size, "steps": steps,
               "params": sum(p.numel() for p in model.parameters()), "evals": []}
    t0 = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        arr = sample_batch(train, args.batch, args.ctx, rng)
        ids = torch.from_numpy(arr).cuda()
        out_model = model(input_ids=ids, labels=ids, use_cache=False)
        loss = out_model.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        running += float(loss.detach())
        if step % 100 == 0:
            print(f"step {step}/{steps} loss {running/100:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"tok {step*args.batch*args.ctx/1e6:.1f}M wall {time.time()-t0:.0f}s", flush=True)
            running = 0.0
        if step % args.eval_every == 0 or step == steps:
            ppl = {p: evaluate(model, val_batches, p) for p in (1, 2, 3, 4)}
            rec = {"step": step, "tokens": step * args.batch * args.ctx,
                   "train_loss": float(loss.detach()), "val_ppl": ppl,
                   "wall_s": round(time.time() - t0, 1)}
            results["evals"].append(rec)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"eval step {step}: " + " ".join(f"p{p}={ppl[p]:.2f}" for p in ppl), flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        if step % args.save_every == 0 and step != steps:
            torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict(),
                        "step": step, "args": vars(args)}, out / f"checkpoint-{step}.pt")
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict(),
                "step": steps, "args": vars(args)}, out / "model.pt")
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out/'model.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
