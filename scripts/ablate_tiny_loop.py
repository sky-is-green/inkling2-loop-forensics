#!/usr/bin/env python3
"""Evaluate a trained tiny Inkling2Loop checkpoint and run weight-space ablations.

Ablations:
  message_gates  zero every *_message_gate parameter (the gate becomes hidden)
  message_read   zero the shared message_input query (uniform message read)
  block_queries  zero attention_input/mlp_input/final_input queries (uniform bank read)
  both           message gates + block queries

Example:
  python scripts/ablate_tiny_loop.py \
    --checkpoint ~/Desktop/scion-loop-work/runs/tiny-loop-p3/model.pt \
    --val .../validation.u16 --manifest .../manifest.json \
    --out ~/Desktop/scion-loop-work/runs/tiny-loop-p3/ablation.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers.models.inkling2_loop import Inkling2LoopForCausalLM, Inkling2LoopTextConfig


def fixed_batches(data, batch, ctx, n_batches):
    offsets = np.linspace(0, len(data) - ctx - 1, num=batch * n_batches, dtype=np.int64)
    return [np.stack([data[o:o + ctx] for o in offsets[i * batch:(i + 1) * batch]]).astype(np.int64)
            for i in range(n_batches)]


@torch.no_grad()
def evaluate(model, batches, passes):
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
    return math.exp(total / count)


def zero_params(model, predicate):
    names = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if predicate(name):
                p.zero_()
                names.append(name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-pass", type=int, default=3)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Inkling2LoopTextConfig(**ckpt["config"])
    model = Inkling2LoopForCausalLM(cfg).cuda()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    data = np.fromfile(args.val, dtype=np.uint16)
    batches = fixed_batches(data, args.eval_batch, args.ctx, args.eval_batches)
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    results = {"checkpoint": args.checkpoint, "trained_passes": cfg.loop_num_passes,
               "eval_pass": args.eval_pass, "baseline": {}, "ablations": {}}
    for p in (1, 2, 3, 4):
        results["baseline"][str(p)] = evaluate(model, batches, p)
        print(f"baseline pass {p}: {results['baseline'][str(p)]:.3f}", flush=True)

    def run_ablation(name, predicate):
        model.load_state_dict(base_state)
        names = zero_params(model, predicate)
        ppl = evaluate(model, batches, args.eval_pass)
        results["ablations"][name] = {
            "ppl": ppl, "delta_pct": 100.0 * (ppl / results["baseline"][str(args.eval_pass)] - 1.0),
            "params": len(names), "examples": names[:6],
        }
        print(f"{name}: ppl={ppl:.3f} delta={results['ablations'][name]['delta_pct']:.1f}% "
              f"({len(names)} tensors)", flush=True)

    run_ablation("message_gates", lambda n: "_message_gate" in n)
    run_ablation("message_read", lambda n: n == "model.message_input.query")
    run_ablation("block_queries", lambda n: n.endswith((".attention_input.query", ".mlp_input.query", ".final_input.query")))
    run_ablation("both", lambda n: "_message_gate" in n or n.endswith((".attention_input.query", ".mlp_input.query", ".final_input.query")))

    Path(args.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
