#!/usr/bin/env python3
"""Plot the tiny loop reproduction: depth curves and ablations.

Reads the `results.json` written by train_tiny_loop.py and the `ablation.json`
written by ablate_tiny_loop.py.  Produces a two-panel SVG/PNG suitable for the
write-up.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path | None):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", required=True)
    ap.add_argument("--p1", default=None)
    ap.add_argument("--ablation", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p3 = load(Path(args.p3))
    p1 = load(Path(args.p1)) if args.p1 else None
    ablation = load(Path(args.ablation)) if args.ablation else None
    if p3 is None:
        raise SystemExit("missing p3 results")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = {1: "#777777", 2: "#d95f02", 3: "#1b9e77", 4: "#7570b3"}
    for model, style, label in ((p3, "-", "trained 3-pass"), (p1, "--", "trained 1-pass")):
        if model is None:
            continue
        evals = model["evals"]
        xs = [e["tokens"] / 1e6 for e in evals]
        for p in (1, 2, 3, 4):
            ys = [e["val_ppl"][str(p)] for e in evals]
            ax.plot(xs, ys, style, color=colors[p], lw=2 if p == 3 else 1.4,
                    label=f"{label}, eval {p} pass{'es' if p != 1 else ''}")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("validation PPL")
    ax.set_yscale("log")
    ax.set_title("Depth curve during training")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    width = 0.36
    for i, (model, label) in enumerate(((p3, "trained 3-pass"), (p1, "trained 1-pass"))):
        if model is None:
            continue
        last = model["evals"][-1]["val_ppl"]
        xs = [1, 2, 3, 4]
        ys = [last[str(p)] for p in xs]
        ax.bar([x + (i - 0.5) * width for x in xs], ys, width=width, label=label)
        for x, y in zip(xs, ys):
            ax.text(x + (i - 0.5) * width, y, f"{y:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("passes at inference")
    ax.set_ylabel("final validation PPL")
    ax.set_title("Trained depth vs inference depth")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"wrote {out.with_suffix('.svg')} and {out.with_suffix('.png')}")

    if ablation:
        labels = ["baseline"] + list(ablation["ablations"].keys())
        vals = [ablation["baseline"][str(ablation["eval_pass"])]] + [
            v["ppl"] for v in ablation["ablations"].values()
        ]
        fig2, ax2 = plt.subplots(figsize=(7, 3.8))
        bars = ax2.bar(labels, vals, color=["#1b9e77"] + ["#d95f02"] * (len(vals) - 1))
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel(f"validation PPL at {ablation['eval_pass']} passes")
        ax2.set_title("Component ablations on the trained loop")
        ax2.grid(axis="y", alpha=0.25)
        fig2.tight_layout()
        fig2.savefig(out.with_name(out.stem + "-ablation.svg"))
        fig2.savefig(out.with_name(out.stem + "-ablation.png"), dpi=180)
        print(f"wrote {out.with_name(out.stem + '-ablation.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
