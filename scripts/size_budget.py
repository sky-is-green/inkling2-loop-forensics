#!/usr/bin/env python3
"""Exact GGUF size budget from the vendor header, per quantization policy.

No weights needed: the 148 KB safetensors header gives every tensor's dtype/shape.
GGUF sizes: Q4_0 = 18 B / 32 weights (4.5 bpw), Q8_0 = 34 B / 32 (8.5 bpw),
F16 = 2 B/weight, F32 = 4 B/weight, BF16 = 2 B/weight.
"""
import json
import struct
from pathlib import Path

V = Path(__file__).resolve().parent.parent / "vendor"
b = (V / "inkling-header.bin").read_bytes()
n = struct.unpack("<Q", b[:8])[0]
h = json.loads(b[8 : 8 + n])
h.pop("__metadata__", None)


def nbytes(shape, bpw):
    n = 1
    for d in shape:
        n *= d
    return n * bpw / 8


def logical_last(shape, dtype):
    shp = list(shape)
    if dtype == "U8":
        shp[-1] *= 2
    return shp


def is_text(k):
    return k.startswith("model.llm.") and not k.startswith("model.visual.")


def is_quant_aux(k):
    return ".mlp.experts." in k and (
        k.endswith("_scale") or k.endswith("_amax") or k.endswith("_outer_scale_denominator")
    )


def is_expert(k):
    # Only the packed weights survive conversion; the NVFP4/MXFP8 scale/amax/denominator
    # tensors are consumed by the dequant and do not appear in the GGUF.
    return k.endswith(".mlp.experts.w13_weight") or k.endswith(".mlp.experts.w2_weight")


def is_embed(k):
    return k in ("model.llm.embed.weight", "model.llm.unembed.weight")


def is_linear(k):
    return k.endswith(".weight") and len(h[k]["shape"]) >= 2


def classify(k):
    if is_expert(k):
        return "expert"
    if is_embed(k):
        return "embed"
    if is_linear(k):
        if any(s in k for s in ("wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn", "wr_du", "attn_sink")):
            return "attn"
        return "other_linear"
    return "small"


POLICIES = {
    # name: (expert, attn, other_linear, embed, small)
    "Q4_0 floor (all linears 4.5bpw)":
        dict(expert=4.5, attn=4.5, other_linear=4.5, embed=4.5, small=32.0),
    "Q4_0 realistic (attn F16, embed Q8)":
        dict(expert=4.5, attn=16.0, other_linear=16.0, embed=8.5, small=32.0),
    "Q4_K_M-like (4.85bpw, attn F16, embed Q8)":
        dict(expert=4.85, attn=16.0, other_linear=16.0, embed=8.5, small=32.0),
    "Q4_0 + attn Q8 + embed Q8":
        dict(expert=4.5, attn=8.5, other_linear=8.5, embed=8.5, small=32.0),
    "Q8_0 text-only":
        dict(expert=8.5, attn=8.5, other_linear=8.5, embed=8.5, small=32.0),
}

counts = {c: 0 for c in ("expert", "attn", "other_linear", "embed", "small")}
for k in h:
    if not is_text(k) or is_quant_aux(k):
        continue
    counts[classify(k)] += 1

print("tensor classes (text-only):", counts)
print()
for name, pol in POLICIES.items():
    tot = 0.0
    parts = {}
    for k, meta in h.items():
        if not is_text(k) or is_quant_aux(k):
            continue
        c = classify(k)
        shp = logical_last(meta["shape"], meta["dtype"])
        v = nbytes(shp, pol[c])
        tot += v
        parts[c] = parts.get(c, 0) + v
    print(f"{name:52s} {tot/1e9:6.2f} GB")
    print("    " + "  ".join(f"{c}={parts[c]/1e9:.2f}" for c in parts))
print()
print("vision (for reference, if ever included):")
vis = sum(nbytes(logical_last(m["shape"], m["dtype"]), 4.5) for k, m in h.items() if k.startswith("model.visual."))
print(f"    at 4.5 bpw = {vis/1e9:.2f} GB")
