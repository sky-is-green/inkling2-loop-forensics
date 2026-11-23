#!/usr/bin/env python3
"""P0-01: parse vendor safetensors header + config into a tensor/precision map.

Reads only the 2 MB header prefix and config/index JSON. Never loads weights.
"""
import json
import struct
import collections
import sys
from pathlib import Path

V = Path(__file__).resolve().parent.parent / "vendor"
OUT = Path(__file__).resolve().parent.parent / "results"


def load_header(path):
    b = path.read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8 : 8 + n])
    h.pop("__metadata__", None)
    return h


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def main():
    h = load_header(V / "inkling-header.bin")
    cfg = json.load(open(V / "inkling-config.json"))
    t = cfg["text_config"]
    loop = t["loop_config"]
    qcfg = json.load(open(V / "inkling-hf_quant_config.json"))

    layers = sorted({int(k.split(".")[3]) for k in h if k.startswith("model.llm.layers.")})
    assert layers == list(range(24)), layers
    global_layers = [i for i in range(24) if (i + 1) % 6 == 0]

    per_layer = {}
    for i in layers:
        pre = f"model.llm.layers.{i}."
        names = {k[len(pre):] for k in h if k.startswith(pre)}
        exp = [n for n in names if n.startswith("mlp.experts.")]
        quant = "NVFP4" if any("amax" in n for n in exp) else "MXFP8"
        nv = sum(numel(h[pre + n]["shape"]) for n in names if n.startswith("mlp.experts."))
        # logical (dequantized) expert params: packed 4-bit halves the last dim
        def logical_elems(name):
            shp = h[pre + name]["shape"]
            cols = shp[-1] * 2 if h[pre + name]["dtype"] == "U8" else shp[-1]
            return shp[0] * shp[1] * cols

        logical = logical_elems("mlp.experts.w13_weight") + logical_elems("mlp.experts.w2_weight")
        attn = sum(numel(h[pre + n]["shape"]) for n in names if ".attn." in "." + n or n.startswith("attn."))
        non_exp = sum(numel(h[pre + n]["shape"]) for n in names if not n.startswith("mlp.experts."))
        per_layer[i] = {
            "attn": "global" if i in global_layers else "sliding",
            "loop": loop["core_start"] <= i < loop["core_end"],
            "expert_quant": quant,
            "expert_logical_params": logical,
            "expert_stored_bytes": nv,
            "layer_non_expert_params": non_exp,
            "tensors": sorted(names),
        }

    # param accounting
    text_layer_logical = sum(v["expert_logical_params"] + v["layer_non_expert_params"] for v in per_layer.values())
    other = {}
    for k in h:
        if k.startswith("model.llm.layers."):
            continue
        if k.startswith("model.visual."):
            continue
        other[k] = numel(h[k]["shape"])
    vision_params = sum(numel(h[k]["shape"]) for k in h if k.startswith("model.visual."))

    expert_logical_total = sum(v["expert_logical_params"] for v in per_layer.values())
    attn_res = sum(v for k, v in other.items() if "attn_res" in k)
    print("== param accounting (logical, BF16-dequantized where experts are quantized) ==")
    print(f"  expert logical params     {expert_logical_total/1e9:9.4f} B")
    print(f"  layer non-expert params    {(text_layer_logical-expert_logical_total)/1e9:9.4f} B")
    print(f"  attn_res + queries         {attn_res/1e6:9.4f} M")
    embed_params = sum(v for k, v in other.items() if 'attn_res' not in k)
    print(f"  embed+unembed+norm         {embed_params/1e9:9.4f} B")
    print(f"  text layers                {text_layer_logical/1e9:9.4f} B")
    print(f"  text total                 {(text_layer_logical + embed_params)/1e9:9.4f} B")
    print(f"  vision total               {vision_params/1e9:9.4f} B")
    print(f"  text+vision                {(text_layer_logical+embed_params+vision_params)/1e9:9.4f} B")

    print("\n== per-layer ==")
    print(f"{'L':>2} {'attn':<8} {'loop':<5} {'expertq':<7} {'expert Gp':>9} {'nonexp Mp':>10}")
    for i in layers:
        v = per_layer[i]
        print(f"{i:>2} {v['attn']:<8} {str(v['loop']):<5} {v['expert_quant']:<7} "
              f"{v['expert_logical_params']/1e9:9.3f} {v['layer_non_expert_params']/1e6:10.2f}")

    print("\n== quant method groups (hf_quant_config) ==")
    for grp, body in qcfg.items():
        if not isinstance(body, dict):
            print(" ", grp, body)
            continue
        for pg in body.get("parameter_groups", []):
            print(f"  {grp}: compute={pg.get('compute_dtype')} out={pg.get('output_dtype')} "
                  f"act_q={pg.get('required_activation_quantization')} n={len(pg.get('tensors', []))}")

    OUT.mkdir(exist_ok=True)
    out = {
        "source": {
            "repo": "thinkingmachines/Inkling-2-Mini",
            "revision": json.load(open(V / "inkling-api.json"))["sha"],
            "total_size_bytes": json.load(open(V / "inkling-model.safetensors.index.json"))["metadata"]["total_size"],
        },
        "config": cfg,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "accounting": {
            "expert_logical_params": expert_logical_total,
            "text_total_params": text_layer_logical + embed_params,
            "vision_total_params": vision_params,
            "attn_res_params": attn_res,
        },
    }
    (OUT / "tensor-map.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT/'tensor-map.json'}")


if __name__ == "__main__":
    sys.exit(main())
