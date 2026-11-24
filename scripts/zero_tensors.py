#!/usr/bin/env python3
"""Zero a named group of tensors in an Inkling 2 GGUF copy (weight-space ablations).

Groups:
  loop_controllers  all 144 loop_{q,k,v,ffn}_ctrl_{down,gain,forget} weights
                    -> message gate becomes (1-0)*cur + 0*message = cur
  relpos            attn_rel (wr_du) + attn_rel_bias (proj) in every layer
                    -> learned relative-position bias becomes zero
  sinks             attn_sink_proj in every layer -> no per-token attention sink
  res_queries       attn_res_query + ffn_res_query + output_res_query
                    -> block-attention read becomes a uniform average over the bank

Work on a copy; this edits tensor payloads in place.

  cp --reflink=auto model.gguf model-abl.gguf
  python3 scripts/zero_tensors.py --model model-abl.gguf --group relpos
"""
import argparse
import os
import re
import sys
from pathlib import Path

GROUPS = {
    "loop_controllers": re.compile(r"\.loop_(q|k|v|ffn)_ctrl_(down|gain|forget)\.weight$"),
    "relpos":           re.compile(r"\.attn_rel(_bias)?\.weight$"),
    "sinks":            re.compile(r"\.attn_sink_proj\.weight$"),
    "res_queries":      re.compile(r"(\.attn_res_query|\.ffn_res_query|^output_res_query)\.weight$"),
}
CHUNK = 1 << 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        from gguf import GGUFReader
        import numpy as np
    except ImportError:
        print("gguf / numpy not found (pip install gguf numpy)", file=sys.stderr)
        return 2

    pattern = GROUPS[args.group]
    path = Path(args.model)
    reader = GGUFReader(str(path))
    targets = [(t.name, t.data_offset, t.n_bytes) for t in reader.tensors if pattern.search(t.name)]
    if not targets:
        print(f"no tensors matched group {args.group!r}", file=sys.stderr)
        return 1

    total = sum(n for _, _, n in targets)
    print(f"group {args.group}: {len(targets)} tensors, {total/1e6:.1f} MB")
    for name, off, n in targets[:4]:
        print(f"  {name} @ {off} ({n} B)")
    if args.dry_run:
        return 0

    del reader
    zeros = b"\0" * CHUNK
    with open(path, "r+b") as f:
        for _, off, n in targets:
            f.seek(off)
            left = n
            while left > 0:
                w = min(left, CHUNK)
                f.write(zeros[:w])
                left -= w

    reader = GGUFReader(str(path))
    for name, _, _ in targets[:3]:
        t = next(t for t in reader.tensors if t.name == name)
        arr = np.asarray(t.data)
        print(f"  verify {name}: max|w| = {float(np.max(np.abs(arr))):.6g}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
