#!/usr/bin/env python3
"""Zero the loop controllers of an Inkling 2 GGUF copy (cross-pass message ablation).

With `loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` all zero, the message gate reduces to
`(1 - 0) * cur + 0 * message = cur`: the passes still run, but no cross-pass information
is mixed in. The block-attention residual is untouched.

Work on a copy; this edits tensor payloads in place.

  cp --reflink=auto model.gguf model-nocontrollers.gguf
  python3 scripts/zero_loop_controllers.py --model model-nocontrollers.gguf
"""
import argparse
import os
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"\.loop_(q|k|v|ffn)_ctrl_(down|gain|forget)\.weight$")
CHUNK = 1 << 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        from gguf import GGUFReader
    except ImportError:
        print("gguf package not found (pip install gguf)", file=sys.stderr)
        return 2

    path = Path(args.model)
    reader = GGUFReader(str(path))
    targets = [(t.name, t.data_offset, t.n_bytes) for t in reader.tensors if PATTERN.search(t.name)]
    if not targets:
        print("no loop controller tensors found; is this an inkling2_loop GGUF?", file=sys.stderr)
        return 1

    total = sum(n for _, _, n in targets)
    print(f"{len(targets)} controller tensors, {total/1e6:.1f} MB")
    for name, off, n in targets[:3]:
        print(f"  {name} @ {off} ({n} B)")
    if args.dry_run:
        return 0

    del reader  # drop the read-only mmap before writing
    zeros = b"\0" * CHUNK
    with open(path, "r+b") as f:
        for _, off, n in targets:
            f.seek(off)
            left = n
            while left > 0:
                w = min(left, CHUNK)
                f.write(zeros[:w])
                left -= w

    # verify a sample
    reader = GGUFReader(str(path))
    import numpy as np
    for name, _, _ in targets[:3]:
        t = next(t for t in reader.tensors if t.name == name)
        arr = np.asarray(t.data)
        print(f"  verify {name}: max|w| = {float(np.max(np.abs(arr))):.6g}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
