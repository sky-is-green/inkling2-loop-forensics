#!/usr/bin/env python3
"""Build fixed-target corpus slices for a clean context-length comparison.

For each requested context C the strided perplexity tool uses
n_ctx = C + stride/2 tokens and scores the last `stride` tokens of the first
n_ctx tokens.  Each slice is built so that its first n_ctx tokens end at the
same paragraph boundary; the text after that boundary is only there to satisfy
the tool's 2*n_ctx validation and is never scored.

Token counts come from llama-tokenize (same model tokenizer as the eval).
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path("/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/models-inkling/gguf/Mainline_Support/Inkling-2-Mini-UD-Q2_K_XL.gguf")
DEFAULT_TOKENIZE = Path("/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion-llama.cpp/build-hip-loop/bin/llama-tokenize")
DEFAULT_CORPUS = ROOT / "eval/long-context/wikitext-2-raw/wiki.test.raw"


def count_tokens(tokenize: Path, model: Path, text: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        proc = subprocess.run(
            [str(tokenize), "-m", str(model), "-f", path, "--show-count"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        )
        out = proc.stdout.decode("utf-8", "replace")
        m = re.search(r"Total number of tokens:\s*(\d+)", out)
        if not m:
            raise RuntimeError(f"could not parse token count from:\n{out[-500:]}")
        return int(m.group(1))
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--tokenize", default=str(DEFAULT_TOKENIZE))
    ap.add_argument("--out", default=str(ROOT / "eval/long-context/fixed-target"))
    ap.add_argument("--contexts", default="4096,8192,16384,32768")
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--target-char", type=int, default=160000)
    ap.add_argument("--extra-chars", type=int, default=300000)
    args = ap.parse_args()

    raw = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    end = raw.find("\n\n", args.target_char)
    if end < 0:
        end = args.target_char
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Path(args.model)
    tokenize = Path(args.tokenize)

    manifest = []
    for C in [int(x) for x in args.contexts.split(",") if x.strip()]:
        n_ctx = C + args.stride // 2
        # Initial estimate: 4.1 chars/token for this corpus.
        start = max(0, int(end - n_ctx * 4.1))
        best = None
        for it in range(30):
            first = raw[start:end]
            n = count_tokens(tokenize, model, first)
            print(f"C={C} iter={it} start={start} first_tokens={n} desired={n_ctx}", flush=True)
            if best is None or abs(n - n_ctx) < abs(best[1] - n_ctx):
                best = (start, n, first)
            if n == n_ctx:
                break
            delta = int((n - n_ctx) * 4.1)
            if delta == 0:
                delta = 1 if n > n_ctx else -1
            start = max(0, min(end - 1, start + delta))
        else:
            start, n, first = best
            print(f"C={C}: WARNING using closest prefix {n} vs desired {n_ctx}", flush=True)

        # Trailing text is only there to satisfy the tool's 2*n_ctx check.
        extra = raw[end:end + args.extra_chars]
        text = first + "\n\n" + extra
        total = count_tokens(tokenize, model, text)
        if total < 2 * n_ctx:
            raise SystemExit(f"C={C}: total {total} < 2*n_ctx {2*n_ctx}; increase --extra-chars")

        name = f"fixed-target-C{C}.txt"
        path = out / name
        path.write_text(text, encoding="utf-8")
        manifest.append({
            "context": C,
            "n_ctx": n_ctx,
            "first_tokens": n,
            "total_tokens": total,
            "target_char_end": end,
            "start_char": start,
            "chars": len(text),
            "target_text_sha256": hashlib.sha256(raw[max(0, end - 10000):end].encode("utf-8")).hexdigest(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "path": str(path.relative_to(ROOT)),
        })
        print(f"wrote {name}: first={n} total={total}, {len(text)} chars", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
