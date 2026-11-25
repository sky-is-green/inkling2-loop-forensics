#!/usr/bin/env python3
"""Build deterministic needle-in-a-haystack prompts for the long-context study.

The haystack is a contiguous slice of the frozen wikitext-2 test corpus.  Character
budgets are deliberately conservative (about 3.8 chars/token for this tokenizer) so
the prompts stay below the requested context even after the question is appended.

Outputs one text file per (length, depth) and a manifest with the exact byte counts,
sha256 hashes, needle and expected answer.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

NEEDLE = "The secret access code for the vault is 7391-ALPHA."
QUESTION = "What is the secret access code for the vault? Answer with the code only."
EXPECTED = ["7391-alpha", "7391"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="8192,16384,32768")
    ap.add_argument("--depths", default="10,50,90")
    ap.add_argument("--chars-per-token", type=float, default=3.8)
    args = ap.parse_args()

    raw = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    # Normalize the wiki markup just enough to keep the haystack prose-like.
    raw = raw.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for length in [int(x) for x in args.lengths.split(",") if x.strip()]:
        target_chars = int(length * args.chars_per_token)
        if len(raw) < target_chars + 2000:
            raise SystemExit(f"corpus too short for {length} tokens: {len(raw)} chars")
        haystack = raw[:target_chars]
        for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
            cut = max(0, min(len(haystack), int(len(haystack) * depth / 100)))
            # Put the needle on a paragraph boundary near the requested depth.
            cut = haystack.rfind("\n", 0, cut) if cut > 0 else 0
            if cut < 0:
                cut = 0
            prompt = (
                haystack[:cut].rstrip()
                + "\n\n"
                + NEEDLE
                + "\n\n"
                + haystack[cut:].lstrip()
                + "\n\nQuestion: "
                + QUESTION
                + "\nAnswer:"
            )
            name = f"needle-{length}-d{depth:02d}.txt"
            path = out / name
            path.write_text(prompt, encoding="utf-8")
            manifest.append({
                "id": f"needle-{length}-d{depth:02d}",
                "length": length,
                "depth": depth,
                "chars": len(prompt),
                "bytes": path.stat().st_size,
                "sha256": sha256_text(prompt),
                "haystack_chars": len(haystack),
                "needle": NEEDLE,
                "question": QUESTION,
                "expected": EXPECTED,
                "path": str(path),
            })
            print(f"{name}: {len(prompt)} chars, sha256 {manifest[-1]['sha256'][:16]}")

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
