#!/usr/bin/env python3
"""Prepare WikiText-103 token arrays for the small loop reproduction.

Trains a 16k byte-level BPE on a sample of the training split, then tokenizes the
train/validation/test text into uint16 arrays.  Data stays outside the analysis
repository; only the tokenizer and this script are kept.

Example:
  python3 scripts/prepare_wikitext.py \
    --data-dir ~/Desktop/scion-loop-work/data/wikitext-103 \
    --out-dir  ~/Desktop/scion-loop-work/data/wikitext-103/tokenized
"""
import argparse
import hashlib
import json
from array import array
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer, Tokenizer, decoders, models, pre_tokenizers, trainers


def read_texts(path: Path) -> list[str]:
    table = pq.read_table(path, columns=["text"])
    return [x.as_py() for x in table.column("text")]


def bytes_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_text(tokenizer: Tokenizer, text: str, eos_id: int, chunk_chars: int = 5_000_000) -> array:
    out = array("H")
    for start in range(0, len(text), chunk_chars):
        chunk = text[start:start + chunk_chars]
        out.extend(tokenizer.encode(chunk, add_special_tokens=False).ids)
    out.append(eos_id)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-chars", type=int, default=50_000_000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = [data_dir / "train-00000-of-00002.parquet", data_dir / "train-00001-of-00002.parquet"]
    val_files = [data_dir / "validation-00000-of-00001.parquet"]
    test_files = [data_dir / "test-00000-of-00001.parquet"]

    train_text = "".join("".join(read_texts(p)) for p in train_files)
    val_text = "".join("".join(read_texts(p)) for p in val_files)
    test_text = "".join("".join(read_texts(p)) for p in test_files)
    print(f"chars train={len(train_text):,} val={len(val_text):,} test={len(test_text):,}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>", "<eos>"],
        min_frequency=2,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator([train_text[: args.sample_chars]], trainer=trainer)
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    print(f"tokenizer vocab={vocab_size} eos={eos_id} pad={pad_id}")
    if vocab_size > 65535:
        raise SystemExit("vocab too large for uint16 token arrays")
    tokenizer.save(str(out_dir / "tokenizer.json"))

    manifest = {"vocab_size": vocab_size, "eos_id": eos_id, "pad_id": pad_id,
                "sample_chars": args.sample_chars, "files": {}}
    for name, text in (("train", train_text), ("validation", val_text), ("test", test_text)):
        ids = encode_text(tokenizer, text, eos_id)
        arr = np.frombuffer(ids, dtype=np.uint16)
        path = out_dir / f"{name}.u16"
        arr.tofile(path)
        manifest["files"][name] = {
            "path": str(path), "tokens": int(len(arr)), "chars": len(text),
            "bytes": path.stat().st_size, "sha256": bytes_hash(path),
        }
        print(f"{name}: {len(arr):,} tokens -> {path} ({path.stat().st_size/1e6:.1f} MB)")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_dir/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
