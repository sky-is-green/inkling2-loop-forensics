# Reproducibility

## Artifacts

Vendor metadata and hashes: [`../vendor/MANIFEST.md`](../vendor/MANIFEST.md). The 2 MB
`inkling-header.bin` is the safetensors header prefix: it contains every tensor name, shape
and dtype, no payloads.

The vendor checkpoint itself is not in this repo:

```
thinkingmachines/Inkling-2-Mini @ b85c3cfa615fc413cc9af3379f491b963f3b586a
model-00001-of-00001.safetensors   19 526 019 168 B
sha256 65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1
```

## Reference runtime

`transformers==5.24.0` is the first release that ships `models/inkling2_loop`
(5.23.0 has `inkling2` but not the loop variant). Wheel sha256
`ff5ebedf9359f94bb63772b6380eba6e7104d010cbbb2cb82c788817ea7486f6`.

The scripts here do not need the weights. For HF reference work:

```bash
python3.12 -m venv venv
venv/bin/pip install torch --extra-index-url ...   # CPU is enough for inspection
venv/bin/pip install transformers==5.24.0
```

Do not instantiate the 26 B model on a 30 GB host: it is ~52 GB in BF16. Use
`torch.device("meta")` for name/shape work, or `results/hf-state-dict-names.json`.

## Analysis scripts

```bash
# tensor/precision map from the 2 MB safetensors header (no weights)
python3 scripts/map_tensors.py        # writes results/tensor-map.json

# exact size budget for Q4_0/Q8_0 policies
python3 scripts/size_budget.py

# loop/block/message semantics on a 0.6 M-parameter random model (CPU)
python3 scripts/trace_loop.py

# long-context eval data (frozen corpus, needles, fixed-target slices)
python3 scripts/build_needle_prompts.py \
  --corpus eval/long-context/wikitext-2-raw/wiki.test.raw \
  --out eval/long-context/prompts
python3 scripts/make_fixed_target_slices.py
```

Dependencies: `numpy` for the scripts; `torch` + `transformers==5.24.0` only for
`trace_loop.py`. See `requirements.txt`.

## Long-context artifacts

- `eval/long-context/wikitext-2-raw/wiki.test.raw` — WikiText-2 test split, 312,991 tokens,
  sha256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
- `eval/long-context/prompts/manifest.json` — 9 needle prompts (8k/16k/32k × 10/50/90 %),
  hashes and expected answer.
- `eval/long-context/fixed-target/manifest.json` — four slices ending at the same paragraph
  boundary; nominal 4k/8k/16k/32k, actual `n_ctx` 5,120/9,216/17,408/33,792.
- `results/long-context.json` — curated PPL, needle, sink and control results.

The raw run logs and the full per-prompt records are kept outside this repo. PPL at 32k
used a streaming-NLL strided perplexity pass so the full-context logits tensor was never
materialised on the host.

## Provenance

Every tensor count, dtype and shape in `docs/ARCHITECTURE.md` comes from
`vendor/inkling-header.bin` or from a meta-device build of `transformers` 5.24.0. The
long-context numbers in `results/long-context.json` were measured on the unsloth GGUFs with
the patched llama.cpp runtime; they are model behaviour, not runtime/port benchmarks.
