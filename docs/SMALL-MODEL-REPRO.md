# Small-model reproduction: the loop is a trained prior

**Status: done (2026-11-25).** We trained the reference `transformers==5.24.0`
`inkling2_loop` model from scratch at 23.4M parameters on WikiText-103 and reproduced
the qualitative depth/ablation behavior of the 26B model. The point is not language
capability; it is to separate a trained loop from a runtime trick.

Artifacts: `results/tiny-loop-repro.json`, `scripts/{prepare_wikitext,train_tiny_loop,ablate_tiny_loop,plot_tiny_loop}.py`.

## Design

A faithful scale-down of the released architecture:

| property | value |
|---|---|
| physical layers | 8 |
| loop core | layers 2–6 (4 layers, 2 blocks of 2) |
| passes | 3 (control: 1) |
| applications | 16 (control: 8) |
| hidden | 256 |
| heads / KV heads / head dim | 8 / 4 / 32 |
| experts | 8 routed, top-2, intermediate 256 |
| attention | hybrid, sliding window 128, full every 6th layer |
| block bank | 2-layer blocks, max 4 states |
| message gates | rank 8, q/k/v/mlp per core layer |
| vocab | 16,384 byte-level BPE |
| parameters | 23.42M |

Ratios preserved: core = 50 % of layers, two loop blocks, three passes, four total
blocks, message rank ≈ 3 % of hidden.

Data: WikiText-103 raw, 122.7M train tokens, 257k validation, 16k byte-level BPE.
Training: 120M tokens, batch 16 × 256, AdamW 3e-4, cosine to 10 %, 500-step warmup,
weight decay 0.1, grad clip 1.0, seed 0. A 1-pass control used the same data, seed
and token budget.

## Depth at 120M tokens

| eval passes | PPL | vs 3 |
|---:|---:|---:|
| 1 | 52.81 | +44.3 % |
| 2 | 40.75 | +11.3 % |
| **3** | **36.61** | — |
| 4 | 37.42 | +2.2 % |

Three is best. The 4-pass penalty is much smaller than on the 26B model (+2.2 % vs
+52 %); the depth preference is real but scale/training-dependent. The 1- and 2-pass
penalties are large.

## Post-hoc loop control

Same 82M-token checkpoint for both models:

| model | eval 1 | eval 2 | eval 3 | eval 4 |
|---|---:|---:|---:|---:|
| trained 3-pass | 56.38 | 44.80 | **40.81** | 41.47 |
| trained 1-pass | **41.41** | 41.45 | 41.44 | 41.45 |

The 1-pass model is flat across 1/2/3/4 passes. Extra inference passes do not help a
model trained without the loop. The 3-pass model is much worse at 1 pass (56.38)
because it expects the trained schedule, and improves as the trained passes are
restored.

The 1-pass control reached 90M tokens before a display-GPU hang stopped it; at 90M
it was still flat (40.33 / 40.32 / 40.35 / 40.32). The 82M checkpoint is the
reproducible comparison; the 90M eval is recorded in `results/tiny-loop-repro.json`.

## Ablations (3-pass model, 3 passes)

| ablation | PPL | Δ | tensors |
|---|---:|---:|---:|
| baseline | 38.92 | — | — |
| message gates zeroed | 56.41 | +44.9 % | 48 |
| shared message read zeroed | 39.53 | +1.6 % | 1 |
| block-bank queries zeroed | 381.05 | +879 % | 16 |
| both | 280.15 | +620 % | 64 |

The block-bank read is the architecture. The per-layer message gates matter a lot;
the shared message-read query alone is weak. The hierarchy matches the 26B result,
with smaller magnitudes.

## Findings

1. **The loop is learned.** The trained depth is the best depth; extra passes do not
   rescue a model that was not trained for them.
2. **The bank and gates are load-bearing at small scale too.** This is not a 26B
   checkpoint artifact.
3. **The architecture is portable, but the weights are not.** A 35B loop is a
   training project — initialize from the model's own core, add the bank/messages,
   and teach the loop — not an inference flag.
4. **The 3-vs-4 margin is scale-dependent.** At 23M the 4-pass penalty is only 2.2 %,
   much smaller than at 26B. A 35B pilot must measure the depth preference rather
   than assume the 26B gap.

## Caveats

- 23M parameters, one seed, WikiText-103, 256-token context.
- 16k BPE and a 128-token sliding window are scaled ratios, not the 26B model's exact
  tokenizer/window.
- The 1-pass control was stopped at 90M tokens by a display-GPU hang; the 82M
  checkpoint is the reproducible comparison.
- The ablation baseline (38.92, 8 eval batches) and the depth table (36.61, 4 eval
  batches) use different validation subsets.
- This is a mechanism reproduction, not a capability or long-context claim.

## Reproduce

Set `DATA_DIR` to wherever you keep the raw WikiText-103 parquet files and `RUN_DIR` to
where you want checkpoints/results. Neither is included in this repository.

```bash
# data (outside this repo)
python3 scripts/prepare_wikitext.py \
  --data-dir $DATA_DIR \
  --out-dir  $DATA_DIR/tokenized

# 3-pass model
python3 scripts/train_tiny_loop.py \
  --train  $DATA_DIR/tokenized/train.u16 \
  --val    $DATA_DIR/tokenized/validation.u16 \
  --manifest $DATA_DIR/tokenized/manifest.json \
  --out $RUN_DIR/tiny-loop-p3 \
  --passes 3 --tokens 120000000

# 1-pass control
python3 scripts/train_tiny_loop.py \
  --train  $DATA_DIR/tokenized/train.u16 \
  --val    $DATA_DIR/tokenized/validation.u16 \
  --manifest $DATA_DIR/tokenized/manifest.json \
  --out $RUN_DIR/tiny-loop-p1 \
  --passes 1 --tokens 120000000

# ablations and figure
python3 scripts/ablate_tiny_loop.py \
  --checkpoint $RUN_DIR/tiny-loop-p3/model.pt \
  --val $DATA_DIR/tokenized/validation.u16 \
  --manifest $DATA_DIR/tokenized/manifest.json \
  --out $RUN_DIR/tiny-loop-p3/ablation.json
python3 scripts/plot_tiny_loop.py \
  --p3 $RUN_DIR/tiny-loop-p3/results.json \
  --p1 $RUN_DIR/tiny-loop-p1/results.json \
  --ablation $RUN_DIR/tiny-loop-p3/ablation.json \
  --out $RUN_DIR/tiny-loop-p3/curve
```

Data and checkpoints stay outside the public repo; only the tokenizer, scripts and
curated results are here.
