# Inkling 2 loop forensics

Independent analysis of **Inkling 2 Mini** (`thinkingmachines/Inkling-2-Mini`), a 26.4 B
looped-MoE transformer, and of the loop as a portable architecture for the Scion 35B-A3B.

This repository is the analysis home for the model itself: recovered architecture, tensor
map, precision layout, loop semantics, and the questions the study must answer. It
redistributes no model weights. It is not affiliated with Thinking Machines Lab.

**Status:** Phase B — loop study. The architecture and tensor layout are recovered; the
first loop-depth result is in [`docs/LOOP-STUDY.md`](docs/LOOP-STUDY.md).

## What the model is

| property | value |
|---|---|
| total / active | 26.389 B / 3.4 B (5.4 B effective with 3 passes) |
| physical layers | 24 |
| looped core | layers 6–17 (12 layers), run 3× |
| layer applications | 48 per token |
| attention | hybrid; 20 sliding (window 1024) + 4 global (5, 11, 17, 23) |
| positional | **no RoPE**; learned relative-bias bank + log scaling + short convolutions |
| residual | block attention residual: a bank of ≤4 states read by learned queries |
| MoE | 128 experts, top-8, sigmoid router, route scale 8, softcapped SwiGLU |
| experts | 24.4612 B = 92.7 % of the 26.07 B text parameters |
| precision | layers 0–19 NVFP4, 20–23 MXFP8; attention/embed BF16 |
| vision | 23 ViT layers, 0.318 B (not needed for the text analysis) |
| tokenizer | 201024 padded / 200058 unpadded; eos 200006 |

## Findings at a glance

- **The loop is a reset, not a recurrence.** Each pass restarts from the pre-loop hidden
  state and block bank. Cross-pass information flows only through four learned message
  queries per core layer (q, k, v, mlp), read from the previous pass's *last core block*
  and final hidden state. KV cache is per pass (48 cache slots for 24 physical layers).
- **No RoPE.** Position comes from a hidden-conditioned relative-bias bank (0–1023), log
  scaling of q and relative states beyond 8192, causal short convolutions, and a 1024
  sliding window. This is the main reason the architecture is not expressible with stock
  transformer attention.
- **The residual stream is a bank.** Every six layers the accumulated state is appended to
  a block bank instead of being carried in one residual stream; attention/MLP inputs are
  read from the bank by learned queries (three on global layers, one elsewhere). Maximum
  bank size is four blocks; reads see at most five vectors.
- **The core is trained for the loop.** The 12-layer core is applied three times and the
  message gates are trained. Measured on the same weights: PPL **6.53** at 3 passes,
  **10.33** at 2, **9.91** at 4; the six-prompt probe is 6/6 clean at 3 and ~3/6 at 2 and 4.
  A loop added post-hoc to a non-looped model is not expected to reproduce the gain. This is
  the central risk for the 35B port.
- **The cross-pass gates are the mechanism.** Zeroing the 144 message-gate weights (113 MB)
  doubles PPL (6.53 → 13.29) and drops the probe from 6/6 to 2/6, while the passes, block
  residual and per-pass KV still run. The loop is not a repeated block; the learned message
  channel is what makes later passes useful.
- **Component order.** Zeroing the block-read queries is catastrophic (PPL 731), the learned
  relative-position bias is next (245), then the loop controllers (13.3); attention sinks are
  secondary (7.21, still 6/6). The learned block-attention read and the learned position are
  the architecture.
- **Quantization is expert-bound, and the loop survives it.** IQ4_XS and Q8_K_XL have the
  same PPL (6.53 vs 6.53) because both keep the dominant layers 0–19 experts at NVFP4. At
  2-bit (`Q2_K_XL`, IQ1/IQ2 experts) the trained 3-pass model still gives 6.73 PPL (+2.9 %)
  and 6/6 on the probe; the loop ordering and the component hierarchy are unchanged. The
  quant gap is smallest at the trained 3 passes and widens off-schedule.
- **The drafter is a dense 3-layer block model** with a 256-rank RNN Markov head, four
  target taps (5/23/41/48), block 8, and shared target embeddings/output. Tensor map and
  forward-pass hypothesis: [`docs/DRAFTER-LAYOUT.md`](docs/DRAFTER-LAYOUT.md).
- **The drafter is a separate 3-layer DSpark block model** (`mtp.safetensors`), tapping
  application layers 5, 23, 41 and 48, block size 8, Markov rank 256. Its exact math is
  open; see `docs/OPEN-QUESTIONS.md`.

## Repository map

| path | contents |
|---|---|
| `docs/ARCHITECTURE.md` | recovered architecture, execution order, tensor/precision map |
| `docs/GGUF-LAYOUT.md` | GGUF metadata, tensor names, expert packing, drafter tensors |
| `docs/OPEN-QUESTIONS.md` | the analysis questions this repo exists to answer |
| `docs/ROADMAP.md` | loop study, drafter reconstruction, 35B implications |
| `docs/REPRODUCIBILITY.md` | artifacts, hashes, tools, commands |
| `vendor/` | config, quant config, safetensors index + 2 MB header, chat template |
| `eval/` | frozen corpus and prompt set |
| `scripts/` | tensor-map parser, size budget, toy loop trace, loop-depth runner |
| `results/` | tensor map JSON, HF state-dict names, loop-depth results |

Large artifacts (GGUFs, safetensors, logs) are gitignored and live outside the repository;
see `vendor/MANIFEST.md` for URLs, sizes and hashes.

## Scope

This repository is **model analysis only**. Runtime integration, the Scion fork branch,
and benchmark results belong to the Scion lane (kept outside this repo). Nothing here
depends on our engine port; the scripts read metadata only.

## Quick reproduction

```bash
# architecture/tensor map (config + 2 MB safetensors header only)
python3 scripts/map_tensors.py

# exact size budget from the header
python3 scripts/size_budget.py

# loop semantics on a 0.6 M-parameter random model (CPU, no weights)
python3 scripts/trace_loop.py
```

The model is Apache-2.0. The vendor config/header files here are metadata kept for
interoperability and analysis; no weights are redistributed.
