# Roadmap

Phase 0 (seed) is done: the architecture and tensor layout are recovered. The next phases
are the actual analysis.

## Phase A — reference and artifacts (done)

- Reference runtime: `transformers==5.24.0` (first release with `inkling2_loop`), wheel hash
  recorded.
- Vendor config, quant config, safetensors index + header, chat template stored in `vendor/`.
- Architecture/tensor map in `docs/ARCHITECTURE.md`; HF names in `results/`.
- GGUF layout and unsloth metadata in `docs/GGUF-LAYOUT.md`.

## Phase B — loop study (in progress)

First result: loop depth is trained. PPL 10.33 / **6.53** / 9.91 for 2/3/4 passes; the
six-prompt probe is 6/6 clean at 3 and ~3/6 at 2 and 4. See `LOOP-STUDY.md`.

Remaining deliverables:
- loop 1/2/3 capability and compute table on a frozen prompt set;
- component ablations (message gates, block residual, per-pass KV, controllers, sinks,
  relative bias vs RoPE);
- quantization-compounding table (pass 1/2/3 drift, IQ4_XS vs Q8_K_XL);
- a minimal loop recommendation for the 35B.

Needs: a loop-count override in the runtime used for the study, and a frozen prompt/eval
protocol.

## Phase C — drafter reconstruction

Deliverables:
- exact DSpark math (tensors, block layout, fuse, Markov head) from the vendor
  `mtp.safetensors` + `mtp_config`;
- a reference implementation that reproduces the vendor drafter's drafts and acceptance;
- acceptance by prompt class and temperature.

This answers whether the drafter is reproducible from the released weights, or why not.

## Phase D — 35B loop design

Only after B/C.

- core selection (8–19, 10–29, 14–25), 2 vs 3 passes;
- distillation from the 35B BF16 teacher, deep supervision across passes;
- eval on hard-subset30, coding10, PPL, tools; capability outranks PPL;
- pod proposal and approval before any training.

## Rules

- No weights in this repo; large artifacts stay outside and are listed in
  `vendor/MANIFEST.md`.
- One variable per experiment; a control alongside every candidate.
- Every finding goes to `docs/`; no silent pivots.
