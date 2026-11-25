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

Results so far (see `LOOP-STUDY.md`):
- loop depth is trained: PPL 10.33 / **6.53** / 9.91 for 2/3/4 passes;
- component order: block-read queries (731) > relative bias (245) > loop controllers (13.3)
  > sinks (7.2);
- quantization is expert-bound: IQ4_XS ≈ Q8_K_XL because both keep experts 0–19 at NVFP4;
  at 2-bit the trained loop still gives 6.73 PPL and 6/6, with the same component hierarchy;
- block residual: replacing the read with a plain residual gives PPL 12.9 M, the strongest
  component result;
- long context: at a fixed 2k target, loop-3 PPL improves from 7.11 to 6.54 (Q2) and 6.88 to
  6.33 (Q8) from 4k to 32k; loop 3 is the only depth that answers 32k needles (3/3, both
  quants), while loops 2/4 give 0/3 even with 2048 generated tokens; sinks cost ~6–8 % PPL
  but do not gate retrieval. See `LONG-CONTEXT.md`;
- small-model reproduction (23.4M, WikiText-103, reference implementation): the 3-pass
  model is best at 3 (36.61 PPL at 120M), a 1-pass control is flat across 1/2/3/4 at 82M,
  and block-bank/message ablations reproduce. See `SMALL-MODEL-REPRO.md`;
- drafter inventory: `DRAFTER-LAYOUT.md` (3 dense layers, 256-rank RNN Markov head).

Remaining deliverables:
- 64k/128k long-context check on Q2 if memory allows, plus a harder long-document QA set;
- a 35B loop transplant pilot: initialize the loop from the 35B's own core, add the
  bank/messages, train with distillation, and measure the depth preference at scale;
- why loops 2/4 fail at 32k: is it the message channel, the block bank, or the pass schedule?
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

Only after B/C. The small-model result makes the training-first path the default: a
runtime-only loop on the existing weights is not supported by the evidence.

- core selection (8–19, 10–29, 14–25 or a block-aligned span), 2 vs 3 passes;
- distillation from the 35B BF16 teacher, deep supervision across passes;
- eval on hard-subset30, coding10, PPL, tools; capability outranks PPL;
- pod proposal and approval before any training.

## Rules

- No weights in this repo; large artifacts stay outside and are listed in
  `vendor/MANIFEST.md`.
- One variable per experiment; a control alongside every candidate.
- Every finding goes to `docs/`; no silent pivots.
