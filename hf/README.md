---
license: other
language:
- en
tags:
- inkling2
- looped-transformer
- mixture-of-experts
- mechanistic-interpretability
- long-context
pretty_name: Inkling 2 Loop Forensics
---

# Inkling 2 Loop Forensics

Independent analysis of the loop in **Inkling 2 Mini** (`thinkingmachines/Inkling-2-Mini`),
a 26.4B looped-MoE transformer. This is an analysis/artifact dataset, not a model release.
It contains no model weights, checkpoints or GGUFs.

The code and the full write-up live on GitHub:
https://github.com/sky-is-green/inkling2-loop-forensics

## What is here

- Recovered architecture, execution order, tensor map and precision layout.
- Loop-depth and component ablations on the released model.
- Quantization and 2-bit robustness results.
- A 32k long-context study: fixed-target PPL and needle retrieval.
- A 23.4M-parameter reproduction trained with the reference `inkling2_loop`
  implementation, including a 1-pass control and component ablations.
- The frozen WikiText-2 test corpus, needle prompts and fixed-target slices used above.

## Key results

**The loop is trained, not a runtime trick.** A 23.4M reference model trained with 3
passes is best at 3; a 1-pass control is flat across 1/2/3/4 inference passes. Zeroing the
block-bank reads or the message gates hurts.

**The loop is a long-context mechanism.** At a fixed 2k target, loop-3 PPL falls from
7.11 to 6.54 (Q2) and 6.88 to 6.33 (Q8) as context grows from 4k to 32k. At 32k, loop 3
answers 3/3 needle probes at 10/50/90% depth; loops 2/4 give 0/3 final answers even with a
2048-token budget.

**The loop survives severe compression.** Q2_K_XL costs +2.9% PPL at the trained 3-pass
point and keeps the component hierarchy.

## Figures

![Tiny-loop depth curve](figures/tiny-loop-depth.png)

![Tiny-loop ablations](figures/tiny-loop-ablations.png)

## Contents

| path | contents |
|---|---|
| `docs/ARCHITECTURE.md` | recovered architecture, execution order, tensor/precision map |
| `docs/LOOP-STUDY.md` | loop depth, components, quantization, 2-bit robustness |
| `docs/LONG-CONTEXT.md` | 32k fixed-target PPL, needle retrieval, sinks |
| `docs/SMALL-MODEL-REPRO.md` | 23.4M reference-model reproduction |
| `docs/DRAFTER-LAYOUT.md` | DSpark drafter tensor map and forward hypothesis |
| `docs/OPEN-QUESTIONS.md` | questions this analysis was built to answer |
| `docs/REPRODUCIBILITY.md` | artifacts, hashes, scripts |
| `results/` | curated JSON results and tensor maps |
| `eval/` | frozen corpus, prompts, fixed-target slices |
| `scripts/` | tensor-map, loop-trace, long-context and tiny-loop tools |
| `vendor/` | config, quant config, safetensors header, chat template (no weights) |

## Reproduce

The scripts are self-contained; see `docs/REPRODUCIBILITY.md` and
`docs/SMALL-MODEL-REPRO.md`. Model weights are not included. The long-context runs use
the released GGUFs; the small-model run trains from scratch on WikiText-103.

## License

Original code and analysis: Apache-2.0. Wikipedia-derived evaluation text and slices:
CC BY-SA 3.0. Vendor metadata: Apache-2.0, no weights. See `LICENSES.md`.

## Citation

```bibtex
@misc{inkling2_loop_forensics,
  title  = {Inkling 2 Loop Forensics},
  author = {sky-is-green},
  year   = {2026},
  url    = {https://github.com/sky-is-green/inkling2-loop-forensics}
}
```

This is independent analysis, not affiliated with, endorsed by, or sponsored by Thinking
Machines Lab.
