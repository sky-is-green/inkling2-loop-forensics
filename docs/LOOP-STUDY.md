# Loop depth study — first result

**Question:** does the number of loop passes buy quality, or is it free depth?

**Method:** same weights, same quantization (IQ4_XS), same runtime. The pass count is changed
through the GGUF metadata override `inkling2_loop.loop.count`; 3 is the trained value.
Perplexity on a frozen 33 KB Wikipedia extract (3 × 2048 tokens), plus a six-prompt greedy
generation probe at `--reasoning-effort none`, 384 tokens.

## Perplexity

| passes | layer applications | PPL | Δ |
|---:|---:|---:|---:|
| 1 | 24 | — | loader rejects `n_passes < 2` for a fixed loop |
| 2 | 36 (−25 %) | **10.3317 ± 0.460** | +58 % vs 3 |
| **3 (trained)** | **48** | **6.5342 ± 0.275** | baseline |
| 4 | 60 (+25 %) | **9.9125 ± 0.414** | +52 % vs 3 |

## Generation probe

Six prompts, greedy, `Thinking effort level: 0`; expected answers frozen in
`eval/prompts.json`. “clean” = final answer present and correct.

| prompt | loop 2 | loop 3 (trained) | loop 4 |
|---|---|---|---|
| capital of France | Paris | Paris | Paris |
| 84·3/2 | 126, but reasoning leaked into content | 126 | 126 |
| bat and ball | **$0.10** (wrong, repeated) | **$0.05** | no final answer within 384 |
| python sum | correct | correct | correct |
| roses / flowers | **degenerate repetition** | **“No. …”** (correct) | no final answer within 384 |
| three primary colors | **degenerate repetition** | **“red, yellow, blue”** | no final answer within 384 |

Clean/correct: **loop 2 ≈ 3/6, loop 3 6/6, loop 4 3/6**.

## Interpretation

The loop is **trained depth, not free depth**:

- Removing one pass is not a 25 % compute saving; it collapses multi-step reasoning and
  produces repetition and reasoning leakage into the answer.
- Adding a pass is not a quality win either: PPL rises to 9.91 and three prompts never reach
  a final answer within 384 tokens.
- The trained 3-pass schedule is a fixed point. The message gates and block-attention
  queries are tuned to exactly this number of applications.

This is the central risk for the 35B port: a loop bolted onto a model that was not trained
with it should not be expected to work. The 35B needs unrolled training/distillation, not
just a shared core.

## What this does not answer

- Whether **pass 3 specifically** is essential, or whether the model is merely sensitive to
  any change in schedule (a 2-pass fine-tune could recover the quality at lower compute).
- Quantization compounding: only IQ4_XS was measured; a Q8/higher-precision control is
  needed.
- Which components carry the gain (message gates, block residual, per-pass KV). Those
  ablations need either runtime flags or weight edits.

## Artifacts

- `results/loop-ppl.json` — numbers, hashes, settings, caveats.
- `results/loop-gen.json` — per-prompt content and reasoning for loops 2/3/4.
- `eval/wiki-blue-64k.txt`, `eval/prompts.json` — frozen inputs.
- `scripts/run_loop_prompts.py` — the generation runner.

Commands:

```bash
llama-perplexity -m IQ4_XS.gguf -f eval/wiki-blue-64k.txt -c 2048 -b 2048 -ngl 99 \
  --chunks 4 --override-kv inkling2_loop.loop.count=int:N
```

---

# Cross-pass message gates are load-bearing

**Question:** is the gain from the three repeated layers, or from the learned cross-pass
messages?

**Method:** reflink copy of the same IQ4_XS GGUF; zero all 144
`loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` weights in layers 6–17 (113.2 MB). The message
gate becomes `(1-0)·cur + 0·message = cur`: the passes still run, the block-attention
residual and per-pass KV are untouched, but no cross-pass information is mixed in.

| metric | baseline | no controllers | Δ |
|---|---:|---:|---:|
| PPL (loop 3) | **6.5342** | **13.2932** | +103 % |
| clean prompts | **6/6** | **2/6** | |

Ablated generation: capital and code correct; arithmetic, bat and ball, and the logic question
never reach a final answer within 384 tokens; the color list repeats.

**Interpretation.** The loop is not “three copies of the same block”. The learned message
channel is what makes the later passes different and useful. Removing it roughly doubles
perplexity and breaks four of six prompts. Any 35B loop must train the cross-pass gates; a
shared core alone is not the mechanism.

**Caveat.** This proves necessity, not that rank 64 (or this exact gate form) is optimal. It
also leaves the passes running, so it isolates the message channel rather than the loop
control flow.

Artifacts: `results/controller-ablation.json`, `results/controller-ablation-gen.json`,
`scripts/zero_loop_controllers.py`.
