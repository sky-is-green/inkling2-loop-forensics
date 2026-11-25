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

---

# Component sensitivity

Same weights, loop 3, one component zeroed per copy. PPL on the frozen corpus; six-prompt
probe at `reasoning_effort none`.

| ablation | tensors | PPL | Δ | clean | verdict |
|---|---:|---:|---:|---:|---|
| plain residual (no block read, graph hook) | — | **12 877 527** | ~2×10^6 | 0/6 | **the architecture** |
| block-read queries (`attn_res_query`, `ffn_res_query`, `output_res_query`) | 48 | **731.36** | 112× | 0/6 | **most critical** |
| relative-position bias (`attn_rel`, `attn_rel_bias`) | 48 | **244.73** | 37× | 0/6 | **positional mechanism** |
| loop controllers (message gates) | 144 | **13.29** | +103 % | 2/6 | **loop mechanism** |
| attention sinks (`attn_sink_proj`) | 24 | 7.21 | +10 % | 6/6 | secondary at 2k |
| baseline | — | 6.53 | — | 6/6 | |

Interpretation:

- The **block-attention residual** is the architecture. Replacing its read with a plain
  residual destroys the model (PPL 12.9 M); even a uniform read over the bank (query-zero,
  PPL 731) is far better than no bank. The learned query matters, but the bank itself matters
  more.
- The **relative-position bias** is the only positional mechanism; with no RoPE, removing it
  destroys the model.
- The **cross-pass messages** are what make the three passes different and useful.
- The **attention sinks** cost 10 % PPL at 2k and ~6–8 % at 8k/16k/32k, but do not break the
  six short prompts or 32k retrieval (3/3 without them); they are secondary.

Artifacts: `results/component-ablations.json`, `results/{relpos,sinks,resq}-ablation-gen.json`,
`scripts/zero_tensors.py`.

---

# Quantization: attention/embed precision is not the bottleneck

| model | attention / embed | experts 0–19 | experts 20–23 | PPL |
|---|---|---|---:|---:|
| UD-IQ4_XS | Q8/Q6 | NVFP4 | IQ4_XS / Q5 | **6.5342 ± 0.275** |
| UD-Q8_K_XL | BF16 | NVFP4 | BF16 | **6.5263 ± 0.274** |

The two differ in attention/embed precision and in the last four expert layers, but both
keep the dominant layers 0–19 experts at **NVFP4** (verified from the GGUF type tables).
PPL is unchanged within noise (0.12 %). So the quantization risk, if any, lives in the
NVFP4 experts, not in attention or the small tail of expert layers.

Q8_K_XL does not fit one 20.5 GB card (22.9 GB); the run used `-ncmoe 6` to put six layers'
MoE on CPU. That is valid for PPL, not for speed.

A true expert-precision control needs a self-converted GGUF with Q4_K/Q5_K experts (or the
vendor MXFP8 release), not another unsloth quant. That is the next quantization step.

---

# Severe compression: Q2_K_XL

Does the loop survive 2-bit experts? `UD-Q2_K_XL` is 8.51 GiB; its expert tensors are a mix
of IQ1_S, IQ2_S/XS/XXS, IQ3_XXS, a 2-bit Q2 type and 11 remaining NVFP4 tensors. The IQ4_XS
NVFP4 experts are largely requantized.

| passes | Q2_K_XL | IQ4_XS | gap |
|---:|---:|---:|---:|
| 2 | 10.9952 ± 0.490 | 10.3317 | +6.4 % |
| **3** | **6.7262 ± 0.282** | **6.5342** | **+2.9 %** |
| 4 | 10.6647 ± 0.451 | 9.9125 | +7.6 % |

Generation probe: **6/6 clean at loop 3**, 4/6 at loops 2 and 4. The loop ordering and the
6-prompt quality are preserved.

Ablations at Q2 (loop 3):

| ablation | PPL | Δ |
|---|---:|---:|
| loop controllers zeroed | 14.6268 | +117 % |
| block-read queries zeroed | 770.92 | 115× |

The component hierarchy is unchanged: the block read and the message gates are still the
architecture.

**Interpretation.** Severe compression costs only 2.9 % PPL at the trained 3-pass point and
does not break the mechanism. The quant gap is smallest at the trained schedule and widens
off-schedule (+6.4 % at 2, +7.6 % at 4): off-schedule passes amplify quantization error, but
the shipped 3-pass loop does not compound it. This is a stronger result than expected — the
loop is not fragile under 2-bit experts.

**Caveats.** One corpus, six short prompts; hard capability is still untested. Long context is
tested in the next section. The Q2 type mix is dynamic, so “2-bit” is an average, not a
uniform format.

Artifacts: `results/low-quant-q2.json`, `results/q2-loop-gen.json`.


---

# Long context: the loop uses the context

The 32k study is in [`LONG-CONTEXT.md`](LONG-CONTEXT.md); this is the headline result.

At a **fixed 2k target** (same text, different amounts of preceding context), loop-3 PPL
falls as context grows:

| nominal C | actual n_ctx | Q2 loop 3 | Q8 loop 3 | Q2 loop 2 | Q8 loop 2 | Q2 loop 4 | Q8 loop 4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 5,120 | **7.107** | **6.879** | 12.345 | 11.446 | 11.560 | 11.225 |
| 8k | 9,216 | **7.049** | **6.765** | 12.281 | 11.393 | 11.532 | 11.075 |
| 16k | 17,408 | **6.575** | **6.388** | 11.416 | 10.745 | 10.508 | 10.435 |
| 32k | 33,792 | **6.544** | **6.331** | 11.607 | 10.754 | 10.235 | 10.186 |

- Loop 3 improves by ~8 % from 4k to 32k in both quants; loops 2/4 improve only slightly and
  stay 1.6–1.8× worse at every context.
- Needle retrieval at 32k: **loop 3 is 3/3 for both quants at all depths**; loops 2/4 give
  0/3 final answers, even with a 2048-token budget (they sometimes find the code in
  reasoning but never emit it).
- Q2 is only ~3 % worse than Q8 at 32k (6.54 vs 6.33); the loop is quant-robust.
- Attention sinks cost ~6–8 % PPL at 8k/16k/32k but are not needed for 32k retrieval
  (3/3 without them).

This makes the loop a plausible long-context mechanism for the 35B, and it makes the 3-pass
schedule and the block bank non-optional. Method, controls and caveats:
[`LONG-CONTEXT.md`](LONG-CONTEXT.md); numbers: `results/long-context.json`.
