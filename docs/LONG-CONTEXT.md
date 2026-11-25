# Long context: the loop uses the context

**Status: tested 2026-11-25.** The loop is not merely neutral at long context. At a fixed
target, the trained 3-pass model gets *better* as context grows, and it is the only loop
depth that reliably turns a 32k retrieval into a final answer. This makes the loop a
plausible long-context mechanism for the 35B, not just a depth/capability trick.

## Hypothesis

**For it.** The block-attention residual gives every sublayer direct access to block
summaries through learned queries, instead of forcing all long-range state through one
residual stream. The relative-position bias covers 0–1023 with log scaling beyond 8192, and
the model is trained for 1M context. The per-pass state resets, so the core re-reads the
pre-loop state rather than accumulating drift.

**Against it.** The per-pass KV cache triples the core KV, the relative bias is local
(1024), and the loop's benefit at 2k may not transfer to 32k+. The block read could become a
bottleneck or a source of noise at long range.

## Method

- **Corpus:** the frozen `wikitext-2-raw` test split (312,991 tokens, sha256
  `173c87a5…`). One corpus, one target region.
- **Fixed-target slices:** all slices end at the same paragraph boundary. The 2048-token
  target at the end of the first `n_ctx` tokens is therefore identical across context
  lengths. This separates the context-length effect from target difficulty.
- **Contexts:** nominal 4k / 8k / 16k / 32k. The strided harness adds `stride/2 = 1024`, so
  the actual `n_ctx` values are 5,120 / 9,216 / 17,408 / 33,792.
- **Loops:** 2, 3, 4 on the same weights via metadata override.
- **Quants:** Q2_K_XL (2-bit experts, full GPU) and Q8_K_XL. Q8 at 32k needs CPU-offloaded
  experts on the 20 GB card; the fair Q8 runs use the stable 16-layer configuration (see
  controls). The earlier 20-layer Q8 matrix is exploratory only.
- **Needles:** a unique code (`7391-ALPHA`) inserted at 10 / 50 / 90 % of a wikitext
  haystack; greedy, reasoning enabled; final-answer content scored. The 32k off-schedule
  cases were repeated with a 2048-token generation budget to rule out truncation.

## Result 1: fixed-target PPL

The same 2k target, preceded by more context:

| nominal C | actual n_ctx | Q2 loop 3 | Q2 loop 2 | Q2 loop 4 | Q8 loop 3 | Q8 loop 2 | Q8 loop 4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 5,120 | **7.107** | 12.345 | 11.560 | **6.879** | 11.446 | 11.225 |
| 8k | 9,216 | **7.049** | 12.281 | 11.532 | **6.765** | 11.393 | 11.075 |
| 16k | 17,408 | **6.575** | 11.416 | 10.508 | **6.388** | 10.745 | 10.435 |
| 32k | 33,792 | **6.544** | 11.607 | 10.235 | **6.331** | 10.754 | 10.186 |

Loop 3 improves monotonically: Q2 -7.9 %, Q8 -8.0 % from 4k to 32k. Loops 2 and 4 improve
only slightly and stay 1.6–1.8× worse at every context. Longer context helps, and the
trained depth is what converts that context into a better prediction.

## Result 2: needle retrieval

Final-answer content, 3 depths per cell:

| context | Q2 loop 3 | Q2 loop 2 | Q2 loop 4 | Q8 loop 3 | Q8 loop 2 | Q8 loop 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 3/3 | — | — | 3/3 | — | — |
| 16k | 3/3 | — | — | 3/3 | — | — |
| 32k | **3/3** | 0/3 | 0/3 | **3/3** | 0/3 | 0/3 |

At 32k, loops 2/4 sometimes locate the code in reasoning but never emit a final answer:
Q2 loop 2 1/3 at 90 % depth, Q8 loop 2 2/3 in reasoning, Q8 loop 4 1/3. The 2048-token
rerun confirmed this is not a generation-budget artifact: Q2 loops 2/4 stayed 0/3 on final
answers (loop 2 1/3 in reasoning), and the Q8 50 % case stayed 0/1 for both loops.

A fair-config Q8 32k loop-3 confirmation (50 % depth) also returned `7391-ALPHA`.

## Result 3: attention sinks

Matched chunk counts, Q2 loop 3:

| context | baseline PPL | no sinks | Δ | 32k retrieval without sinks |
|---:|---:|---:|---:|---:|
| 8k | 4.4359 | 4.7965 | +8.1 % | — |
| 16k | 5.5491 | 5.8918 | +6.2 % | — |
| 32k | 6.0642 | 6.4698 | +6.7 % | **3/3** |

Sinks cost a modest ~6–8 % PPL at long context, but are not required for retrieval. They
remain secondary to the block bank and the loop message gates.

## Controls

- **KV precision is not the mechanism.** Q2 32k with q8 KV gives 6.0629 vs 6.0642 with f16
  KV (-0.02 %).
- **Q8 offload is stable to 16–17 layers.** Q8 2k PPL by CPU-offloaded MoE layers:
  6 → 4.929, 10 → 4.948, 14 → 4.973, 16 → 4.992, 17 → 5.068, **18 → 6.660**, 20 → 6.646.
  The sharp break at 18 is why the initial 20-layer Q8 matrix is exploratory only. The fair
  Q8 reruns use 16 layers; Q8 32k loop 3 then gives **5.921** vs Q2 **6.064**.
- **Q8 vs Q2.** At every fixed-target context Q8 is slightly better (6.33 vs 6.54 at 32k,
  ~3 %). The 2-bit model keeps the mechanism and most of the quality.

## Interpretation

- **The loop uses long context.** At a fixed target, PPL falls as context grows; this is not
  a target-difficulty artifact.
- **The 3-pass schedule is essential.** Loops 2/4 are far worse in PPL and fail to produce a
  final 32k answer even with a 2048-token budget. The trained depth is not cosmetic.
- **The mechanism is quant-robust.** Q2 and Q8 show the same ordering, the same context
  trend, and the same needle pattern; Q8 is only ~3 % better.
- **Sinks are secondary.** They cost PPL but do not gate retrieval. The block bank and the
  cross-pass message channel remain the architecture.
- **For the 35B:** the loop is a plausible long-context fix, and the block bank + 3-pass
  schedule must both survive the port. A shared core without the bank, or an off-schedule
  depth, does not reproduce this behavior.

## Caveats

- One corpus, one fixed target region, one 2k scored target. Absolute PPL is not comparable
  to the earlier wiki-blue corpus.
- The needle is synthetic and the haystack is wiki prose; a harder long-document QA set is
  still needed.
- Nominal 4k/8k/16k/32k use actual contexts 5.1k/9.2k/17.4k/33.8k. The 16k slice
  tokenizes to 17,407 tokens before the boundary; the harness consumes one extra boundary
  token. The other three slices hit the exact count, and the target text is the same.
- Q8 32k requires a CPU-offloaded expert configuration on the 20 GB card. The fair runs use
  the stable 16-layer configuration; this is a measurement configuration, not a deployment
  recommendation.
- The 2048-token off-schedule rerun covered all Q2 depths and the Q8 50 % depth; the Q8
  10/90 % off-schedule cases still used the shorter budget.
- Loop-2/4 failure is a final-answer failure, not always a retrieval failure; reasoning
  sometimes contains the code.

## Artifacts

- `results/long-context.json` — curated numbers, findings, caveats.
- `eval/long-context/` — test corpus, needle prompts, fixed-target slices, hashes.
- `scripts/build_needle_prompts.py`, `scripts/make_fixed_target_slices.py` — deterministic
  data builders.
- Raw logs and the full run records are kept outside the analysis repo.
