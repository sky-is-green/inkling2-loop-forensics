# Long-context hypothesis (not yet tested)

The Scion 35B lane originally wanted a long-context fix. The loop is a candidate mechanism,
and it could go either way:

**For it.** The block-attention residual gives every sublayer direct access to block
summaries through learned queries, instead of forcing all long-range state through one
residual stream. The relative-position bias covers 0–1023 with log scaling beyond 8192, and
the model is trained for 1M context. The per-pass state resets, so the core re-reads the
pre-loop state rather than accumulating drift.

**Against it.** The per-pass KV cache triples the core KV, the relative bias is local
(1024), and the loop's benefit at 2k may not transfer to 32k+. The block read could become a
bottleneck or a source of noise at long range.

This is a hypothesis, not a result. Do not claim a long-context gain until the overnight run.

## Planned overnight test

- Contexts: 8k / 16k / 32k (one card, IQ4_XS; Q2_K_XL as a second point if time).
- Loop counts: 2 / 3 / 4, same weights, metadata override.
- Metrics:
  - PPL on a fixed long document, reported by position bucket (0–2k, 2–8k, 8–16k, 16–32k);
  - a needle-in-a-haystack retrieval probe at several depths;
  - a fixed long-document QA set (greedy, `reasoning_effort none`);
  - decode speed and VRAM at each length.
- Ablations: attention sinks at 8k/16k (the 2k test showed 6/6 but +10 % PPL); block-read
  queries and plain residual only at 2k so far.
- Control: the 3-pass IQ4_XS at 2k already measured (PPL 6.5342, 6/6).

Artifacts to add when run: `eval/long-context/` (frozen document + prompts + hashes),
`results/long-context-<date>.json`, and a section in `LOOP-STUDY.md`.

## What would change the 35B plan

- **Loop helps at long context** → the 35B loop is a plausible long-context fix, and the
  per-pass state / block bank become first-class design choices.
- **Loop is neutral or hurts** → keep the 35B long-context work separate; the loop remains a
  depth/capability mechanism, not a context fix.
