# Open questions

The analysis questions this repository exists to answer. Each should end with a number, a
negative result, or a design decision recorded in these docs.

## A. The loop itself

1. **What does loop depth buy?** Run the same weights at 1, 2 and 3 passes on fixed text,
   code and tool prompts. Is pass 3 better than pass 2 at matched compute, or is it just
   more FLOPs? Measure capability, not only PPL.
2. **Is the reset essential?** The passes restart from the pre-loop state and bank. What
   happens if pass 2 starts from pass 1's output instead? Is the gain from depth or from
   the message channel?
3. **Which components are essential?**
   - message gates (`q/k/v/mlp`, rank 64) — zero them, freeze them, or remove them;
   - block attention residual — replace the bank read with a plain residual;
   - per-pass KV cache — share KV across passes;
   - relative-position bias vs a RoPE baseline;
   - attention sinks; q/k L2 clips; log scaling; static vs dynamic short conv.
   Ablate in the reference runtime.
4. **How much of the gain is trained behavior?** **At small scale, essentially all of it.**
   With the reference implementation at 23.4M parameters, a 3-pass-trained model is best
   at 3, while a 1-pass control is flat across 1/2/3/4 inference passes. The 3-pass model
   is also much worse when evaluated at 1 pass. See `SMALL-MODEL-REPRO.md`. This is the
   central risk for the 35B port and the reason the 35B needs a training run.

## B. Quantization across passes

5. **Does quant error compound?** Compare hidden-state drift and logit KL at the end of
   pass 1, 2 and 3 for IQ4_XS vs Q8_K_XL, on fixed prompts. If pass 3 amplifies the error,
   the loop needs higher precision in the core or loop-aware quantization.
6. **Which tensors are pass-sensitive?** Per-zone (pre/core/post) and per-expert probes;
   does the message-gate path accumulate error faster than attention?

## C. The DSpark drafter

7. **What is the exact vendor drafter?** Recover the math from `mtp.safetensors` +
   `mtp_config` (3 layers, block 8, markov rank 256, target layers 5/23/41/48) and the
   Unsloth GGUF tensor list:
   - `markov_w1` [256, 201024] — vocab → rank?
   - `dspark_hidden_out` [256, 2304] + bias — hidden → rank?
   - `dspark_fuse` [4,4], `dspark_norm` [2304], `mask_embd` [2304], `dflare_norm` [2304]
   - `joint_proj` [5120, 768] — [hidden ; 2×rank?] → 3×rank
   - block tensors: `attn_k_ctx`, `attn_v_ctx`, `attn_q_norm`, `attn_k_ctx_norm`.
8. **What is the anchor-first block layout?** How are the 8 draft slots ordered, which one
   is the anchor, and how does `dspark_fuse` mix the 4×4 features?
9. **Can it be reproduced?** Build the drafter from the vendor weights, compare drafts and
   acceptance to the reference behavior on the same prompts. Target: match the acceptance
   profile, then improve with prompt/temperature tuning.
10. **What is the acceptance ceiling?** Measure by prompt class (code, factual, chat) and
    temperature; this sets the useful draft length.

## D. Long context

13. **Is the loop a long-context fix?** **Yes at 32k.** At a fixed 2k target, loop-3 PPL
    improves from 7.11 to 6.54 (Q2) and 6.88 to 6.33 (Q8) from 4k to 32k. Loop 3 is 1.6–1.8×
    better than loops 2/4 at every context and the only depth that answers 32k needles
    (3/3, both quants); loops 2/4 give 0/3 final answers even with a 2048-token budget.
    Remaining: 64k/128k, a harder long-document QA set, and why loops 2/4 fail (message
    channel vs block bank vs schedule). Results: `LONG-CONTEXT.md`.

## E. Long-context mechanism

16. **Why do loops 2 and 4 fail at 32k?** They sometimes find the code in reasoning but do
    not emit it. Is the message channel too weak at two passes, or does the fourth pass
    over-write the retrieved state? Probe the per-pass block bank and message vectors on the
    32k needles.
17. **Do the sinks matter at 64k+?** They cost ~6–8 % PPL at 8k/16k/32k but are not needed
    for 32k retrieval. Test 64k/128k if Q2 fits.
18. **Does the result survive a harder long-document QA set?** The needle is synthetic and
    the haystack is wiki prose; add multi-hop and code-repo retrieval.

## F. 35B and numerics

19. **35B design.** Which 12 layers form the core (8–19, 10–29, 14–25), 2 or 3 passes, and
    which components survive ablation? What is the distillation recipe and the eval gate?
20. **Numerics.** F32 block states, softcaps, L2 clips, router precision, sinks: what is the
    minimum precision that preserves the reference logits, and where does the loop need F32?
21. **Does the depth preference survive at 35B scale?** At 23M the 4-pass penalty is only
    +2.2 % vs +52 % at 26B. Is the 3-pass advantage stronger at larger scale, or does it
    depend on the training budget/data? The 35B pilot must measure 2/3/4 rather than
    inherit the 26B gap.
