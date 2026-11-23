# GGUF layout — unsloth mainline quantization

Third-party GGUF used to inspect the text model's tensor naming and expert packing.
Repo: `unsloth/Inkling-2-Mini-GGUF` @ `45fd08764f2b0bf16b03a5483a80e5c7aaec8772`.
Base: `thinkingmachines/Inkling-2-Mini`, Apache-2.0. Use the `Mainline_Support/` files
(rename `.gguf_file` → `.gguf`); the root files use Unsloth's own layout.

## Files

| file | GiB | note |
|---|---:|---|
| `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` | 21.30 | primary; **rename to `.gguf`** |
| `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` | 16.42 | |
| `Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file` | 15.47 | |
| `Mainline_Support/Inkling-2-Mini-UD-IQ4_XS.gguf_file` | 14.65 | |
| root `Inkling-2-Mini-UD-*.gguf` | same | **Unsloth layout**, not mainline |
| `mmproj-F16.gguf` / `mmproj-F32.gguf` | 0.60 / 1.19 | vision projector, skip |
| `dspark-Inkling-2-Mini-Q8_0.gguf` | 0.37 | draft model for spec-decode, later |
| `dspark/dspark-Inkling-2-Mini-BF16.gguf` | 0.60 | |

Unsloth's KLD table (vs their UD-Q8_K_XL): IQ4_XS 0.0109, Q4_K_M 0.0108, Q4_K_XL 0.0107,
Q8_K_XL reference. Mainline PPL 7.653 for Q8_K_XL with the #32704 fix (7.651 root layout).

## GGUF metadata (parsed from the Q4_K_M header)

```
general.architecture            inkling2_loop
general.name                    Inkling-2-Mini
general.file_type               15 (mostly Q4_K_M)
inkling2_loop.block_count       24
inkling2_loop.context_length    1048576
inkling2_loop.embedding_length  2304
inkling2_loop.feed_forward_length 1152
inkling2_loop.expert_count      128
inkling2_loop.expert_used_count 8
inkling2_loop.expert_feed_forward_length 1152
inkling2_loop.attention.head_count    24 (per layer)
inkling2_loop.attention.head_count_kv 8  (per layer)
inkling2_loop.attention.key_length / value_length 128
inkling2_loop.attention.key_length_swa / value_length_swa 128
inkling2_loop.attention.sliding_window 1024
inkling2_loop.attention.sliding_window_pattern [24 bools]
inkling2_loop.rope.dimension_count 0          <-- no RoPE
inkling2_loop.shortconv.l_cache     4
inkling2_loop.attn_res.block_size   6
inkling2_loop.loop.count            3
inkling2_loop.loop.start            6
inkling2_loop.loop.end              18
inkling2_loop.loop.gate_rank        64
inkling2_loop.loop.varying          false
inkling2_loop.expert_weights_scale  8.0
inkling2_loop.activation.situ_beta         4.0
inkling2_loop.activation.situ_linear_beta  25.0
inkling2_loop.final_logit_softcapping      30.0
inkling2_loop.attention.relative_length    16
inkling2_loop.attention.relative_extent    1024
inkling2_loop.attention.qk_clip            22.627416610717773
inkling2_loop.attention.relative_clip      8.0
inkling2_loop.attention.log_scale_floor    8192
inkling2_loop.attention.log_scale_alpha    0.1
inkling2_loop.vocab_size         201024
inkling2_loop.vocab_size_unpadded 200058
tokenizer.ggml.model             gpt2
tokenizer.ggml.pre               inkling2
tokenizer.ggml.tokens            array[200058]
tokenizer.ggml.merges            array[424413]
tokenizer.ggml.eos_token_id      200006
tokenizer.chat_template          present (same as vendor)
```

744 tensors, 59 KV pairs.

## Tensor types (Q4_K_M)

| type | count | where |
|---|---:|---|
| Q8_0 | 150 | attention linears, token_embd, output |
| F32 | 518 | norms, convs, scales, res queries, gates, router bias |
| BF16 | 32 | (layers 20–23 attention? derived tensors) |
| NVFP4 (40) | 40 | **expert weights layers 0–19** (w13 + w2) |
| Q5_K | 4 | expert gate_up layers 20–23 |

Expert tensors:
```
blk.N.ffn_gate_up_exps.weight   (2304, 2304, 128)   NVFP4 for N<=19, Q5_K for N>=20
blk.N.ffn_gate_up_exps.scale    (128,)             F32, per-expert, N<=19
blk.N.ffn_down_exps.weight      (1152, 2304, 128)   NVFP4 for N<=19, Q8_0 for N>=20
blk.N.ffn_down_exps.scale       (128,)             F32, per-expert, N<=19
```
This is the same NVFP4 expert path as the vendor release. PR #32704 exists because the loop
slots are copies of the physical layers and were made **before** the scale pass, so passes
2/3 lost the per-expert scales (PPL 130 instead of 7.65). Merged 2026-11-21.

## Name map (GGUF ↔ HF)

| GGUF | HF / vendor |
|---|---|
| `token_embd.weight` | `embed_tokens.weight` (padded 201024) |
| `token_embd_norm.weight` | `embed_tokens.embed_norm.weight` |
| `output.weight` | `lm_head.weight` (padded to 201024) |
| `output_norm.weight` | `norm.weight` |
| `output_res_query.weight` | `final_input.query` |
| `blk.N.attn_res_query.weight` | `layers.{N+1}.attention_input.query` (23 of them) |
| `blk.N.ffn_res_query.weight` | `layers.N.mlp_input.query` |
| `blk.N.attn_norm.weight` | `input_layernorm.weight` (global layers only; 20 total) |
| `blk.N.ffn_norm.weight` | `post_attention_layernorm.weight` |
| `blk.N.attn_q.weight` | `q_proj.linear` |
| `blk.N.attn_k.weight` / `attn_v.weight` | `k_proj.linear` / `v_proj.linear` |
| `blk.N.attn_output.weight` | `o_proj.linear` |
| `blk.N.attn_gate.weight` | `o_proj.gate_proj` |
| `blk.N.attn_sink_proj.weight` | `sink_proj` |
| `blk.N.attn_q_clip.weight` | `q_proj.norm.scale` (flattened) |
| `blk.N.attn_rel.weight` | `relative_position_bias.linear` |
| `blk.N.attn_rel_bias.weight` | `relative_position_bias.proj` |
| `blk.N.attn_rel_clip.weight` | `relative_position_bias.norm.scale` |
| `blk.N.attn_k_conv.weight` / `attn_v_conv.weight` | K/V short conv |
| `blk.N.attn_post_conv.weight` / `ffn_post_conv.weight` | post-attn / post-MLP conv |
| `blk.N.ffn_gate_inp.weight` / `exp_probs_b.bias` | router weight / e_score_correction_bias |
| `blk.N.loop_{q,k,v,mlp}_ctrl_{down,gain,forget}.weight` | message gates (12 core layers) |

Vision is in `mmproj-*`, not in the text GGUF.

## Requirements / caveats

1. Loaders must share the NVFP4 per-expert scales across the loop slots; otherwise passes
   2/3 run without scales (PPL ~130 instead of ~7.65). This is a model/format property.
2. Sampling per model card: temp 1.0, top-p 1.0, top-k 0, min-p 0.
3. `output.weight` and `token_embd.weight` are padded to 201024 rows; HF lm_head is 200058.
   This matches the vendor padding.

