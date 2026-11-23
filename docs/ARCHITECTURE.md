# Inkling 2 Mini — Architecture & Tensor Map (P0-01)

Status: derived from the vendor repo artifacts (config, safetensors header, quant config,
model card) + `transformers` 5.24.0 source. **No weights downloaded, no model instantiated on
real devices.** Date: 2026-11-23.

Sources (all copied into `vendor/`):
- `thinkingmachines/Inkling-2-Mini` @ `b85c3cfa615fc413cc9af3379f491b963f3b586a`
- `inkling-config.json`, `inkling-generation_config.json`, `inkling-hf_quant_config.json`,
  `inkling-model.safetensors.index.json`, `inkling-header.bin` (first 2 MB = full 148 408 B
  safetensors JSON header), `inkling-card.md`, `chat_template.jinja`
- `transformers==5.24.0` wheel (`vendor/transformers-5.24.0-py3-none-any.whl`,
  sha256 `ff5ebedf9359f94bb63772b6380eba6e7104d010cbbb2cb82c788817ea7486f6`), module
  `transformers/models/inkling2_loop/`.

---

## 1. Identity and size

| item | value |
|---|---|
| architectures | `Inkling2LoopForConditionalGeneration` (text: `Inkling2LoopTextModel`) |
| model_type | `inkling2_loop` / `inkling2_loop_text` |
| total params (API) | 26 388 646 440 (26.389 B) |
| total params (HF meta build) | 26.386 B + 2.2 M padded lm_head rows dropped on load |
| text params | 26.07 B |
| vision params | 0.318 B (23 ViT layers, hidden 1024) |
| experts | 24.4612 B = 92.7 % of text; 128 experts/top-8 |
| attention + convs + controllers | 0.6826 B |
| embed + unembed | 0.9264 B |
| stored bytes on disk | 19 526 870 752 (19.53 GB) |
| storage layout | single `model-00001-of-00001.safetensors`, 1244 tensors; separate `mtp.safetensors` |
| license | Apache-2.0 |

Precision groups (`hf_quant_config.json`, `quant_method: inkling2`):
- 40 tensors NVFP4 = layers 0–19 × {w13, w2}
- 8 tensors MXFP8 = layers 20–23 × {w13, w2}
- 993 tensors BF16, 74 F32, 1 BF16 (mtp group: 54 BF16)
- NVFP4: E2M1, block 16, FP8_E4M3 scales, per-expert amax/outer_scale_denominator
- MXFP8: E4M3, block 32, UE8M0 scales

## 2. Layer plan

24 physical layers. `layer_types` is derived: `"hybrid"` (global) iff `(i+1) % 6 == 0`,
otherwise `"hybrid_sliding"`.

| range | layers | behaviour |
|---|---|---|
| pre-loop | 0–5 | executed once |
| **core** | **6–17** | executed **3×** (loop_config core_start/end, num_passes=3) |
| post-loop | 18–23 | executed once |
| global attention | 5, 11, 17, 23 | full attention, rel_extent 1024, log-scaled |
| sliding attention | the other 20 | window 1024 |

Layer applications per token = `24 + (3-1)*12 = 48` (`total_num_hidden_layers`).
The core is 12 × (1.019 B experts + ~29.6 M non-expert) ≈ 12.58 B of weights reused 3×.

## 3. Execution order (text model, `Inkling2LoopTextModel.forward`)

```
x = RMSNorm(embed[input_ids])                    # NormedEmbedding, learned weight
block_states = zeros(B, T, 0, H)
for layer in 0..5:      x = layer(x, block_states)
pre_x, pre_blocks = x, block_states              # snapshot
for layer in 6..17:     x = layer(x, block_states)          # pass 1
for pass in 2..3:
    prev_x, prev_blocks = x, block_states
    x, block_states = pre_x, pre_blocks          # RESET to pre-loop state
    for j, layer in enumerate(6..17):
        messages = message_input(prev_blocks, prev_x, j)
        x = layer(x, block_states, messages, cache_offset=...)
for layer in 18..23:    x = layer(x, block_states)
x = final_input(block_states, x)                 # one more block-attention read
x = RMSNorm(x)
logits = softcap_30(lm_head(x[:, -1]))           # lm_head = first 200058 embed rows
```

Critical loop facts:
- Each pass **restarts from the pre-loop hidden state**, not from the previous pass's output.
  Cross-pass information flows only through `messages`.
- `block_states` is also reset each pass, so pass 2/3 accumulate only the two core block
  boundaries (layers 6 and 12) on top of the pre-loop bank.
- The message for loop layer `j` is a 4-vector (q, k, v, mlp) read from the previous pass's
  `[prev_blocks_last ; prev_x]` with **learned per-layer queries** (`message_input.query`,
  shape (12, 4, 2304), F32). It is *not* a KV-cache read. Note the slice
  `block_states[..., num_pre_loop_blocks + 1 :, :]`: messages see only the **last core block**
  (the state entering layer 12), not the whole bank — 2 source vectors per position.
  Verified on a toy model (`scripts/trace_loop.py`).
- **KV cache is per pass**: `DynamicCache` layers are deep-copied for each extra pass
  (`num_hidden_layers + (passes-1)*12` = 48 cache entries). Passes do not share attention KV.
- Conv state caches are also per pass (`cache_offset`).

## 4. Residual stream: block attention residuals

`hidden` is **not** a single running residual stream. Every layer where `i % 6 == 0`
(0, 6, 12, 18) is a **block boundary**:

```
block_residual = x_in
x = attn(...)                       # no residual add on boundary layers
block_states = cat(block_states, block_residual)   # bank grows
x = x + mlp(...)
```
For non-boundary layers: `x = x_in + attn(...)` then `x = x + mlp(...)`.

Every attention/MLP input is read from the bank by a learned query
(`Inkling2LoopBlockAttentionResidual`), not by a plain residual:

```
src = RMSNorm_nolearn(cat([block_states, partial.unsqueeze(-2)], -2))  # float32
q   = learned_query.expand(B, T, n_q, H)
out = softmax(q @ src^T / sqrt(H)) @ src      # causal=FALSE, all positions visible
```
- `attention_input`: 3 queries on global layers (separate q/k/v), 1 query on sliding
  layers (same vector reused for q, k and v). Absent on layer 0.
- `mlp_input`: 1 query every layer.
- `final_input`: 1 query before the final norm.
- Queries are F32 parameters. This is a per-token, per-layer attention over the block bank
  (at most **4** blocks, because boundaries are 0/6/12/18) plus the current partial state:
  reads see ≤5 vectors, one query (3 on global attention inputs).
  Verified on a toy model (`scripts/trace_loop.py`): bank sizes 1→2→3→4, one 3-query read.

## 5. Attention (no RoPE)

There is **no rotary embedding anywhere** in the text model. Positional signal comes from:
1. hidden-conditioned **relative position bias** (learned bank, backward distances 0..1023),
2. **log scaling** of queries and relative states for positions ≥ 8192,
3. causal short convolutions (kernel 4),
4. the 1024 sliding window.

Common to both layer types:
- `wq_du`: H→24×128, `wk_dv`/`wv_dv`: H→8×128, `wo_ud`: 24×128→H (GQA 3:1)
- **gated output**: `o = wo( attn_out * sigmoid(gate_attn(x)) )`
- **per-token attention sink**: `sink_proj(x)` → one extra logit per head added to the
  softmax with a zero value column (absorbs probability mass). This disables
  flash-attention/SDPA in HF (`_supports_flash_attn = False`, `_supports_sdpa = False`).
- **Q/K L2 clip**: per-head vector norm clipped to `qk_l2_clip_threshold` = 22.627
  (= 16·√2), after a learned per-head scale (`q_l2_clip.scale`, (24,128)).
- **V norm**: unweighted RMSNorm (no gamma), eps 1e-6.
- **relative bias**: `wr_du`: H→24×16 → per-head L2 clip (threshold 8) → dot with learned
  `proj` (16,1024) → gather by `q_pos - kv_pos`, zero outside [0,1024). For global layers the
  relative states are additionally log-scaled.
- softmax in float32, scaling = head_dim^-0.5.

Sliding layers:
- `k_sconv`/`v_sconv`: depthwise causal conv, kernel 4, 1024 channels, **static** weights;
  K then L2-clipped, V then RMS-normed.
- q: no log scaling.

Global layers:
- K/V use **dynamic** short conv: `static_w` (4,1024) + per-token weights
  `weight_proj(hidden)` → 8×4 taps, shared within each KV head; then
  `(conv + residual)/sqrt(2*4+1)`. K L2-clipped, V RMS-normed.
- q and relative states use `log_scale(t) = 1 + 0.1 * log(max((t+1)/8192, 1))`.
- separate `q_input_norm` / `k_input_norm` / `v_input_norm` RMSNorms (sliding layers reuse
  one `input_layernorm` for q/k/v).

Post-sublayer: `attn_sconv` and `mlp_sconv` are *additional* depthwise causal convs over the
sublayer output (F32), applied before the residual add. So each layer has 4 conv states.

## 6. MoE

- Router (F32): `weight` (128, 2304) + `e_score_correction_bias` (128, F32). `sigmoid(logits)`;
  top-8 on `scores + bias` (unsorted); weights = selected **scores** / sum × `route_scale` 8.0.
  No shared experts.
- Experts packed as 3-D tensors:
  - `gate_up_proj` (128, 2304, 2304) = [gate(1152) ; up(1152)] × hidden
  - `down_proj` (128, 2304, 1152) = hidden × intermediate
- Activation: `softcap_4(gate) * sigmoid(gate) * softcap_25(up)` with
  `softcap_c(x) = tanh(x/c) * c`; sigmoid sees the raw gate. Accumulate in F32.

## 7. Loop controllers (message gates)

Per core layer, 4 gates (q, k, v, mlp), each rank 64:

```
g = [RMSNorm_nolearn(h) ; RMSNorm_nolearn(message)]     # 2H
z = down_proj(g)                                        # 64
forget = tanh(forget_gate_proj(z))                      # H
gain   = tanh(message_gate_proj(z))                     # H
out    = (1 - forget) * h + gain * message
```
`down_proj` (64, 4608), `forget_gate_proj` (2304, 64), `message_gate_proj` (2304, 64), all
BF16 weights, math in F32. Sliding layers apply q/k/v gates before the single input norm;
global layers apply them after the block-attention read and before q/k/v norms; the mlp gate
applies after `mlp_input`.

## 8. Output head / tokenizer

- `embed.weight` (201024, 2304), `unembed.weight` (201024, 2304); lm_head is the **first
  200058 rows** (`unpadded_vocab_size`), remaining 966 rows zero-padded on save. Not tied.
- Final logits soft-capped at ±30.
- eos 200006 (`<|content_model_end_sampling|>`), pad 199999, image token 200054.
- Tokenizer: `tiktoken/tokenizer.model` + `tokenizer.json`, chat template with
  `reasoning_effort` header, `<|message_*|>`, `<|content_*|>`, `<|end_message|>`.

## 9. Checkpoint → HF tensor map

### Global
| checkpoint | HF (5.24.0) |
|---|---|
| `model.llm.embed.weight` | `model.language_model.embed_tokens.weight` |
| `model.llm.embed_norm.weight` | `model.language_model.embed_tokens.embed_norm.weight` |
| `model.llm.norm.weight` | `model.language_model.norm.weight` |
| `model.llm.unembed.weight` | `lm_head.weight` (truncate to 200058 rows) |
| `model.llm.attn_res.output_query` | `model.language_model.final_input.query` |
| `model.llm.attn_res.attn_queries.{i}` | `model.language_model.layers.{i+1}.attention_input.query` (**offset +1**) |
| `model.llm.attn_res.mlp_queries.{i}` | `model.language_model.layers.{i}.mlp_input.query` |
| `model.llm.middle_cycle_cross_pass_queries` (24,2304) | rows 2j,2j+1 = q,mlp of core layer j |
| `model.llm.middle_cycle_cross_pass_kv_queries` (24,2304) | rows 2j,2j+1 = k,v of core layer j |
| both above | merge → `model.language_model.message_input.query` (12,4,2304), order q,k,v,mlp |

`attn_queries.23` maps to layer 24, which does not exist (23 attention_input modules for
layers 1–23). Treat as a harmless orphan; confirm against a real HF load.

### Per layer (i = 0..23)
| checkpoint | HF |
|---|---|
| `attn_norm.weight` | `input_layernorm.weight` |
| `mlp_norm.weight` | `post_attention_layernorm.weight` |
| `attn.wq_du.weight` | `self_attn.q_proj.linear.weight` |
| `attn.wk_dv.weight` | `self_attn.k_proj.linear.weight` |
| `attn.wv_dv.weight` | `self_attn.v_proj.linear.weight` |
| `attn.wo_ud.weight` | `self_attn.o_proj.linear.weight` |
| `attn.gate_attn.weight` | `self_attn.o_proj.gate_proj.weight` |
| `attn.attn_sink_proj.weight` | `self_attn.sink_proj.weight` |
| `attn.q_l2_clip.scale` | `self_attn.q_proj.norm.scale` |
| `attn.wr_du.weight` | `self_attn.relative_position_bias.linear.weight` |
| `attn.rel_logits_proj.proj` | `self_attn.relative_position_bias.proj` |
| `attn.rel_l2_clip.scale` | `self_attn.relative_position_bias.norm.scale` |
| `attn.k_sconv.weight` (sliding) | `self_attn.k_proj.sconv.conv1d.weight` |
| `attn.v_sconv.weight` (sliding) | `self_attn.v_proj.sconv.conv1d.weight` |
| `attn.k_sconv.static_w` / `.weight_proj` (global) | `self_attn.k_proj.sconv.static_weight` / `.weight_proj.weight` |
| `attn.v_sconv.*` (global) | `self_attn.v_proj.sconv.*` |
| `attn_sconv.weight` | `post_attention_convolution.conv1d.weight` |
| `mlp_sconv.weight` | `post_mlp_convolution.conv1d.weight` |
| `q_input_norm` / `k_input_norm` / `v_input_norm` (global) | `q_input_layernorm` / `k_input_layernorm` / `v_input_layernorm` |
| `mlp.gate.weight` | `mlp.gate.weight` |
| `mlp.gate.bias` | `mlp.gate.e_score_correction_bias` |
| `controllers.attn.{down,gain,forget}_weight` | `q_message_gate.{down_proj,message_gate_proj,forget_gate_proj}.weight` |
| `controllers.k.*` / `.v.*` / `.mlp.*` | `k_message_gate.*` / `v_message_gate.*` / `mlp_message_gate.*` |

### Expert packing (the important one)

`w13` is **(E, out=2304, in_bytes)**, `in_bytes = 2304/2` (NVFP4) or `2304` (MXFP8).
`w2` is **(E, out=2304, in_bytes)**, `in_bytes = 1152/2` or `1152`.
So the checkpoint stores `(E, out, in)`, i.e. the transpose of `nn.Linear` weight layout
`(out, in)` — but 3-D per expert, already `(out, in)`. **No transpose is applied.**

Dequant (from `Inkling2LoopDequantizeExperts`):
```
NVFP4 (amax present):
  codes   = w[k] viewed as uint8 -> (low nibble, high nibble) via E2M1 table
            (even element = low nibble)
  values  = packed * scale[..., blk]  * (amax[e] / outer_scale_denominator)
            scale block = 16, scale dtype FP8_E4M3
MXFP8 (amax absent):
  values  = w[k].float() * exp2(scale - 127),  scale block = 32, scale dtype UE8M0
```
Then `Interleave(dim=1)` de-interleaves the packed row order
`[g0,u0,g1,u1,...]` → `[g0..g1151, u0..u1151]` to get `gate_up_proj`.
`w2` needs only dequant.

Exact stored shapes per layer (verified from header):
- NVFP4 (0–19): `w13_weight` (128,2304,1152) U8, `w13_weight_scale` (128,2304,144) F8_E4M3,
  `w13_weight_amax` (128,) F32, `w13_weight_outer_scale_denominator` (1,) F32;
  `w2_weight` (128,2304,576) U8, `w2_weight_scale` (128,2304,72) F8_E4M3, `w2_weight_amax`
  (128,), `w2_weight_outer_scale_denominator` (1,).
- MXFP8 (20–23): `w13_weight` (128,2304,2304) F8_E4M3, `w13_weight_scale` (128,2304,72) U8;
  `w2_weight` (128,2304,1152) F8_E4M3, `w2_weight_scale` (128,2304,36) U8.

### Dropped for the text-only GGUF
`model.visual.*` (23 ViT layers + merger, 0.318 B) and `mtp.safetensors` (MTP/draft sidecar,
3 layers, BF16, target application ids 5/23/41/48).

## 10. Per-layer parameter table (logical, dequantized)

| layer | attn | core? | expert quant | expert params | non-expert params |
|---:|---|---|---|---:|---:|
| 0–4 | sliding | no | NVFP4 | 1.019 B | 27.24 M |
| 5 | global | no | NVFP4 | 1.019 B | 27.39 M |
| 6–10 | sliding | yes | NVFP4 | 1.019 B | 29.60 M |
| 11 | global | yes | NVFP4 | 1.019 B | 29.75 M |
| 12–16 | sliding | yes | NVFP4 | 1.019 B | 29.60 M |
| 17 | global | yes | NVFP4 | 1.019 B | 29.75 M |
| 18–19 | sliding | no | NVFP4 | 1.019 B | 27.24 M |
| 20–22 | sliding | no | MXFP8 | 1.019 B | 27.24 M |
| 23 | global | no | MXFP8 | 1.019 B | 27.39 M |

Non-expert delta on core layers = 4 message gates (≈ 2.36 M). Full JSON:
`results/tensor-map.json`; HF names: `results/hf-state-dict-names.json`.

## 11. Non-standard components

Pieces that no stock transformer attention/MLP stack expresses:

1. **No RoPE** — attention is position-blind except for the learned relative bias.
2. **Block attention residual** — a bank of ≤5 hidden states read by learned queries
   (a tiny cross-attention per sublayer).
3. **Loop with reset** — 12-layer core run 3× from the same pre-loop state, per-pass KV cache,
   per-layer message gates consuming the previous pass.
4. **Per-token attention sinks** and **gated output projection**.
5. **Dynamic short conv** on global K/V (per-token taps) and **static short conv** on
   sliding K/V, plus post-attn/post-MLP convs; 4 conv states per layer.
6. **Q/K L2 clipping** and position **log scaling**.
7. **Softcapped SwiGLU** in experts; **sigmoid router with e_score_correction_bias**.
8. **F32 residuals**: block states, gates and expert accumulation are F32; hidden states BF16.

The loop itself is control flow, not a new op: run layers 6..17 three times with weight
reuse. The per-pass KV cache multiplies KV memory by 3 for the 12 core layers
(48 KV tensors at 1k context ≈ 200 MB).

## 12. Open questions

1. `attn_queries.23` orphan: drop or remap? Verify with a real HF load.
2. Does `message_input` read the *pre-norm* previous hidden state? Yes — `prev_x` is the raw
   hidden state after the last core layer, plus prev block bank.
3. Exact numerics of `block_states` in F32 vs BF16 for parity: HF keeps them F32
   (`hidden_states = hidden_states.float()` at layer input; block cat uses `hidden_states`).
4. DSpark drafter layout — see `OPEN-QUESTIONS.md`; the vendor `mtp.safetensors` is a
   3-layer block drafter with a Markov head.
5. Vision projector (`proj_in` (4096,4096), `proj_out` (2304,4096)) is larger than the ViT
   itself; irrelevant for text-only.
