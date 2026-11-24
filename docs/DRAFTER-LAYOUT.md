# DSpark drafter layout

The vendor release ships a separate 3-layer block drafter, `mtp.safetensors` (0.636 GB,
sha256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`). It is not part of
the language model; it proposes draft blocks for speculative decoding.

Sources:
- `mtp.safetensors` header (all 54 tensors, BF16).
- `config.json → mtp_config`.
- `unsloth/Inkling-2-Mini-GGUF` `dspark-Inkling-2-Mini-Q8_0.gguf` (54 tensors, same set).

## Configuration

```
hidden_size          2304
num_hidden_layers    3
attention heads      24 / 8 KV, head_dim 128
intermediate_size    9216          (dense MLP, not MoE)
local_layer_ids      [0, 2]        (sliding window 1024)
sliding_window       1024
block_size           8
markov_rank          256
markov_head_type     rnn_hidden
target_layer_ids     [5, 23, 41, 48]   (application indices in the 48-layer target)
```

`sliding_window_pattern = [True, False, True]`: layers 0 and 2 are sliding, layer 1 is
global. Attention is **non-causal** over the block (`attention.causal = false`). The drafter
borrows the target's token embeddings and output head; neither is in the file.

## Tensor inventory (vendor → GGUF)

Per layer (×3):

| vendor | shape | GGUF | role |
|---|---|---|---|
| `layers.N.input_layernorm` | 2304 | `blk.N.attn_norm` | pre-attention norm |
| `layers.N.post_attention_layernorm` | 2304 | `blk.N.ffn_norm` | pre-MLP norm |
| `layers.N.dflare_norm` | 2304 | `blk.N.dflare_norm` | extra norm on the context path |
| `layers.N.self_attn.q_proj` | 3072×2304 | `blk.N.attn_q` | query |
| `layers.N.self_attn.k_proj` | 1024×2304 | `blk.N.attn_k` | key |
| `layers.N.self_attn.v_proj` | 1024×2304 | `blk.N.attn_v` | value |
| `layers.N.self_attn.o_proj` | 2304×3072 | `blk.N.attn_output` | output |
| `layers.N.self_attn.k_proj_ctx` | 1024×2304 | `blk.N.attn_k_ctx` | key from target context |
| `layers.N.self_attn.v_proj_ctx` | 1024×2304 | `blk.N.attn_v_ctx` | value from target context |
| `layers.N.self_attn.q_norm` | 128 | `blk.N.attn_q_norm` | per-head Q norm |
| `layers.N.self_attn.k_norm` | 128 | `blk.N.attn_k_norm` | per-head K norm |
| `layers.N.self_attn.k_norm_ctx` | 128 | `blk.N.attn_k_ctx_norm` | per-head context-K norm |
| `layers.N.mlp.gate_proj` | 9216×2304 | `blk.N.ffn_gate` | dense gate |
| `layers.N.mlp.up_proj` | 9216×2304 | `blk.N.ffn_up` | dense up |
| `layers.N.mlp.down_proj` | 2304×9216 | `blk.N.ffn_down` | dense down |

Markov head and block tensors:

| vendor | shape | GGUF | role |
|---|---|---|---|
| `markov_head.markov_w1` | 201024×256 | `markov_w1` [256,201024] | state → vocab logits |
| `markov_head.hidden_out.weight` | 2304×256 | `dspark_hidden_out` [256,2304] | state → hidden |
| `markov_head.hidden_out.bias` | 2304 | `dspark_hidden_out.bias` | |
| `markov_head.rnn_proj.weight` | 768×5120 | `joint_proj` [5120,768] | RNN update, 3×256 |
| `markov_head.rnn_proj.bias` | 768 | `joint_proj.bias` | |
| `markov_head.dflare_norm` | 2304 | `dspark_norm` | head norm |
| `fuse_weight` | 4×4 | `dspark_fuse` | mixes the 4 target taps |
| `mask_embed` | 2304 | `mask_embd` | masked/placeholder slot embedding |
| `norm.weight` | 2304 | `output_norm` | final norm |

The GGUF conversion transposes the 2-D weights for ggml (`[out,in]` → `[in,out]`). No tensor
is missing: 54 in the vendor file, 54 in the GGUF.

## Forward-pass hypothesis

1. The block of 8 draft slots is embedded with the target's token embeddings, plus
   `mask_embed` for slots that are masked. `sample_from_anchor = false`: the first slot is
   the last accepted token, the rest are drafts.
2. The target taps `[5, 23, 41, 48]` supply four hidden states. `fuse_weight` (4×4) mixes
   them, and the result drives the `k_proj_ctx` / `v_proj_ctx` / `k_norm_ctx` path — a
   context-injection attention alongside the ordinary self-attention.
3. Three dense layers (0/2 sliding, 1 global, non-causal) produce the block hidden states.
4. The Markov head is `rnn_hidden`: `rnn_proj` maps a 5120-wide concatenation to 3×256
   (GRU-like gates/candidate), the 256 state is projected back to hidden by `hidden_out`,
   and `markov_w1` maps the state to the 201024-way vocabulary logits. The head is
   semi-autoregressive across the 8 slots, so slot *i* conditions on the state of slot *i−1*.
5. Draft logits are compared with the target in the usual speculative acceptance rule.
   `has_confidence_head = false`: there is no learned confidence cutoff in this release.

The exact gate order, the 4×4 fuse application, and the slot/anchor ordering are **not**
recovered yet. The public Unsloth prebuilt runs this drafter, but its source commit
(`189f431bf`) is not public; mainline's DSpark is a different variant (`markov_w2`,
`conf_proj`).

## Why this matters

The drafter is a dense 3-layer model, not a mini-MoE: 24/8 heads, 9216-wide MLP, 3 layers,
0.32 B parameters. Its cost is small relative to the target, and its only connection to the
target is four hidden-state taps plus the shared embedding/output. That makes it a clean
port target once the forward pass is pinned down.

## Open questions

- What exactly does `fuse_weight` multiply: the four tap features per channel, per slot, or
  per attention head?
- Is `rnn_proj` a GRU (reset/update/candidate) or a custom 3-gate cell, and what are the
  activations?
- How are the 8 slots ordered and masked, and where does the anchor enter?
- How are draft logits combined with the target's at acceptance time?
- Can the vendor `mtp.safetensors` forward pass be reproduced and matched to the Unsloth
  prebuilt's acceptance (~0.46 on our probe)?
