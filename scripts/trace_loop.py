#!/usr/bin/env python3
"""Trace the Inkling2Loop graph on a tiny random model (CPU, no weights).

Purpose: confirm the loop/block/residual semantics for the analysis,
and print the exact shapes the block-attention and message reads see per application.
"""
import torch
from transformers.models.inkling2_loop import Inkling2LoopTextConfig, Inkling2LoopForCausalLM
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

torch.manual_seed(0)

cfg = Inkling2LoopTextConfig(
    vocab_size=128,
    unpadded_vocab_size=120,
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=16,
    sliding_window_size=8,
    d_rel=4,
    rel_extent=8,
    log_scaling_n_floor=8,
    log_scaling_alpha=0.1,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
    conv_kernel_size=2,
    moe_intermediate_size=32,
    n_routed_experts=8,
    num_experts_per_tok=2,
    route_scale=8.0,
    qk_l2_clip_threshold=22.627416997969522,
    rel_l2_clip_threshold=8.0,
    n_layers_per_attn_res_block=2,
    loop_layer_index_start=2,
    loop_layer_index_end=6,
    loop_num_passes=3,
    loop_gate_rank=8,
    eos_token_id=1,
    pad_token_id=0,
)
print("layer_types:", cfg.layer_types)
print("total applications:", cfg.total_num_hidden_layers)

model = Inkling2LoopForCausalLM(cfg).eval()
print("params:", sum(p.numel() for p in model.parameters()))

records = []
orig_bar = M.Inkling2LoopBlockAttentionResidual.forward
orig_msg = M.Inkling2LoopMessageAttentionResidual.forward


def bar(self, block_states, partial_block_states):
    records.append(("block_read", tuple(block_states.shape), tuple(partial_block_states.shape),
                    tuple(self.query.shape)))
    return orig_bar(self, block_states, partial_block_states)


def msg(self, block_states, partial_block_states, loop_layer_idx):
    records.append(("message_read", loop_layer_idx, tuple(block_states.shape),
                    tuple(partial_block_states.shape)))
    return orig_msg(self, block_states, partial_block_states, loop_layer_idx)


M.Inkling2LoopBlockAttentionResidual.forward = bar
M.Inkling2LoopMessageAttentionResidual.forward = msg

ids = torch.randint(0, 128, (1, 6))
with torch.no_grad():
    out = model(input_ids=ids, use_cache=False)
print("logits:", tuple(out.logits.shape), "finite:", bool(torch.isfinite(out.logits).all()))

print("\n=== block reads (in execution order) ===")
for r in records:
    if r[0] == "block_read":
        # block_states: [B, T, n_blocks, H]
        print(f"  bank={r[1][2]:2d} queries={r[3][0] if len(r[3]) == 2 else 1}")

print("\n=== message reads ===")
for r in records:
    if r[0] == "message_read":
        print(f"  loop_layer_idx={r[1]:2d} bank={r[2][2]:2d} (prev hidden {r[3]})")

print("\ncache test")
with torch.no_grad():
    o1 = model(input_ids=ids[:, :4], use_cache=True)
    o2 = model(input_ids=ids[:, 4:], cache=o1.cache, use_cache=True,
               position_ids=torch.arange(4, 6).unsqueeze(0))
print("  cache layers:", len(o1.cache.layers), "(24 + 2*12 = 48 in the real model)")
print("  step-2 logits:", tuple(o2.logits.shape), "finite:", bool(torch.isfinite(o2.logits).all()))
