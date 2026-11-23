# Vendor artifact manifest — Inkling 2 Mini (P0-01)

All hashes are sha256 of the local copy. Nothing here is a weight file except the header
prefix; the 19.53 GB checkpoint has **not** been downloaded.

| local file | source | bytes | sha256 |
|---|---|---:|---|
| `inkling-config.json` | `thinkingmachines/Inkling-2-Mini` @ `b85c3cfa615fc413cc9af3379f491b963f3b586a` `config.json` | 2 722 | `a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2` |
| `inkling-generation_config.json` | same repo `generation_config.json` | 98 | (see `inkling-sha.txt`) |
| `inkling-hf_quant_config.json` | same repo `hf_quant_config.json` | 67 924 | `7c656207d3ce37879c7be209a036b0dde43bf01f78159bff06880b868ed57143` |
| `inkling-model.safetensors.index.json` | same repo `model.safetensors.index.json` | 109 300 | `c1631065b7b00787993522dddc66a68b3abee869192977fb3dcb796e63bc5838` |
| `inkling-header.bin` | first 2 097 152 bytes of `model-00001-of-00001.safetensors` (header is 148 408 B) | 2 097 152 | `6d787ba2114aa93957b120d38d7cc42a01bf4a19772482c944115cf49f6bd990` |
| `inkling-card.md` | same repo `README.md` | 54 870 | `e3b8db13cbe1e3638988a7b26e51a7e2b857c1bbc8ffcaeba251bdbc696e9142` |
| `inkling-api.json` | `https://huggingface.co/api/models/thinkingmachines/Inkling-2-Mini` | 9 579 | — |
| `chat_template.jinja` | extracted from `inkling-api.json` → `config.chat_template_jinja` | 6 868 | — |
| `hf-config/config.json` | copy of `inkling-config.json` (name required by `AutoConfig`) | 2 722 | — |
| `transformers-5.24.0-py3-none-any.whl` | PyPI `transformers` 5.24.0 (2026-11-21) | 13 549 890 | `ff5ebedf9359f94bb63772b6380eba6e7104d010cbbb2cb82c788817ea7486f6` |

The wheel is kept outside this repository; the hash above is the identity.

Vendor checkpoint facts (from index/API, not downloaded):
- file: `model-00001-of-00001.safetensors`, **19 526 019 168 B**, LFS sha256
  `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`
  (the index `total_size` 19 525 870 752 B is the tensor payload; the file adds the header)
- `mtp.safetensors`, **635 801 416 B**, LFS sha256
  `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`
- additional files: `processor_config.json`, `special_tokens_map.json`,
  `tiktoken/tokenizer.model` (3.1 MB), `tokenizer.json` (25.9 MB), `tokenizer_config.json`,
  `chat_template.jinja`
- repo `usedStorage` 20 191 385 756 B (20.19 GB); public, `gated: false`, Apache-2.0
- revision: `b85c3cfa615fc413cc9af3379f491b963f3b586a` (last modified 2026-11-21)

## Download plan (not executed)
Source only: `https://huggingface.co/thinkingmachines/Inkling-2-Mini` at the revision above,
via `huggingface_hub` / `hf download`. No mirrors, no community GGUFs/quants, no vendor
redistributions. Token (if used) read from `~/.cache/huggingface/token`, never printed.
First pass can skip `mtp.safetensors` (0.64 GB) until spec-decode work.

## Not part of this lane
`decision-*` files in `vendor/` are from the community derivative
`eltokh7/inkling-2-mini-decision` and are kept **only** as a code-shape cross-check
(it ships a `modeling.py` and records `transformers_version: 5.24.0`). They are not a
baseline and must not be used as one.
