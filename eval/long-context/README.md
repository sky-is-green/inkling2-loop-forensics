# Long-context eval data

Frozen inputs for the 32k loop study. No model weights are stored here.

## Corpus

- `wikitext-2-raw/wiki.test.raw` — WikiText-2 test split (Merity et al., 2016), derived from
  Wikipedia; CC BY-SA 3.0. 312,991 tokens with the Inkling tokenizer.
- Source: `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`
- sha256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`

## Needle prompts

`prompts/` contains the 8k / 16k / 32k haystacks with the code `7391-ALPHA` inserted at
10 / 50 / 90 %. `prompts/manifest.json` records byte counts, hashes, depth and the expected
answer. Build them with `scripts/build_needle_prompts.py`.

## Fixed-target slices

`fixed-target/` contains four slices that all end at the same paragraph boundary. The first
`n_ctx` tokens of each slice are followed by the same 2048-token target; the text after that
boundary only satisfies the harness's length check and is never scored. This separates the
context-length effect from target difficulty. `fixed-target/manifest.json` records the
nominal context, actual `n_ctx`, token counts and hashes. Build them with
`scripts/make_fixed_target_slices.py` (`llama-tokenize` is used to hit exact token counts):

```bash
python3 scripts/make_fixed_target_slices.py \
  --model /path/to/Inkling-2-Mini.gguf \
  --tokenize /path/to/llama-tokenize
```

## Scored target

The target is the last 2048 tokens of the first `n_ctx` tokens of each slice. Nominal
4k/8k/16k/32k use `n_ctx` 5,120 / 9,216 / 17,408 / 33,792 because the strided harness adds
`stride/2 = 1024`.
