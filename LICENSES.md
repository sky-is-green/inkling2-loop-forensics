# Licenses and attribution

This repository mixes original analysis/code with Wikipedia-derived evaluation data. The
licenses are separated below.

## Code and original analysis

`scripts/`, the original docs, the curated `results/` JSON files, and the repository
structure are licensed under the **Apache License 2.0**; see `LICENSE`.

## Wikipedia-derived evaluation data

The following files are derived from Wikipedia text distributed through WikiText-2 and
WikiText-103:

- `eval/wiki-blue-64k.txt`
- `eval/long-context/wikitext-2-raw/wiki.test.raw`
- `eval/long-context/prompts/*.txt`
- `eval/long-context/fixed-target/*.txt`

They are distributed under **CC BY-SA 3.0** (https://creativecommons.org/licenses/by-sa/3.0/).
The needle sentences and questions are original, but the haystacks and slices are derived
from the WikiText text and inherit that license. Attribution: WikiText-2 and WikiText-103
were introduced by Merity et al. and are distributed by Salesforce
(https://huggingface.co/datasets/Salesforce/wikitext), derived from Wikipedia.

`eval/prompts.json` contains original short prompts and is Apache-2.0.

## Vendor metadata

`vendor/` contains configuration, a safetensors index, a 2 MB safetensors header prefix, the
chat template and a Hugging Face API response for Inkling 2 Mini. These are metadata, not
weights. Inkling 2 Mini is released by Thinking Machines Lab under the Apache License 2.0.
They are retained for interoperability and analysis; no tensor payloads are redistributed.

## Results and model outputs

The `results/` JSON files contain analysis outputs and model-generated text from Inkling 2
Mini. The model is Apache-2.0; the generated text is included for reproducibility and
analysis. No model weights, checkpoints or GGUFs are included.

## Small-model reproduction

The 23M reproduction was trained from scratch on WikiText-103. Its token arrays and
checkpoints are not included in this repository. The scripts that generate them are
Apache-2.0; the underlying WikiText-103 text remains CC BY-SA 3.0.

## Trademarks

"Thinking Machines", "Inkling" and "Inkling 2" are trademarks of their respective owners.
This is independent analysis, not affiliated with, endorsed by, or sponsored by Thinking
Machines Lab.
