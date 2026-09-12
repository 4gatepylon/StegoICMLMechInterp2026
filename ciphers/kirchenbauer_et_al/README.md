# A Watermark for Large Language Models (Kirchenbauer et al., 2023)

Experiments based on [A Watermark for Large Language
Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html).

- [`binary_classification_mvp`](binary_classification_mvp/) trains a Qwen base
  model to select one of two fixed red/green policies from a literal text prefix.

## Validations

- **Every bit must show up as a different token in the prompt.** We try all 10-bit strings up to 10 bits on 100 FineWeb documents in [`binary_classification_mvp/inspect_prefix_tokenization.ipynb`](binary_classification_mvp/inspect_prefix_tokenization.ipynb).