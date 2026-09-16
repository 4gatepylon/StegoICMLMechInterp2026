# Qwen3-4B, 8-bit prefix KL

Original run: 1,024 steps, global batch 128, delta 2. The runnable YAML now uses
`data_length: 4096` (prefix extra), following the current trainer’s budget convention.
Hypothesis: gated prefix training learns eight-bit control; weak control or
worse text quality would limit that conclusion to this model and setup.

Requires `stego`, GPU dependencies, a bfloat16-capable GPU, W&B login, and
`STEGO_ARTIFACTS_DIR` containing the `fineweb-500k` cache (see [setup](../../README.md)).
From the repo root:

```bash
bash ciphers/kirchenbauer_et_al/experiments/E20260912_qwen3_4b_8bit/run.sh
```

Checkpoints: `$STEGO_ARTIFACTS_DIR/qwen3-4b-fineweb-prefix-kl-lora/`.
[Original run and results](https://wandb.ai/4gate/stego-kirchenbauer-prefix-kl/runs/9j4ies5w).

The checked-in configuration leaves document-length filtering disabled.
Optional [nested GPT-2 then Qwen bounds](../../README.md#filtering-cached-training-documents)
can be set in `config.yaml`; the current budget requires 131,328 documents
after both stages. Choose a different `run_name` for a filtered run so its
checkpoints do not reuse the original run's directory.
