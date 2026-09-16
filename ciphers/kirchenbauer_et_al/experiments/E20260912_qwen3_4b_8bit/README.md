# Qwen3-4B, 8-bit prefix KL

Original configuration: 1,024 steps, global batch 128, delta 2.
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
