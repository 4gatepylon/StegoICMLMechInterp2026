# Qwen3-4B, 4-bit prefix KL

Hypothesis: gated prefix training learns four-bit control using documents with
at least 1,024 Qwen tokens. Falling prefix and data losses would support this
training setup; weak control or worse text quality would limit that conclusion
to optimization rather than reliable message recovery.

The experiment uses `data_length: 1024` with control-prefix tokens added outside
that budget, global batch 128, delta 2, and 256 optimizer steps. This uses
33,554,432 training data tokens and requires 33,024 documents including
256 validation documents. The run was shortened after observing that training
loss drops quickly; this observation does not establish final message quality. It first filters
cached GPT-2 counts to at least 756, then tokenizes only survivors and requires
at least 1,024 Qwen tokens. The padding guard is enabled, so every bit-bearing
data slot must contain a real token. Documents longer than 1,024 Qwen tokens
are truncated. These length filters do not establish text quality or diversity.

Requires `stego`, GPU dependencies, a bfloat16-capable GPU, W&B login, and
`STEGO_ARTIFACTS_DIR` containing the `fineweb-500k` cache (see [setup](../../README.md)).
From the repo root:

```bash
bash ciphers/kirchenbauer_et_al/experiments/E20260912_qwen3_4b_4bit/run.sh
```

Checkpoints: `$STEGO_ARTIFACTS_DIR/qwen3-4b-4bit-filtered-prefix-kl-lora/`.
The loader requires 33,024 documents after both filters for the default run.
The measured cache has 78,724 survivors under `Qwen/Qwen3-4B-Base`.
