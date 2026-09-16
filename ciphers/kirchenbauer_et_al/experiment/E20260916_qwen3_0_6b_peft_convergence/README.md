# Qwen3-0.6B PEFT convergence

## Hypothesis and interpretation

The base model [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
can learn the one-bit Kirchenbauer control-prefix objective with LoRA. We expect
both the prefix negative log-likelihood and the data KL divergence to decrease
on training examples and the fixed validation set over 1,024 optimizer steps.
This is the smaller counterpart of the existing one-bit Qwen3-4B experiment.

Decreasing training and validation losses would support using this cheaper model
for subsequent experiments. Training improvement without validation improvement
would suggest overfitting; flat, increasing, or non-finite losses would motivate
investigating optimization, capacity, or the data pipeline. A falling total loss
alone is insufficient: inspect its two components separately. This experiment
does not establish message-recovery accuracy, text quality, or superiority over
4B models; those require separate evaluations or controlled comparisons.

## Fixed setup

`train.py` uses the shared Pydantic configuration, FineWeb loader, collator, and
`PrefixKLTrainer` from `ciphers/kirchenbauer_et_al/src/`.

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B-Base` (base, not instruction-tuned) |
| Objective | One bit, block partition, prefix NLL + data KL, alpha 1, delta 2 |
| Dataset | `fineweb-500k`; first 256 documents held out with fixed prefix controls |
| Sequence length | 4,096 padded tokens, token-space concatenation |
| LoRA | Rank 32, alpha 16, dropout 0.05, all linear layers |
| Precision | bfloat16, gradient checkpointing enabled |
| Batch | Global 128; per-device microbatch 8 |
| Schedule | 1,024 optimizer steps; learning rate 0.0003; 50 warmup steps |
| Logging / validation | Every optimizer step / every 4 optimizer steps |
| Checkpoints | Every 32 optimizer steps; retain all 32 |
| W&B run name | `E20260916_qwen3_0_6b_peft_convergence` |

Accumulation is `128 / (8 * WORLD_SIZE)`: 16 microbatches per optimizer step on
one GPU, 8 on two, and 4 on four. Supported process counts divide 16 exactly.
The loader requires at least 131,328 documents (131,072 training plus 256
validation); the existing 500,000-document cache is sufficient.

## Run

From the repository root, activate `stego` and install the repository's GPU
requirements if needed. Use CUDA hardware supporting bfloat16 with enough memory
for the per-device batch and the full-vocabulary KL objective.

```bash
conda activate stego
export STEGO_ARTIFACTS_DIR="${STEGO_ARTIFACTS_DIR:-$PWD/artifacts}"
export WANDB_PROJECT=stego-icml-2026
wandb login
```

If the `fineweb-500k` cache does not already exist, create it once:

```bash
python -m ciphers.kirchenbauer_et_al.src.cache_fineweb --cache-name fineweb-500k --documents 500000
```

Launch on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train
```

Alternatively, launch on four visible GPUs with one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --module ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train
```

The script has a fixed configuration and a Click `--help` command. It reads
exported environment variables; it does not load `.env` automatically. Choose a
fresh artifact directory for an independent repeat to avoid reusing checkpoint
paths. Training is not automatically resumed from existing files.

## Outputs and analysis

Outputs are under
`$STEGO_ARTIFACTS_DIR/E20260916_qwen3_0_6b_peft_convergence/`, including local W&B
logs. The W&B project comes from `WANDB_PROJECT`; runs receive the existing
`stego-icml-2026-git-archive` tag.

After each successful checkpoint save, the shared KL trainer prints
`Saved checkpoint at step <step>: <path>` on the saving process. This console
message is visible without enabling INFO logging; failed saves do not print it.

Plot `train/loss`, `train/prefix_loss`, and `train/data_loss` alongside
`eval/loss`, `eval/prefix_loss`, and `eval/data_loss`, against `train/global_step`.
Compare early and late validation losses and inspect sustained trends rather
than interpreting a single noisy step as convergence.

`checkpoint-32/`, `checkpoint-64/`, ..., `checkpoint-1024/` each contain
`adapter_model.safetensors` and `adapter_config.json`, plus tokenizer and Trainer
metadata. PEFT saves only adapter weights; `save_only_model=True` additionally
omits optimizer, scheduler, and RNG state. No merged or full base-model weights
are saved in these checkpoints. They support adapter evaluation, not exact
training resumption. Load the original base model together with an adapter:

```python
import os
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = Path(os.environ["STEGO_ARTIFACTS_DIR"]) / "E20260916_qwen3_0_6b_peft_convergence" / "checkpoint-1024"
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
model = PeftModel.from_pretrained(base_model, checkpoint).eval()
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

## Verification scope

Local verification covers distributed batch divisibility, CLI dispatch, and
actual 32-step checkpoint cadence, compact contents, retention, and adapter
reload on a tiny randomly initialized CPU Qwen model with synthetic documents.
They omit full 0.6B training, GPU memory fit, distributed execution, live W&B
delivery, and claims of convergence. Existing Kirchenbauer tests cover the shared
data and loss machinery.
