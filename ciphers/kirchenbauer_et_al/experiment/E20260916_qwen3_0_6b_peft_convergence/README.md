# Qwen3 PEFT convergence and ablations

## Hypothesis and interpretation

The base model [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
can learn the one-bit Kirchenbauer control-prefix objective with LoRA. We expect
both the prefix negative log-likelihood and the data KL divergence to decrease
on training examples and the fixed validation set over 1,024 optimizer steps.
This is the smaller counterpart of the existing one-bit Qwen3-4B experiment.
The same entry point also compares Qwen3 Base models at 0.6B, 1.7B, and 4B,
message lengths, learning rates, and loss settings while holding the training
token budget fixed. The folder retains the original experiment name.

Decreasing training and validation losses would support using this cheaper model
for subsequent experiments. Training improvement without validation improvement
would suggest overfitting; flat, increasing, or non-finite losses would motivate
investigating optimization, capacity, or the data pipeline. A falling total loss
alone is insufficient: inspect its two components separately. This experiment
does not establish message-recovery accuracy, text quality, or superiority over
4B models; those require separate evaluations or controlled comparisons.

## Default setup

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
| W&B project | `E20260916_qwen3_peft_convergence` |
| W&B run name | `qwen3-0.6b-1bit-lr0.0003-gb128-nll-a1-d2` |

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

The script exposes the ablation and batch settings through Click; run with
`--help` for defaults and explanations of when to change each flag. It reads
exported environment variables; it does not load `.env` automatically. Choose a
fresh artifact directory for an independent repeat to avoid reusing checkpoint
paths. Training is not automatically resumed from existing files.

## Ablation flags

| Flag | Default | Effect and reason to change it |
| --- | --- | --- |
| `--model` (alias `--model-name`) | `Qwen/Qwen3-0.6B-Base` | Accepts only Qwen3 **Base** at 0.6B, 1.7B, or 4B. Larger models test capacity scaling with greater memory and compute requirements. |
| `--n-bits` | `1` | Message bits per sequence. More bits increase payload, dividing the same text into shorter blocks per bit. The sweep uses 1, 2, 4, and 8, which evenly partition the text after its control prefix. |
| `--lr` | `3e-4` | Optimizer learning rate. Higher values make larger updates; lower values can stabilize training. |
| `--loss-type` | `nll` | `nll` uses prefix NLL + alpha × data KL. `ignore_prefix` uses only data KL (ignoring alpha), to test whether learning the prefix itself matters. |
| `--alpha` | `1` | In `nll` mode, larger values prioritize data KL relative to prefix NLL; zero trains only the prefix. It does not change LoRA alpha, which stays 16. |
| `--delta` | `2` | Adds a logit boost to the teacher vocabulary subset encoding the requested bit when encoding is enabled. Increasing delta strengthens that target: selected-vs-unselected token odds are multiplied by `exp(delta)`. This may improve bit recovery at a text-quality cost; measure both. Zero removes the boost, while data KL still trains against the unboosted teacher. |

Learning rate must be positive; alpha and delta must be nonnegative; all three
must be finite. `n_bits` must be positive, and the shared collator requires the
post-prefix text length to divide evenly into bit blocks. Prefix NLL and data
KL are training objectives, not measurements of recovery or text quality.

## Model × message-length sweep

After the setup above, run each command from the repository root. These 12 runs
use global batch 32, local batch 2, and otherwise default settings (learning rate
3e-4, loss `nll`, alpha 1, delta 2). Each processes 536,870,912 padded training
tokens over 4,096 optimizer steps. On one GPU, accumulation is 16; on multiple
GPUs it is `16 / WORLD_SIZE`, so the process count must divide 16. Use the
`torchrun --standalone --nproc_per_node=N --module` launcher instead of `python -m`
for multiple GPUs. Hardware memory fit must be checked for each model.

```bash
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-0.6B-Base --n-bits 1 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-0.6B-Base --n-bits 2 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-0.6B-Base --n-bits 4 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-0.6B-Base --n-bits 8 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-1.7B-Base --n-bits 1 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-1.7B-Base --n-bits 2 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-1.7B-Base --n-bits 4 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-1.7B-Base --n-bits 8 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-4B-Base --n-bits 1 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-4B-Base --n-bits 2 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-4B-Base --n-bits 4 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiment.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-4B-Base --n-bits 8 --global-batch-size 32 --local-batch-size 2
```

Compare model sizes and bit counts at matched token counts. Falling losses can
support learnability, but objective scales differ across loss/alpha/delta
settings, so total-loss magnitudes alone do not establish which encoding works
best. Run independent recovery and text-quality evaluations on saved adapters.

## Outputs and analysis

Outputs are under
`$STEGO_ARTIFACTS_DIR/<run-name>/`, including local W&B logs. The default run name
is `qwen3-0.6b-1bit-lr0.0003-gb128-nll-a1-d2`. Names encode model, bits, learning
rate, global batch, loss type, alpha, and delta. All sizes share W&B project
`E20260916_qwen3_peft_convergence`, which overrides an exported `WANDB_PROJECT`.
Runs receive the existing `stego-icml-2026-git-archive` tag.

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

checkpoint = Path(os.environ["STEGO_ARTIFACTS_DIR"]) / "qwen3-0.6b-1bit-lr0.0003-gb128-nll-a1-d2" / "checkpoint-1024"
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
model = PeftModel.from_pretrained(base_model, checkpoint).eval()
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

## Verification scope

Local verification covers all 12 documented model/bit commands, invalid model
and numeric parameters, objective forwarding, derived W&B names, distributed
batch divisibility, CLI dispatch, and
actual 32-step checkpoint cadence, compact contents, retention, and adapter
reload on a tiny randomly initialized CPU Qwen model with synthetic documents.
Tests omit full-size training, GPU memory fit, distributed execution, live W&B
delivery, and claims of convergence. Existing Kirchenbauer tests cover the shared
data and loss machinery.
