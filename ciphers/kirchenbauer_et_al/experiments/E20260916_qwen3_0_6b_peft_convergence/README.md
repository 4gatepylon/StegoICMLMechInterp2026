# Qwen3 PEFT convergence and ablations

## Hypothesis and interpretation

We test whether Qwen3 Base models can learn the Kirchenbauer control-prefix
objective with LoRA for a message length selected by `--n-bits`. We expect the
optimized loss terms to decrease on training examples and the fixed validation
set over the configured training budget. The entry point compares model sizes,
message lengths, learning rates, and loss settings at a matched data-token
budget. The folder retains the original experiment name; `--model` selects
the base checkpoint for each run.

Decreasing training and validation losses would support using the selected model
for subsequent experiments. Training improvement without validation improvement
would suggest overfitting; flat, increasing, or non-finite losses would motivate
investigating optimization, capacity, or the data pipeline. A falling total loss
alone is insufficient: inspect its two components separately. This experiment
does not establish message-recovery accuracy, text quality, or superiority over
other models; those require separate evaluations or controlled comparisons.

The default budget is **33,554,432 training data tokens**: 256 optimizer steps
at global batch 128 with 1,024 document tokens per example. This uses 32,768
training documents plus 256 validation documents. Compared with the prior
1,024-step, 4,096-token setup at the same batch size, this uses four times fewer
documents and sixteen times fewer data tokens. The shorter run follows the
observation that training loss drops quickly; generalization and message
recovery still require evaluation.

## Configuration and length accounting

`train.py` uses the shared Pydantic configuration, FineWeb loader, collator, and
`PrefixKLTrainer` from `ciphers/kirchenbauer_et_al/src/`. Use `--help` for current
CLI defaults; `experiment_config()` defines the remaining dataset, LoRA,
precision, logging, and checkpoint settings.

| Quantity | Defined by |
| --- | --- |
| Message length | `--n-bits` |
| Data positions per example | `data_length`, set from `DATA_LENGTH` in the experiment configuration |
| Prefix length | Measured with the selected model's tokenizer for the configured bit count |
| Total model-input length | `max_length = data_length + prefix_length + int(prepend_student_bos)` |
| Data positions per bit | `data_length / n_bits` |
| Optimizer steps | `num_training_tokens / (data_length * global_batch_size)` |
| Gradient accumulation | `global_batch_size / (local_batch_size * WORLD_SIZE)` |
| Required cached documents | `validation_samples + max_steps * global_batch_size` |

Step and accumulation calculations must produce integers. The cache loader
checks that enough documents remain after filtering for training and validation.

The tokenizer's control-prefix width is added to `data_length` to derive the
model-input `max_length`, plus one token when `--prepend-student-bos` is enabled
(default: disabled). Teacher BOS is always present. Startup output reports both
input lengths, the resolved BOS ID, and the bit count. Missing BOS in both model
config and tokenizer is an error; EOS is never substituted implicitly.
Prefix overhead can change with `--n-bits` or the tokenizer. The default filters
require GPT-2 length >=756 followed by Qwen length >=1,024. With the padding
guard enabled, all 1,024 data positions contain real document tokens; longer
documents are truncated. The supported message widths are 1, 2, and 4 bits.

`--num-training-tokens` budgets **data** positions, excluding prefixes
and validation. Logged padded-input tokens additionally include the prefix
overhead. Earlier versions applied the configured length to the whole input;
the corrected length setting allocates that budget to data and adds the prefix.

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
CUDA_VISIBLE_DEVICES=0 python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train
```

Alternatively, launch on four visible GPUs with one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --module ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train
```

The script exposes the ablation and batch settings through Click; run with
`--help` for defaults and explanations of when to change each flag. It reads
exported environment variables; it does not load `.env` automatically. Choose a
fresh artifact directory for an independent repeat to avoid reusing checkpoint
paths. Training is not automatically resumed from existing files.

## Ablation flags

| Flag | Effect and reason to change it |
| --- | --- |
| `--model` (alias `--model-name`) | Selects a supported Qwen3 **Base** checkpoint. Larger models test capacity scaling with greater memory and compute requirements; `--help` lists the supported choices. |
| `--n-bits` | Message bits per sequence. More bits increase payload, dividing the same data budget into shorter blocks per bit. |
| `--lr` | Optimizer learning rate. Higher values make larger updates; lower values can stabilize training. |
| `--loss-type` | `nll` uses prefix NLL + alpha × data KL. `ignore_prefix` uses only data KL (ignoring alpha), to test whether learning the prefix itself matters. |
| `--alpha` | In `nll` mode, larger values prioritize data KL relative to prefix NLL; zero trains only the prefix. It does not change the separate LoRA alpha setting. |
| `--delta` | Adds a logit boost to the teacher vocabulary subset encoding the requested bit when encoding is enabled. Increasing delta strengthens that target: selected-vs-unselected token odds are multiplied by `exp(delta)`. This may improve bit recovery at a text-quality cost; measure both. Zero removes the boost, while data KL still trains against the unboosted teacher. |
| `--min-gpt2-document-tokens`, `--max-gpt2-document-tokens` | Inclusive cached GPT-2 document-length bounds, applied first. Default: minimum 756, unlimited maximum. |
| `--min-qwen-document-tokens`, `--max-qwen-document-tokens` | Inclusive full-document counts under the selected Qwen tokenizer, applied only to GPT-2 survivors. Default: minimum 1,024, unlimited maximum. |

See [nested filtering](../../README.md#filtering-cached-training-documents) for
counting semantics and the additional tokenization cost. Both stages precede
the train/validation split, and enough documents must remain for the configured
budget. Active length bounds are included in run/output names to distinguish
filtered ablations. Set both minimum bounds to zero to explicitly disable filtering;
the padding guard remains enabled unless separately disabled.

Learning rate must be positive; alpha and delta must be nonnegative; all three
must be finite. `n_bits` must be between 1 and 4, and `data_length` must divide evenly
into bit blocks. The prefix is outside that budget. Prefix NLL and data
KL are training objectives, not measurements of recovery or text quality.

The [padding guard](../../README.md#padding-guard) is enabled by default.
`--allow-document-padding` explicitly permits right padding; left padding is
always forbidden. Prefer Qwen filtering at or above `data_length` so every
example fills its bit-bearing data positions.

## Model × message-length sweep

The commands below use global batch 32, so the same 33,554,432-token budget
runs for **1,024 steps**. Changing the batch size changes the step count, while
the token budget and the required 33,024 documents stay fixed. All nine runs
can reuse the same cache.

After the setup above, run each command from the repository root. The example
sweep below selects model, bit count, and batch sizes explicitly; other flags
use their current defaults (1,024 data tokens, GPT-2 minimum 756, Qwen minimum 1,024). Each run uses the configured data-token budget,
with steps and accumulation derived as described above. Use the
`torchrun --standalone --nproc_per_node=N --module` launcher instead of `python -m`
for multiple GPUs. Hardware memory fit must be checked for each model.

```bash
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-0.6B-Base --n-bits 1 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-0.6B-Base --n-bits 2 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-0.6B-Base --n-bits 4 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-1.7B-Base --n-bits 1 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-1.7B-Base --n-bits 2 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-1.7B-Base --n-bits 4 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-4B-Base --n-bits 1 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-4B-Base --n-bits 2 --global-batch-size 32 --local-batch-size 2
python -m ciphers.kirchenbauer_et_al.experiments.E20260916_qwen3_0_6b_peft_convergence.train --model Qwen/Qwen3-4B-Base --n-bits 4 --global-batch-size 32 --local-batch-size 2
```

Compare model sizes and bit counts at matched token counts. Falling losses can
support learnability, but objective scales differ across loss/alpha/delta
settings, so total-loss magnitudes alone do not establish which encoding works
best. Run independent recovery and text-quality evaluations on saved adapters.

## Outputs and analysis

Outputs are under
`$STEGO_ARTIFACTS_DIR/<run-name>/`, including local W&B logs. Names encode the
selected model, bits, learning rate, global batch, loss type, alpha, delta,
training-token budget, and a `-student-bos0`/`-student-bos1` suffix added to the
configured base run name. Different budgets and BOS modes use separate checkpoint
directories. Identical configurations still reuse the same name; use a fresh
artifacts directory for independent repeats.
All sizes share W&B project
`E20260916_qwen3_peft_convergence`, which overrides an exported `WANDB_PROJECT`.
Runs receive the existing `stego-icml-2026-git-archive` tag.

After each successful checkpoint save, the shared KL trainer prints
`Saved checkpoint at step <step>: <path>` on the saving process. This console
message is visible without enabling INFO logging; failed saves do not print it.

Plot `train/loss`, `train/prefix_loss`, and `train/data_loss` alongside
`eval/loss`, `eval/prefix_loss`, and `eval/data_loss`, against `train/global_step`.
Compare early and late validation losses and inspect sustained trends rather
than interpreting a single noisy step as convergence.

Checkpoints are saved at the configured `save_steps` cadence and at the final
step, with retention derived to keep every save. Each `checkpoint-<step>/` contains
`adapter_model.safetensors` and `adapter_config.json`, plus tokenizer and Trainer
metadata. PEFT saves only adapter weights; `save_only_model=True` additionally
omits optimizer, scheduler, and RNG state. No merged or full base-model weights
are saved in these checkpoints. They support adapter evaluation, not exact
training resumption. Load the original base model together with an adapter:

```python
import os
from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = Path(os.environ["STEGO_ARTIFACTS_DIR"]) / "<run-name>" / "checkpoint-<step>"
adapter_config = PeftConfig.from_pretrained(checkpoint)
base_model = AutoModelForCausalLM.from_pretrained(adapter_config.base_model_name_or_path)
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
