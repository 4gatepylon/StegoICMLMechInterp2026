# Prefix-selected red/green policies

This experiment asks whether `Qwen/Qwen3-4B-Base` can internalize a one-bit
version of the soft watermark from Kirchenbauer et al. A literal prefix selects
one of three policies:

```text
<encoding> <do_encoding> yes </do_encoding> <encoding_value> 0 </encoding_value> </encoding>
<encoding> <do_encoding> yes </do_encoding> <encoding_value> 1 </encoding_value> </encoding>
<encoding> <do_encoding> no </do_encoding> <encoding_value> none </encoding_value> </encoding>
```

These strings are ordinary text. The tokenizer is not extended and no special
tokens are added. Label `0` selects red, label `1` selects green, and `none`
selects the original model policy.

`shared.compile_prefix(bits)` constructs the text protocol. It accepts either a
non-empty binary string (including a future multi-bit value such as `"010"`) or
`None` for the original policy, and rejects malformed values. This experiment's
three policies are compiled from `"0"`, `"1"`, and `None`, so all current runs
remain one-bit by default.

## Policy targets

All non-special vocabulary IDs are shuffled once with a seeded PRNG and divided
into equal green and red halves (`gamma = 0.5`). Unlike the paper, the partition
is fixed rather than re-keyed from the preceding token. This is intentional: the
experiment tests whether one prefix bit can select between two complementary
global policies.

For original-model logits `z`, color set `C_b`, and `delta = 2.0`, the encoded
teacher distribution is

```text
q_b = softmax(z + delta * 1[token in C_b]).
```

This is the paper's soft-logit rule. The null teacher is `q_none = softmax(z)`.
Training minimizes the forward KL `KL(q || p_student)` at every FineWeb
prediction position. The reference sees only raw FineWeb history. The student
sees the literal control prefix followed by the identical history. Prefix
positions are excluded from the loss.

Teacher logits are computed just in time and are never saved. Training uses one
Hugging Face PEFT model resident on the selected device. For each batch, the
model first sees padded raw FineWeb histories in `eval()` mode inside
`disable_adapter()` and `torch.inference_mode()`; this produces the original
base-policy teacher logits. The same model then returns to `train()` mode with
LoRA enabled and sees the control prefix plus the identical histories. Padding
positions are excluded from the loss and metrics, and the final partial batch
is retained. `training.batch_size` and `training.eval_batch_size` control the
training and teacher-forced validation batches; the corresponding CLI flags
are `--batch-size` and `--eval-batch-size`.

In normal Python mode, model construction performs a small equivalence
assertion: it captures pristine base-model logits before PEFT wrapping and
checks that disabling the adapter reproduces them exactly afterward. Teacher
passes also assert that adapters disable and restore correctly, and each loss
asserts that KL is finite and not exactly zero. Run Python with `-O` to omit
these diagnostic assertions, including the extra model-construction forwards.

## Disjoint data

All scripts reproduce the same stream:

```python
load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True,
    revision="9bb295ddab0e05d785b879661af7260fed5140fc",
).shuffle(seed=42, buffer_size=10_000)
```

Whole source documents are allocated sequentially to four splits, and exact
text hashes are checked for overlap. A source document can occur in only one
split. The pinned dataset revision makes those boundaries stable across runs.
The first stage writes the complete tokenized selection to
`$ARTIFACTS_DIR/cache/corpus_splits.json`; every later stage loads those exact
IDs instead of accessing FineWeb. Cache metadata pins the dataset, corpus
arguments, and tokenizer implementation. A mismatch fails loudly rather than
silently training on different data. Use `--cache-dir PATH` to share or relocate
the cache.
The defaults are:

| Split | Default size | Use |
| --- | ---: | --- |
| `prefix_train` | at least 65,536 prediction tokens | Stage 1, `none` only |
| `encoding_train` | at least 65,536 unique prediction tokens | Stage 2, reused under `0`, `1`, and `none` |
| `validation` | at least 16,384 prediction tokens | Teacher-forced KL and expected color counts |
| `generation` | 256 documents × 128 prompt tokens | Sampled evaluation of labels `0` and `1` |

Documents are truncated to 256 tokens for training. Because documents are kept
whole and the budget is a lower bound, a split can exceed its requested token
count by at most one accepted sequence. Keep all corpus-related CLI options the
same across scripts so that boundaries remain identical.

## Code organization

The three executable scripts use a `shared` package split by responsibility:

| Module | Responsibility |
| --- | --- |
| `shared/constants.py` | Checkpoints, dataset revision, signals, prefixes, and environment names |
| `shared/configuration.py` | Strict JSON/YAML schema, path resolution, and CLI precedence |
| `shared/data.py` | FineWeb streaming, token budgets, de-duplication, and disjoint splits |
| `shared/colors.py` | Seeded red/green vocabulary partition and prefix tokenization |
| `shared/models.py` | Devices, dtypes, Hugging Face/PEFT loading, and adapter-disabled teacher passes |
| `shared/objectives.py` | Biased teacher distributions, forward KL, and expected color mass |
| `shared/training.py` | Training loop and teacher-forced validation |
| `shared/tracking.py` | Optional W&B runs and per-policy metric curves |
| `shared/artifacts.py` | JSON/JSONL outputs and cross-stage configuration validation |

`shared/__init__.py` is the package's public interface. The entry-point scripts
import from that interface rather than reaching into implementation modules.

## Run the experiment

Install the repository and experiment requirements, authenticate with Hugging
Face if needed, log in to W&B, and run commands from the repository root:

```bash
pip install -r requirements.txt
pip install -r ciphers/kirchenbauer_et_al/binary_classification_mvp/requirements.txt
wandb login
```

The official configuration runs all three stages with Qwen3-4B and the full
token budgets described above:

```bash
ARTIFACTS_DIR=ciphers/kirchenbauer_et_al/binary_classification_mvp/artifacts/official \
  python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.run_experiment \
  --config ciphers/kirchenbauer_et_al/binary_classification_mvp/configurations/official.yaml \
  --wandb-mode online \
  --wandb-project stego-kirchenbauer-binary-classification \
  --wandb-run-name qwen3-4b-base-fineweb-64k
```

The human-runnable CPU integration configuration runs the same three scripts
and codepaths with a deterministic one-layer Qwen3 model and tiny budgets:

```bash
ARTIFACTS_DIR=ciphers/kirchenbauer_et_al/binary_classification_mvp/artifacts/cpu_smoke \
  python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.run_experiment \
  --config ciphers/kirchenbauer_et_al/binary_classification_mvp/configurations/cpu_smoke.yaml \
  --wandb-mode offline \
  --wandb-project stego-kirchenbauer-binary-classification \
  --wandb-run-name qwen3-tiny-cpu-smoke
```

The CPU run covers the real Qwen tokenizer, FineWeb streaming and disjoint
splits, cache creation/reuse, batch-size-two adapter-disabled teacher passes,
LoRA training, KL objectives, checkpoint handoff, generation, color counts, and
AUROC. It does not download the 4B weights. It does access Hugging Face for the
tokenizer and FineWeb unless they are cached.

The official configuration defaults to online W&B logging. It creates two runs
in the `stego-kirchenbauer-binary-classification` project, grouped under the
base run name:

- `qwen3-4b-base-fineweb-64k-stage1-prefix`
- `qwen3-4b-base-fineweb-64k-stage2-encoding`

Stage 1 logs `loss/train/none` and `loss/validation/none`. Stage 2 logs separate
`loss/{train,validation}/{red,green,none}` curves. Both runs also log expected
red, green, and uncolored counts and rates under `policy/...`. The CPU config
uses the same integration in offline mode under the base name
`qwen3-tiny-cpu-smoke`; its local W&B files remain inside `ARTIFACTS_DIR`.
Set `wandb.entity` in YAML or pass `--wandb-entity` when the project belongs to
a specific team. Use `--wandb-mode disabled` to keep a training run fully
local. Dry runs never initialize W&B, even when the configuration says online.

`ARTIFACTS_DIR` is mandatory for the launcher and all three stage scripts. An
unset or empty value fails immediately. By default, the code writes the
following layout beneath it:

```text
$ARTIFACTS_DIR/
├── cache/
│   ├── corpus_splits.json
│   └── color_partition.json
├── prefix_adapter/
├── encoding_adapter/
├── generation_evaluation/
└── dry_run.json  # only when --dry-run is used
```

The color cache contains the exact agreed-upon red, green, and special token-ID
sets. Its metadata pins the tokenizer implementation, model vocabulary size,
partition seed, and cache schema. It is validated for disjointness, complete
vocabulary coverage, and an even red/green split whenever it is loaded.

An absolute `ARTIFACTS_DIR` is used directly. A relative value is resolved
against the shell's current working directory; the commands above therefore
assume they are run from the repository root. The directory is created as
needed. Choose a different root for each run to keep its artifacts isolated.

Both YAML files live in `configurations/`. JSON files with the same schema are
also accepted. Configuration paths passed to `--config` are interpreted
relative to the shell's current working directory.

Inside a configuration, `paths_relative_to` must be either `repo_root` or
`cwd`. It controls `model.weights_path`, `model.config_path`, and
`model.tokenizer_path`; it does not affect `ARTIFACTS_DIR`. Absolute model paths
pass through unchanged. Hugging Face IDs use the separate `model.weights` and
`model.tokenizer` fields and are never interpreted as filesystem paths.

`model.config` can contain an inline Hugging Face configuration, while
`model.config_path` can point to a JSON or YAML Hugging Face configuration.
Configuration is optional when pretrained weights already supply it. Setting
both `model.weights` and `model.weights_path`, or both an inline configuration
and `config_path`, is rejected.

The launcher invokes the production entry points in this order:

1. `finetune_prefix.py`
2. `finetune_encoding.py`
3. `evaluate_generation.py`

To run only selected stages, use, for example,
`--stages prefix encoding`. Each stage can also be invoked directly with the
same configuration.

To inspect the exact data selection without loading a model or training, add
`--dry-run` to the launcher:

```bash
ARTIFACTS_DIR=ciphers/kirchenbauer_et_al/binary_classification_mvp/artifacts/dry_run \
  python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.run_experiment \
  --config ciphers/kirchenbauer_et_al/binary_classification_mvp/configurations/official.yaml \
  --dry-run
```

This runs the production tokenizer and FineWeb split-building path once, then
writes `$ARTIFACTS_DIR/dry_run.json`. The report contains every selected
example as token IDs and decoded text, source hashes, per-example lengths,
per-split minimum/maximum/mean/median lengths, requested corpus settings, and
overall document/token counts. It also reports tokenized prefix lengths and the
effective workload after expanding each source example across its policies. No
reference, student, or inference model is loaded, and no optimizer step or
generation occurs.

Stage 1 adapts the model to the unfamiliar null prefix using KL only:

```bash
ARTIFACTS_DIR=ciphers/kirchenbauer_et_al/binary_classification_mvp/artifacts/official \
  python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.finetune_prefix \
  --config ciphers/kirchenbauer_et_al/binary_classification_mvp/configurations/official.yaml
```

Its output is always `$ARTIFACTS_DIR/prefix_adapter`.

Stage 2 starts from that adapter and trains all three policies on a new,
disjoint 64K-token FineWeb split:

```bash
ARTIFACTS_DIR=ciphers/kirchenbauer_et_al/binary_classification_mvp/artifacts/official \
  python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.finetune_encoding \
  --config ciphers/kirchenbauer_et_al/binary_classification_mvp/configurations/official.yaml
```

Its default input is `$ARTIFACTS_DIR/prefix_adapter`, and its output is always
`$ARTIFACTS_DIR/encoding_adapter`. An adapter from elsewhere can still be used
as the input, but it cannot change the output location:

```bash
ARTIFACTS_DIR=/path/to/new/run \
  python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.finetune_encoding \
  --input-model /path/to/existing/prefix_adapter
```

Finally, sample 200 tokens at temperature 0.7 for each label on paired,
held-out FineWeb prompts:

```bash
ARTIFACTS_DIR=ciphers/kirchenbauer_et_al/binary_classification_mvp/artifacts/official \
  python -m ciphers.kirchenbauer_et_al.binary_classification_mvp.evaluate_generation \
  --config ciphers/kirchenbauer_et_al/binary_classification_mvp/configurations/official.yaml
```

Generation is performed only for the true binary classes. The `none` condition
is not sampled; it is a teacher-forced control whose relevant metric is KL to
the original model.

Explicit stage CLI flags override JSON/YAML values, which in turn override the
built-in defaults. Artifact outputs are the exception: they are controlled only
by the required `ARTIFACTS_DIR`. The cache defaults beneath that root, while an
explicit `--cache-dir` can point elsewhere. The launcher also accepts
`--batch-size` and `--eval-batch-size` and forwards them to both training
stages. Relative cache paths are resolved from the invoking shell's current
directory. Run `python -m MODULE --help` from the repository root for the
available overrides.

## Metrics and artifacts

Both training stages write `training_metrics.jsonl`. Training and validation
records report, separately for each applicable signal:

- forward KL to the original or color-biased teacher;
- expected red and green counts, obtained by summing student probability mass;
- expected uncolored count, consisting only of excluded special-token mass;
- normalized red and green rates.

The generation script writes `generations.jsonl` and `summary.json`. It counts
actual sampled red and green token IDs and uses

```text
green_count / (red_count + green_count)
```

as the classifier score, with label `1` as the positive class. `summary.json`
reports `sklearn.metrics.roc_auc_score` and per-class mean counts. Since this
experiment uses a single fixed partition rather than context-dependent random
lists, the paper's analytic null z-test does not apply; paired held-out AUROC is
the primary detection metric.

Each adapter directory also contains `experiment_config.json`, the tokenizer,
and PEFT adapter files. The example in-repository `artifacts/` directory is
ignored by Git.
