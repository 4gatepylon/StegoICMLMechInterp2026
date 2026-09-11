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

Teacher logits are computed just in time and are never saved. Two Hugging Face
model instances are kept in CPU memory. For each example, the LoRA student is
moved to CPU, the frozen reference is moved to the accelerator to produce
teacher logits, and then their locations are reversed for forward/backward.
Only one 4B base model occupies accelerator memory at a time. This deliberately
trades runtime for lower accelerator memory use.

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
| `shared/constants.py` | Checkpoints, dataset revision, signals, prefixes, and default paths |
| `shared/data.py` | FineWeb streaming, token budgets, de-duplication, and disjoint splits |
| `shared/colors.py` | Seeded red/green vocabulary partition and prefix tokenization |
| `shared/models.py` | Devices, dtypes, Hugging Face/PEFT loading, and CPU/GPU swapping |
| `shared/objectives.py` | Biased teacher distributions, forward KL, and expected color mass |
| `shared/training.py` | Training loop and teacher-forced validation |
| `shared/metrics.py` | Metric aggregation and dependency-free binary AUROC |
| `shared/artifacts.py` | JSON/JSONL outputs and cross-stage configuration validation |

`shared/__init__.py` is the package's public interface. The entry-point scripts
import from that interface rather than reaching into implementation modules.

## Run the experiment

Install the repository requirements, authenticate with Hugging Face if needed,
and run the scripts from this directory or the repository root.

Stage 1 adapts the model to the unfamiliar null prefix using KL only:

```bash
python ciphers/kirchenbauer_et_al/binary_classification_mvp/finetune_prefix.py
```

Its default output is `outputs/prefix_adapter` within this directory.

Stage 2 starts from that adapter and trains all three policies on a new,
disjoint 64K-token FineWeb split:

```bash
python ciphers/kirchenbauer_et_al/binary_classification_mvp/finetune_encoding.py
```

Its default input is stage 1's output and its default output is
`outputs/encoding_adapter`. Override the chain explicitly when desired:

```bash
python ciphers/kirchenbauer_et_al/binary_classification_mvp/finetune_encoding.py \
  --input-model /path/to/prefix_adapter \
  --output-dir /path/to/encoding_adapter
```

Finally, sample 200 tokens at temperature 0.7 for each label on paired,
held-out FineWeb prompts:

```bash
python ciphers/kirchenbauer_et_al/binary_classification_mvp/evaluate_generation.py
```

Generation is performed only for the true binary classes. The `none` condition
is not sampled; it is a teacher-forced control whose relevant metric is KL to
the original model.

Run `python SCRIPT.py --help` for all options. Useful smoke-test overrides are
small token budgets, fewer generation prompts, and `--max-steps 2`.

## CPU integration test

Run the network-free integration suite from the repository root:

```bash
python -m unittest discover \
  -s ciphers/kirchenbauer_et_al/binary_classification_mvp/tests \
  -p 'test_*.py' \
  -v
```

It mocks the FineWeb stream and checks source-disjoint allocation, color
partitioning, AUROC (including ties), a full CPU distillation/validation/save
step, and stage-1-to-stage-2 LoRA handoff using a tiny native Qwen3 model. It
does not download Qwen weights or require an accelerator.

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
reports binary AUROC and per-class mean counts. Since this experiment uses a
single fixed partition rather than context-dependent random lists, the paper's
analytic null z-test does not apply; paired held-out AUROC is the primary
detection metric.

Each adapter directory also contains `experiment_config.json`, the tokenizer,
and PEFT adapter files. Generated `outputs/` are ignored by Git.
