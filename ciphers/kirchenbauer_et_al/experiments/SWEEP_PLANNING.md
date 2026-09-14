# FineWeb hyperparameter sweep planning

Generate a resolved experiment manifest, review its estimated cost, and edit the
specification before committing to training. This first implementation provides
**`generate` and `estimate` only**. Neither command starts training, reads the
FineWeb cache, downloads a model, or accesses the W&B network. A sequential
`run` command and its results/checkpoint handling are deferred.

## Commands

Run from the repository root with `STEGO_ARTIFACTS_DIR` set to the artifact root
you use on the training node. The wrapper activates the `stego` Conda environment
and locates the repository root in the same way as `run_bit_training_sequence.sh`.
Planning itself is CPU-only and does not need the four GPUs to be present.

```bash
bash ciphers/kirchenbauer_et_al/scripts/run_hyperparameter_sweep.sh \
  generate \
  --spec ciphers/kirchenbauer_et_al/experiments/hyperparameter_sweep.yaml \
  --output sweeps/initial/manifest.json

bash ciphers/kirchenbauer_et_al/scripts/run_hyperparameter_sweep.sh \
  estimate --manifest sweeps/initial/manifest.json
```

The equivalent Python entrypoint is
`conda run -n stego python -m ciphers.kirchenbauer_et_al.src.sweep_kl_fineweb`.
It uses Click. The existing individual-training CLI still uses argparse.

`generate` refuses to overwrite an existing manifest. After changing the YAML,
use a different output directory so trial outputs stay separate. `estimate`
replaces its two reports and records the input manifest's SHA-256 digest.
Both commands validate input using Pydantic and reject unknown fields.

To select another model, override the selected presets at generation time:

```bash
bash ciphers/kirchenbauer_et_al/scripts/run_hyperparameter_sweep.sh \
  generate \
  --spec ciphers/kirchenbauer_et_al/experiments/hyperparameter_sweep.yaml \
  --model qwen3-1.7b --model qwen3-0.6b \
  --output sweeps/smaller-models/manifest.json
```

This creates 1,280 trials. Repeat `--model qwen3-4b` too to create all 1,920.
You can edit model identifiers and local-batch targets in the YAML; those values
are presets, not hardcoded memory limits.

## Grid and equal document exposure

The default selects Qwen3-4B and expands 640 coordinates:

| Axis | Values |
|---|---|
| Bits | 1, 2, 4, 8 |
| Learning rate | 0.0001, 0.0003, 0.001, 0.004, 0.01 |
| Global batch | 8, 16, 32, 128 |
| Objective | NLL + alpha × KL for alpha 0.25, 1, 4; KL alone |
| Strategy | block, modulo |

YAML `null` in `alphas` selects `loss_mode: ignore_prefix`. Its resolved alpha
is 1 solely because the existing training schema requires a number; the trainer
ignores alpha in that mode. Alpha scales **KL**, not NLL. Compare unweighted KL
across these objectives, not total loss or the existing alpha-weighted data loss.
The full grid retains both strategies at one bit, where their position selections
are equivalent.

The manifest specifies validation cache rows `[0, 32)` and training rows
`[32, 4128)`, in the cache loader's existing order, with training seed 42.
A future runner must take those slices **before** sampling/shuffling and preserve
the cache contents between trials. The current planning commands do not verify
cache availability or snapshot its contents. Passing only a trial's training
configuration to the existing training CLI does not implement this bounded
selection contract; this manifest is not yet an executable training plan.

Each trial sees 4,096 documents. Warmup spans 512 documents, and validation runs
after each 2,048 documents, for two validation events. The input token ceiling is
16,777,216 at max_length 4096; actual non-padding tokens depend on document length,
prefix length, and tokenizer. This is an equal-document budget, not an exact
count of non-padding tokens.

| Global batch | Optimizer steps | Warmup steps | Eval interval (steps) |
|---:|---:|---:|---:|
| 8 | 512 | 64 | 256 |
| 16 | 256 | 32 | 128 |
| 32 | 128 | 16 | 64 |
| 128 | 32 | 4 | 16 |

There are four training processes by default. The generator chooses the largest
local batch no larger than the model's target that divides global_batch / world_size.
This makes small global batches feasible even when a small model supports a
larger local batch. Gradient accumulation = global / (world_size × local).

| Model | Local-batch target | Actual local batches for globals 8 / 16 / 32 / 128 | Accumulation |
|---|---:|---|---|
| Qwen3-4B | 2 | 2 / 2 / 2 / 2 | 1 / 2 / 4 / 16 |
| Qwen3-1.7B | 4 | 2 / 4 / 4 / 4 | 1 / 1 / 2 / 8 |
| Qwen3-0.6B | 8 | 2 / 4 / 8 / 8 | 1 / 1 / 1 / 4 |

## Files and schemas

All generated paths are relative to `STEGO_ARTIFACTS_DIR`. Source YAML and
calibration archive paths are relative to the repository root.

```text
sweeps/initial/
  manifest.json
  estimates.csv
  estimates.md
```

`manifest.json` uses `schema_version: 1` and `stage: planning-only`. Its fields are:

- `name`: sweep identity, also used in W&B tags.
- `specification_sha256`: digest of the effective validated specification,
  including any model selection override.
- `documents`: cache name, validation/training stop indices (exclusive), and seed.
- `timing`: archive sources, safety factor, and explicit overhead allowances.
- `trials`: resolved coordinates, each containing `trial_id`, `model_preset`,
  `world_size`, `resolved_gradient_accumulation_steps`, `training_documents`,
  complete `training` settings, `output_directory`, `log_path`, and
  `adapter_directory`.

The training settings use `global_batch_size` and leave
`gradient_accumulation_steps` null, preserving the existing entrypoint's mutually
exclusive batching interface. The separate resolved value makes it reviewable.
IDs are derived from model preset, bits, LR, global batch, objective, and strategy.
The manifest validates uniqueness, budget arithmetic, and shared dataset slices.

Per-trial paths reserve `runs/<trial-id>/training.log` and
`runs/<trial-id>/adapter/`. Those paths, checkpoints, W&B runs, and `results.json`
are **future runner outputs**; planning creates none of them. The training
`run_name` matches its artifact-relative output directory, as required by the
existing training entrypoint. Planned W&B tags include
`stego-icml-2026-git-archive`, `grid-run`, and `sweep:<name>` so these trials can be
archived and distinguished from hero runs.

`estimates.csv` has one row per trial: coordinates, actual local batch,
accumulation, steps, training/evaluation/overhead seconds, total seconds after
margin, and the timing basis. `estimates.md` contains the overall sequential
node-time, per-trial range, marginal totals for each grid axis, calibration
measurements, archive digests, and assumptions. Marginal totals overlap; do not
sum totals from different axes.

## Timing methodology and initial estimate

The checked-in ZIP archives provide offline calibration from:

- [Batch-32 run cfvhydei](https://wandb.ai/4gate/deleteme/runs/cfvhydei).
- [Batch-32 run ke71bap7](https://wandb.ai/4gate/deleteme/runs/ke71bap7).
- [Batch-128 run 9j4ies5w](https://wandb.ai/4gate/stego-kirchenbauer-prefix-kl/runs/9j4ies5w)
  in the [training project](https://wandb.ai/4gate/stego-kirchenbauer-prefix-kl).

Training intervals use `_runtime` and `train/global_step`; `_step` also counts
evaluation log events and is unsuitable. Separately logged evaluation durations
are subtracted. The estimator takes each run's nearest-rank p95 interval and
normalizes it to device-seconds/document using the inferred process count.

The batch-32 runs infer four processes from local batch 2 and accumulation 4.
The batch-128 run infers one process from local batch 2 and accumulation 64,
even though W&B reports four visible GPUs. Its global batch and validation count
come from the repository YAML named in its archived launch arguments; they are
historical assumptions, not facts independently recorded in that archive.
Those reviewed counts are pinned explicitly beside each archive path in the
specification and manifest. The estimator checks them against explicit archived
CLI flags and never reads the current experiment YAML, so later YAML changes
cannot silently alter historical timing interpretation.

For each trial:

```text
training = documents × slowest_p95_device_seconds_per_document / world_size
           + optimizer_steps × 0.1 seconds
validation = validation_events × validation_documents
             × slowest_observed_eval_seconds_per_document
estimate = (training + validation + 120s startup + 60s finalization) × 1.2
```

Evaluation receives no speedup credit from extra GPUs. Smaller models and larger
local batches initially receive no speedup credit either. These are conservative
4B proxies, not fitted model-size scaling laws. Matching sequence length is
required; no silent sequence-length extrapolation is performed. The GPU scaling
assumes comparable H100 hardware and communication efficiency. Startup,
finalization, and optimizer allowances are configurable assumptions, not
measurements. Downloads, retries, OOMs, cold caches, and contention are excluded.

The initial 640-trial 4B manifest estimates **475.87 node-hours (19.83 days)**,
with **44.21–45.17 minutes per trial**, including margins. Four occupied GPUs mean
approximately 1,903.48 GPU-hours. Timing is nearly constant across batches because
all trials process the same documents; the small variation comes from the
optimizer-step allowance. This is an estimate, not a time cap or guarantee.

## Gradient clipping

Training now exposes `--max-grad-norm`, passes it to `SFTConfig`, and sets 4.0 in
the schema, sweep specification, and four existing bit experiment YAMLs.
The chosen value is based on the user's review of the
[training project](https://wandb.ai/4gate/stego-kirchenbauer-prefix-kl),
[cfvhydei](https://wandb.ai/4gate/deleteme/runs/cfvhydei), and
[ke71bap7](https://wandb.ai/4gate/deleteme/runs/ke71bap7).
Those archived runs used max_grad_norm 1.0; they do not experimentally validate
4.0. Existing launches will therefore use the newly chosen clipping threshold
unless overridden explicitly.

## Validation

Focused tests cover all model/batch/objective partitions, equal document and
schedule budgets, invalid batch/manifest edits, interleaved train/eval history,
misleading visible-GPU counts, estimator arithmetic, manifest serialization,
overwrite/path protection, and explicit clipping-argument overrides. The actual
640-trial manifest and reports are also generated and inspected as an integration
check. No GPU training, convergence, dataset content, live W&B access, or empirical
runtime validation is included.
