"""Generate reviewable FineWeb sweep manifests and estimate their sequential cost.

This planning CLI never loads a model, contacts W&B, or starts training. All
configuration boundaries use Pydantic; the manifest stores resolved training
settings rather than depending on future changes to the source YAML.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Self

import click
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic_yaml import parse_yaml_raw_as

from ciphers.kirchenbauer_et_al.src.configuration_kl_fineweb import REPO_ROOT, PrefixKLTrainingConfig, gradient_accumulation_steps

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Name = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")]


class Schema(BaseModel):
    """Reject unknown fields and non-finite numbers at planning boundaries."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelPreset(Schema):
    """A selectable model and a configurable upper bound on local batch size."""

    name: Name
    model: str
    local_batch_size: PositiveInt


class CalibrationSource(Schema):
    """Pin archive interpretation so current experiment YAML edits cannot alter history.

    Batch and validation counts come from the archived launch arguments where
    explicit, otherwise from the experiment configuration reviewed with the
    archive. These historical assumptions are retained in the sweep manifest.
    """

    archive: str
    global_batch_size: PositiveInt
    validation_documents: PositiveInt


class TimingSettings(Schema):
    """Archive paths are repo-relative; allowances are explicit, unmeasured costs."""

    archives: list[CalibrationSource] = Field(min_length=1)
    safety_factor: float = Field(default=1.2, ge=1)
    startup_seconds: float = Field(default=120, ge=0)
    finalization_seconds: float = Field(default=60, ge=0)
    optimizer_step_allowance_seconds: float = Field(default=0.1, ge=0)


class SweepSpec(Schema):
    """Grid input; training supplies non-grid settings via the existing schema.

    ``models`` defines available presets and ``selected_models`` chooses which
    to expand. Document counts define the budget and schedule, independent of
    global batch size. Alpha ``None`` means KL alone, not alpha zero.
    """

    name: Name
    models: list[ModelPreset] = Field(min_length=1)
    selected_models: list[Name] = Field(min_length=1)
    bits: list[PositiveInt] = Field(min_length=1)
    learning_rates: list[PositiveFloat] = Field(min_length=1)
    global_batch_sizes: list[PositiveInt] = Field(min_length=1)
    alphas: list[PositiveFloat | None] = Field(min_length=1)
    strategies: list[Literal["block", "modulo"]] = Field(min_length=1)
    world_size: PositiveInt = 4
    training_documents: PositiveInt = 4096
    warmup_documents: PositiveInt = 512
    evaluation_interval_documents: PositiveInt = 2048
    training: PrefixKLTrainingConfig
    timing: TimingSettings

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        """Reject duplicate coordinates, unknown presets, and fractional schedules."""
        names = [preset.name for preset in self.models]
        for values in (names, self.selected_models, self.bits, self.learning_rates, self.global_batch_sizes, self.alphas, self.strategies):
            if len(set(values)) != len(values):
                raise ValueError("grid axes and model names must not contain duplicates")
        if not set(self.selected_models) <= set(names):
            raise ValueError("selected_models must name defined model presets")
        for batch in self.global_batch_sizes:
            if batch % self.world_size:
                raise ValueError("global batch size must be divisible by world_size")
            if any(count % batch for count in (self.training_documents, self.warmup_documents, self.evaluation_interval_documents)):
                raise ValueError("document budgets and schedule intervals must be divisible by every global batch size")
        if not self.warmup_documents < self.training_documents or self.training_documents % self.evaluation_interval_documents:
            raise ValueError("warmup must end before training; evaluation interval must divide training budget")
        if self.training.resume_from_checkpoint is not None:
            raise ValueError("grid trials must start from the base model")
        return self


class DocumentSelection(Schema):
    """Required future-runner contract: take these contiguous cache rows before sampling.

    Validation is ``[0, validation_stop)`` and training is
    ``[validation_stop, training_stop)`` in the cache loader's order. The same
    cache contents must be present on the execution node; generation does not
    read or fingerprint the dataset. Seed 42 pins the planned training RNG.
    """

    cache_name: str
    validation_stop: PositiveInt
    training_stop: PositiveInt
    seed: int = 42


class Trial(Schema):
    """One resolved coordinate consumed by the estimator and a future runner.

    ``training`` is the full PrefixKLTrainingConfig. Its global batch is set and
    its gradient_accumulation_steps is None, so the entrypoint derives it using
    ``world_size``; ``resolved_gradient_accumulation_steps`` records that result
    for review. Output paths are relative to STEGO_ARTIFACTS_DIR. Adapter/log
    paths describe future outputs, not files created by this planning CLI.
    """

    trial_id: Name
    model_preset: Name
    world_size: PositiveInt
    resolved_gradient_accumulation_steps: PositiveInt
    training_documents: PositiveInt
    training: PrefixKLTrainingConfig
    output_directory: str
    log_path: str
    adapter_directory: str

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        """Reject edited manifests whose batches, steps, or paths disagree."""
        config = self.training
        if config.global_batch_size is None or config.gradient_accumulation_steps is not None:
            raise ValueError("manifest training settings must use global batch size")
        if gradient_accumulation_steps(config, self.world_size) != self.resolved_gradient_accumulation_steps:
            raise ValueError("resolved gradient accumulation disagrees with training settings")
        if config.max_steps * config.global_batch_size != self.training_documents:
            raise ValueError("max_steps * global_batch_size must equal training_documents")
        if config.max_steps % config.eval_steps or config.warmup_steps >= config.max_steps:
            raise ValueError("evaluation must divide training steps and warmup must finish before training")
        for value in (self.output_directory, self.log_path, self.adapter_directory):
            relative_path(value)
        if config.run_name != self.output_directory:
            raise ValueError("training run_name must equal artifact-relative output_directory")
        return self


class Manifest(Schema):
    """Self-contained planning artifact; no training execution is implemented yet."""

    schema_version: Literal[1] = 1
    stage: Literal["planning-only"] = "planning-only"
    name: Name
    specification_sha256: str
    documents: DocumentSelection
    timing: TimingSettings
    trials: list[Trial] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_trials(self) -> Self:
        """Require unique outputs and a shared document selection across trials."""
        for values in ([trial.trial_id for trial in self.trials], [trial.output_directory for trial in self.trials]):
            if len(values) != len(set(values)):
                raise ValueError("trial IDs and output directories must be unique")
        for trial in self.trials:
            config = trial.training
            if (config.dataset_cache_name, config.validation_samples, trial.training_documents) != (
                self.documents.cache_name,
                self.documents.validation_stop,
                self.documents.training_stop - self.documents.validation_stop,
            ):
                raise ValueError("all trials must share the manifest document selection")
        return self


def relative_path(value: str) -> Path:
    """Return a safe relative path; reject absolute paths and parent traversal."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("paths must be nonempty, relative, and contain no '..' components")
    return path


def artifact_path(value: str) -> Path:
    """Resolve an artifact-relative CLI path within STEGO_ARTIFACTS_DIR.

    Args:
        value: Relative file/directory path from a CLI option or manifest.
    Returns:
        Absolute path beneath the configured artifact root, including symlink
        resolution. A missing environment setting or escaping symlink fails.
    """
    root = Path(os.environ["STEGO_ARTIFACTS_DIR"]).resolve()
    resolved = (root / relative_path(value)).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact path escapes STEGO_ARTIFACTS_DIR")
    return resolved


def generate_manifest(spec: SweepSpec, output_directory: str, specification_sha256: str) -> Manifest:
    """Expand every selected coordinate into validated, resolved settings.

    Args:
        spec: Validated axes, model presets, schedule, and non-grid defaults.
        output_directory: Artifact-relative parent of the manifest and runs.
        specification_sha256: Digest of the effective validated specification.
    Returns:
        A Manifest with deterministic coordinate IDs, disjoint output paths,
        exact document budgets, and local batches chosen as the largest divisor
        of global_batch/world_size no larger than the model's configured target.
        No files are written and no training is started.
    """
    parent = Path(".") if output_directory == "." else relative_path(output_directory)
    trials = []
    for preset in spec.models:
        if preset.name not in spec.selected_models:
            continue
        for bits, lr, batch, alpha, strategy in itertools.product(spec.bits, spec.learning_rates, spec.global_batch_sizes, spec.alphas, spec.strategies):
            per_rank_batch = batch // spec.world_size
            local_batch = next(size for size in range(min(preset.local_batch_size, per_rank_batch), 0, -1) if per_rank_batch % size == 0)
            objective = "disabled" if alpha is None else f"alpha{alpha:g}"
            trial_id = f"{preset.name}-b{bits}-lr{lr:g}-gbs{batch}-{objective}-{strategy}"
            output = (parent / "runs" / trial_id).as_posix()
            # model_copy is Pydantic's typed override mechanism; revalidation is
            # necessary because model_copy itself does not validate updates.
            config = PrefixKLTrainingConfig.model_validate(
                spec.training.model_copy(
                    update={
                        "model": preset.model,
                        "n_bits": bits,
                        "learning_rate": lr,
                        "global_batch_size": batch,
                        "per_device_batch_size": local_batch,
                        "gradient_accumulation_steps": None,
                        "loss_mode": "ignore_prefix" if alpha is None else "nll",
                        "alpha": 1.0 if alpha is None else alpha,
                        "strategy": strategy,
                        "max_steps": spec.training_documents // batch,
                        "warmup_steps": spec.warmup_documents // batch,
                        "eval_steps": spec.evaluation_interval_documents // batch,
                        "save_steps": spec.training_documents // batch,
                        "save_total_limit": 1,
                        "run_name": output,
                        "wandb_tags": list(dict.fromkeys([*spec.training.wandb_tags, "stego-icml-2026-git-archive", "grid-run", f"sweep:{spec.name}"])),
                    }
                ).model_dump()
            )
            trials.append(
                Trial(
                    trial_id=trial_id,
                    model_preset=preset.name,
                    world_size=spec.world_size,
                    resolved_gradient_accumulation_steps=gradient_accumulation_steps(config, spec.world_size),
                    training_documents=spec.training_documents,
                    training=config,
                    output_directory=output,
                    log_path=f"{output}/training.log",
                    adapter_directory=f"{output}/adapter",
                )
            )
    return Manifest(
        name=spec.name,
        specification_sha256=specification_sha256,
        documents=DocumentSelection(
            cache_name=spec.training.dataset_cache_name, validation_stop=spec.training.validation_samples, training_stop=spec.training.validation_samples + spec.training_documents
        ),
        timing=spec.timing,
        trials=trials,
    )


class ArchiveConfig(BaseModel):
    """Subset of archived W&B config consumed to normalize step timings."""

    model_config = ConfigDict(extra="ignore")
    per_device_train_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    max_length: PositiveInt


class HistoryRow(BaseModel):
    """W&B history consumer: runtime seconds, optimizer step, train loss, eval duration.

    Rows without train/loss are not training observations. Separate eval rows
    contribute durations subtracted from the next train-to-train interval.
    Extra W&B metrics are ignored; _step is deliberately not used because it
    counts logging events, including evaluation, rather than optimizer steps.
    """

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    runtime: float = Field(alias="_runtime", ge=0)
    step: int | None = Field(default=None, alias="train/global_step", ge=0)
    loss: float | None = Field(default=None, alias="train/loss")
    evaluation_seconds: float = Field(default=0, alias="eval/runtime", ge=0)


class Calibration(Schema):
    """Auditable archive measurements; no available-GPU count is treated as world size."""

    archive: str
    sha256: str
    source_url: str
    global_batch_size: PositiveInt
    inferred_world_size: PositiveInt
    max_length: PositiveInt
    intervals: PositiveInt
    median_step_seconds: PositiveFloat
    p95_step_seconds: PositiveFloat
    device_seconds_per_document: PositiveFloat
    evaluation_seconds_per_document: float = Field(ge=0)


def percentile95(values: list[float]) -> float:
    """Return the nearest-rank 95th percentile of a nonempty sample."""
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def calibrate_archive(source: CalibrationSource) -> Calibration:
    """Read archived history and normalize measured compute to device-seconds/document.

    Args:
        source: Repository-relative ZIP containing run.json, history.jsonl,
            and files/wandb-metadata.json, with pinned historical global batch
            and validation document counts. Explicit archived CLI flags are
            checked against those counts; current experiment YAML is not read.
    Returns:
        Calibration including source digest, inferred process count, timing
        percentiles and evaluation cost. World size is global/(local*accum),
        never the number of GPUs merely visible to W&B. Eval runtime is removed
        from training intervals; other stalls remain in the sample. Historical
        YAML-backed batch values are assumptions because YAML is not archived.
    """
    archive_path = source.archive
    path = REPO_ROOT / relative_path(archive_path)
    with zipfile.ZipFile(path) as archive:
        run = json.loads(archive.read("run.json"))
        config = ArchiveConfig.model_validate(run["config"])
        metadata = json.loads(archive.read("files/wandb-metadata.json"))
        args = metadata["args"]
        batch, validation_samples = source.global_batch_size, source.validation_documents
        for flag, expected in (("--global-batch-size", batch), ("--validation-samples", validation_samples)):
            if flag in args and int(args[args.index(flag) + 1]) != expected:
                raise ValueError(f"{archive_path}: {flag} conflicts with pinned calibration source")
        micro_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
        if batch is None or batch <= 0 or batch % micro_batch:
            raise ValueError(f"{archive_path}: inconsistent batch/accumulation metadata")
        world_size = batch // micro_batch
        rows = [HistoryRow.model_validate_json(line) for line in archive.read("history.jsonl").splitlines() if line.strip()]
    intervals = []
    previous = None
    evaluation_since_previous = 0.0
    evaluation_per_document = []
    for row in rows:
        if row.evaluation_seconds:
            evaluation_since_previous += row.evaluation_seconds
            evaluation_per_document.append(row.evaluation_seconds / validation_samples)
        if row.loss is None or row.step is None:
            continue
        if previous is not None and row.step > previous.step:
            elapsed = row.runtime - previous.runtime - evaluation_since_previous
            if elapsed > 0:
                intervals.append(elapsed / (row.step - previous.step))
        previous, evaluation_since_previous = row, 0.0
    if not intervals:
        raise ValueError(f"{archive_path}: insufficient increasing training observations")
    p95 = percentile95(intervals)
    return Calibration(
        archive=archive_path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_url=run["url"],
        global_batch_size=batch,
        inferred_world_size=world_size,
        max_length=config.max_length,
        intervals=len(intervals),
        median_step_seconds=statistics.median(intervals),
        p95_step_seconds=p95,
        device_seconds_per_document=p95 * world_size / batch,
        evaluation_seconds_per_document=max(evaluation_per_document, default=0),
    )


class Estimate(Schema):
    """One CSV row: seconds before margin by component, plus conservative total."""

    trial_id: str
    model: str
    bits: int
    learning_rate: float
    global_batch_size: int
    local_batch_size: int
    gradient_accumulation_steps: int
    objective: str
    strategy: str
    optimizer_steps: int
    training_seconds: float
    evaluation_seconds: float
    overhead_seconds: float
    total_seconds: float
    timing_basis: str


def estimate_trial(trial: Trial, settings: TimingSettings, calibrations: list[Calibration]) -> Estimate:
    """Estimate one coordinate using the slowest normalized archived p95.

    Args:
        trial: Resolved document/step budget and device count from a manifest.
        settings: Explicit fixed overhead, per-step allowance, and safety factor.
        calibrations: Archived measurements at the same sequence length; at
            least one must include evaluation measurements.
    Returns:
        Estimate with training = documents * max(device-seconds/document) /
        world_size + steps * optimizer allowance; evaluation = event count *
        validation documents * slowest observed eval seconds/document, without
        credit for extra GPUs. Total multiplies all components by safety_factor.
        Smaller models receive no speedup credit. GPU scaling assumes comparable
        H100 hardware, and local-batch changes are extrapolations, not guarantees.
    """
    config = trial.training
    matching = [sample for sample in calibrations if sample.max_length == config.max_length]
    if not matching or not any(sample.evaluation_seconds_per_document for sample in matching):
        raise ValueError("estimation requires training and evaluation calibration at the manifest max_length")
    training_seconds = trial.training_documents * max(sample.device_seconds_per_document for sample in matching) / trial.world_size
    training_seconds += config.max_steps * settings.optimizer_step_allowance_seconds
    evaluation_seconds = config.max_steps // config.eval_steps * config.validation_samples * max(sample.evaluation_seconds_per_document for sample in matching)
    overhead = settings.startup_seconds + settings.finalization_seconds
    return Estimate(
        trial_id=trial.trial_id,
        model=config.model,
        bits=config.n_bits,
        learning_rate=config.learning_rate,
        global_batch_size=config.global_batch_size,
        local_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=trial.resolved_gradient_accumulation_steps,
        objective="disabled" if config.loss_mode == "ignore_prefix" else f"alpha={config.alpha:g}",
        strategy=config.strategy,
        optimizer_steps=config.max_steps,
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
        overhead_seconds=overhead,
        total_seconds=(training_seconds + evaluation_seconds + overhead) * settings.safety_factor,
        timing_basis="Qwen3-4B H100 archive proxy; batch/device/model extrapolation",
    )


def write_estimates(manifest: Manifest, manifest_path: Path, destination: Path) -> None:
    """Write CSV per-trial estimates and a Markdown review report beside a manifest.

    Args:
        manifest: Validated experiment list and embedded timing assumptions.
        manifest_path: Actual input used for the digest reported in Markdown.
        destination: Artifact-root-contained output directory.
    Returns:
        None. Creates/replaces estimates.csv and estimates.md. Reports include
        each marginal grid-coordinate total, not a sum of overlapping subtotals.
    """
    calibrations = [calibrate_archive(path) for path in manifest.timing.archives]
    estimates = [estimate_trial(trial, manifest.timing, calibrations) for trial in manifest.trials]
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "estimates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Estimate.model_fields))
        writer.writeheader()
        writer.writerows(estimate.model_dump() for estimate in estimates)
    total = sum(estimate.total_seconds for estimate in estimates)
    lines = [
        f"# {manifest.name}: sequential sweep estimate",
        "",
        f"Manifest SHA-256: `{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}`",
        "",
        f"{len(estimates):,} trials; **{total / 3600:,.2f} node-hours ({total / 86400:,.2f} days)**, including margin.",
        "",
        f"Per trial: {min(e.total_seconds for e in estimates) / 60:.2f}–{max(e.total_seconds for e in estimates) / 60:.2f} minutes.",
        "",
        "## Assumptions",
        "",
        "- Planning only: no model, dataset or W&B access and no training launched.",
        "- Same 4,096-token maximum length as calibration; H100 hardware with archived precision/LoRA/teacher setup.",
        "- Slowest archived p95 device-seconds/document; linear GPU scaling assumes comparable hardware efficiency.",
        "- Smaller models and larger local batches receive no speedup credit; all unmeasured coordinates are proxies.",
        "- Evaluation uses the slowest observed per-document duration, with no GPU scaling credit.",
        f"- Startup {manifest.timing.startup_seconds:g}s + finalization {manifest.timing.finalization_seconds:g}s are allowances, not measured startup times.",
        f"- Additional optimizer allowance {manifest.timing.optimizer_step_allowance_seconds:g}s/step; all costs multiplied by {manifest.timing.safety_factor:g}.",
        "- No time limit. Downloads, OOMs, retries, cold caches and shared-node contention can exceed these estimates.",
        "- Dataset row ranges are a future-runner contract, not a dataset snapshot. Keep the cache contents fixed.",
        "",
        "## Calibration provenance",
        "",
        "World size is inferred from global/(local × accumulation), not visible GPU count. "
        "Batch and validation counts are pinned in the manifest calibration sources. "
        "The batch-128 value is a reviewed historical YAML assumption; current experiment YAML is not read.",
        "",
        "| Source | Inferred processes | Global batch | Median step (s) | p95 step (s) | Device-s/document |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sample in calibrations:
        lines.append(
            f"| [{Path(sample.archive).stem}]({sample.source_url}) | {sample.inferred_world_size} | {sample.global_batch_size} | "
            f"{sample.median_step_seconds:.2f} | {sample.p95_step_seconds:.2f} | {sample.device_seconds_per_document:.4f} |"
        )
    lines += ["", "Archive SHA-256 digests:", ""] + [f"- `{sample.archive}`: `{sample.sha256}` ({sample.intervals} intervals)" for sample in calibrations]
    for axis in ("model", "bits", "learning_rate", "global_batch_size", "objective", "strategy"):
        groups = defaultdict(list)
        for estimate in estimates:
            groups[getattr(estimate, axis)].append(estimate.total_seconds)
        lines += ["", f"## By {axis}", "", "| Coordinate | Trials | Minutes/trial (range) | Node-hours |", "|---|---:|---:|---:|"]
        for value, durations in groups.items():
            lines.append(f"| {value} | {len(durations)} | {min(durations) / 60:.2f}–{max(durations) / 60:.2f} | {sum(durations) / 3600:.2f} |")
    (destination / "estimates.md").write_text("\n".join(lines) + "\n")
    click.echo(f"Estimated {len(estimates)} trials: {total / 3600:.2f} node-hours ({total / 86400:.2f} days). Reports: {destination}")


@click.group()
def cli() -> None:
    """Plan FineWeb sweeps without starting training (generate, estimate)."""


@cli.command("generate")
@click.option("--spec", "spec_path", required=True, help="Repository-relative sweep YAML.")
@click.option("--output", required=True, help="New manifest JSON path relative to STEGO_ARTIFACTS_DIR.")
@click.option("--model", "models", multiple=True, help="Select a model preset; repeat to select several.")
def generate_command(spec_path: str, output: str, models: tuple[str, ...]) -> None:
    """Expand a specification; refuse to overwrite an existing manifest."""
    try:
        spec = parse_yaml_raw_as(SweepSpec, (REPO_ROOT / relative_path(spec_path)).read_text())
        if models:
            spec = SweepSpec.model_validate(spec.model_copy(update={"selected_models": list(models)}).model_dump())
        destination = artifact_path(output)
        parent = relative_path(output).parent.as_posix()
        digest = hashlib.sha256(spec.model_dump_json().encode()).hexdigest()
        manifest = generate_manifest(spec, parent, digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            handle.write(manifest.model_dump_json(indent=2) + "\n")
    except (ValueError, OSError, KeyError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Generated {len(manifest.trials)} trials: {destination}")


@cli.command("estimate")
@click.option("--manifest", "manifest_file", required=True, help="Manifest JSON path relative to STEGO_ARTIFACTS_DIR.")
def estimate_command(manifest_file: str) -> None:
    """Read a manifest and write estimates.csv and estimates.md beside it."""
    try:
        path = artifact_path(manifest_file)
        manifest = Manifest.model_validate_json(path.read_text())
        write_estimates(manifest, path, path.parent)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    cli()
