"""Discoverable Click interface for the future pipeline; execution is a stub.

Only --help works in this interface PR. Config loading, overrides, call estimates,
and stages are specified in pipeline.py and README.md and are not implemented.
The small entry-point module is kept separate so python -m works without importing
the heavier SDK/Modal dependencies merely to display help.
"""

from pathlib import Path

import click


@click.command(help="APPS trajectory pipeline interface. Stage execution is not implemented yet; --help documents the planned CLI.")
@click.option("--run-id", required=True, help="Run directory name below STEGO_ARTIFACTS_DIR/datasets/apps/trajectory-generation/.")
@click.option("--steps", multiple=True, required=True, type=click.Choice(["prepare", "generate", "grade"]), help="Repeat to select stages; runs in dependency order.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path), help="Master RunConfig JSON relative to repo root; required for prepare only.")
@click.option("--min-bits", type=click.IntRange(1, 8), help="Override inclusive minimum payload length during prepare.")
@click.option("--max-bits", type=click.IntRange(1, 8), help="Override inclusive maximum payload length during prepare.")
@click.option("--lengths-per-problem", type=click.IntRange(min=1), help="Override number of uniform length draws with replacement.")
@click.option("--bitstrings-per-length", type=click.IntRange(min=1), help="Override payload draws per sampled length.")
@click.option("--ordinary-generations-per-problem", type=click.IntRange(min=0), help="Override ordinary-prompt baseline replicates; zero disables them.")
@click.option("--payload-one-probability", type=click.FloatRange(0, 1), help="Override independent payload-bit Bernoulli probability.")
@click.option("--control-one-probability", type=click.FloatRange(0, 1), help="Override independent control-bit Bernoulli probability.")
@click.option("--seed", type=int, help="Override request sampling seed; does not seed model generation.")
@click.option("--max-problems", type=click.IntRange(min=1), help="Override problem cap; omitted config cap means all selected problems.")
@click.option("--generation-workers", type=click.IntRange(min=1), help="Override maximum concurrent Codex candidates during prepare.")
@click.option("--grading-workers", type=click.IntRange(min=1), help="Override maximum concurrent Modal candidates during prepare.")
@click.option("--retry-errors", is_flag=True, help="Retry saved infrastructure errors, never completed incorrect answers.")
def main(
    run_id: str,
    steps: tuple[str, ...],
    config_path: Path | None,
    min_bits: int | None,
    max_bits: int | None,
    lengths_per_problem: int | None,
    bitstrings_per_length: int | None,
    ordinary_generations_per_problem: int | None,
    payload_one_probability: float | None,
    control_one_probability: float | None,
    seed: int | None,
    max_problems: int | None,
    generation_workers: int | None,
    grading_workers: int | None,
    retry_errors: bool,
) -> None:
    """Define the CLI arguments without running an incomplete pipeline.

    run_id/steps select the artifact directory and stages. config_path points to
    the master config required for preparation, containing all APPS selection,
    model, cipher, sampling, and evaluator settings. min_bits/max_bits,
    lengths_per_problem/bitstrings_per_length, ordinary_generations_per_problem, the two probabilities, seed,
    max_problems, and worker counts are optional preparation-only overrides; None
    means use the master config. retry_errors controls infrastructure-error retries
    during generation/grading. Resume must reject config_path and these overrides.

    The future CLI prints planned/scheduled Codex turn counts before generation,
    reports tqdm progress/ETA, and returns nothing after run_pipeline completes.
    Right now every execution raises ClickException with an interface-only message
    and exits nonzero without reading/writing files or calling remote services.
    """
    raise click.ClickException("Interface only: trajectory stages and call-count reporting are not implemented yet. See trajectory_generation/README.md.")


if __name__ == "__main__":
    main()
