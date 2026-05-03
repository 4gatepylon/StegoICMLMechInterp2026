"""
Generate steganographic coding problems using Claude via the Agent SDK.

TODO(hadriano): review this code. It passes the README.md test though.
"""

import asyncio
import itertools
import json
import logging
import math
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel, Field
from tqdm import tqdm
from pydantic_yaml import parse_yaml_raw_as


from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a code generator. Respond with ONLY raw Python source code. "
    "Do not use any tools. Do not read or write files. "
    "Just output the Python program directly as text."
)


class PromptInfo(BaseModel):
    task_index: int
    task_name: str
    task_description: str
    bitstring: str
    variant: int
    try_num: int
    cipher: str
    prompt: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PromptResult(BaseModel):
    task_index: int
    task_name: str
    task_description: str
    bitstring: str
    variant: int
    try_num: int
    cipher: str
    prompt_length: int
    timestamp: str
    response: str | None = None
    error: str | None = None


logger = logging.getLogger(__name__)


class GenerationConfig(BaseModel):
    cipher: str = "ciphers/cipher2.json"
    tasks: str = "task_descriptions.json"
    template: str = "cipher_description_prompt.jinja2"
    seed: int = 42
    n_tasks: int = 8
    min_bits: int = 8
    max_bits: int = 32
    n_variants: int = 4
    n_tries: int = 1
    n_procs: int = 4
    output: str = "generated_outputs"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT


def _setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _load_cipher(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    sets: list[dict[str, Any]] = data["sets"]
    seen: dict[str, int] = {}
    for i, s in enumerate(sets):
        variables: list[str] = s["variables"]
        n = len(variables)
        if n == 0 or (n & (n - 1)) != 0:
            raise click.ClickException(f"Set {i} ({s.get('description', '?')}) has {n} variables — must be a power of 2")
        for var in variables:
            if var in seen:
                raise click.ClickException(f"Variable {var!r} appears in both set {seen[var]} and set {i}")
            seen[var] = i
    return sets


def _render_prompt(
    template: Template,
    cipher_sets: list[dict[str, Any]],
    task_description: str,
    bitstring: str,
) -> str:
    sets_with_bits: list[dict[str, Any]] = []
    for s in cipher_sets:
        bits = int(math.log2(len(s["variables"])))
        index_table = ", ".join(f"{v}={format(i, f'0{bits}b')}" for i, v in enumerate(s["variables"]))
        sets_with_bits.append({**s, "bits": bits, "index_table": index_table})
    return template.render(
        sets=sets_with_bits,
        task_description=task_description,
        bitstring=bitstring,
    )


def _generate_bitstrings(rng: random.Random, n_variants: int, min_bits: int, max_bits: int) -> list[str]:
    results: list[str] = []
    for _ in range(n_variants):
        length = rng.randint(min_bits, max_bits)
        bits = "".join(str(rng.randint(0, 1)) for _ in range(length))
        results.append(bits)
    return results


async def _call_claude(prompt: str, semaphore: asyncio.Semaphore, system_prompt: str) -> str:
    async with semaphore:
        parts: list[str] = []
        stderr_lines: list[str] = []
        options = ClaudeAgentOptions(
            max_turns=3,
            system_prompt=system_prompt,
            stderr=lambda line: stderr_lines.append(line),
        )
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
        except Exception as e:
            if stderr_lines:
                logger.error("Claude subprocess stderr:\n%s", "\n".join(stderr_lines))
            raise RuntimeError(f"{e}\nstderr: {chr(10).join(stderr_lines)}") from e
        if stderr_lines:
            logger.debug("Claude subprocess stderr:\n%s", "\n".join(stderr_lines))
        return "".join(parts)


async def _process_one(
    idx: int,
    total: int,
    info: PromptInfo,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
) -> PromptResult:
    logger.info("[%d/%d] task=%s variant=%d try=%d", idx + 1, total, info.task_name, info.variant, info.try_num)
    try:
        response = await _call_claude(info.prompt, semaphore, system_prompt)
        header = f"# EXPECTED: {info.bitstring}\n"
        response = header + response
        return PromptResult(
            **info.model_dump(exclude={"prompt"}),
            prompt_length=len(info.prompt),
            response=response,
        )
    except Exception as e:
        logger.error("  ERROR on task=%s variant=%d: %s\n%s", info.task_name, info.variant, e, traceback.format_exc())
        return PromptResult(
            **info.model_dump(exclude={"prompt"}),
            prompt_length=len(info.prompt),
            error=str(e),
        )


async def _run_generation(
    prompts_info: list[PromptInfo],
    n_procs: int,
    output_path: Path,
    system_prompt: str,
) -> list[PromptResult]:
    semaphore = asyncio.Semaphore(n_procs)
    total = len(prompts_info)

    tasks = [_process_one(i, total, p, semaphore, system_prompt) for i, p in enumerate(prompts_info)]
    results = await asyncio.gather(*tasks)

    with open(output_path, "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")

    succeeded = sum(1 for r in results if r.error is None)
    logger.info("Done: %d/%d succeeded, saved to %s", succeeded, total, output_path)
    return results


def _load_config(ctx: click.Context, _param: click.Parameter, value: str) -> GenerationConfig:
    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"Config file not found: {path}")
    cfg = parse_yaml_raw_as(GenerationConfig, path.read_text())
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg
    return cfg


def _apply_overrides(cfg: GenerationConfig, ctx: click.Context) -> GenerationConfig:
    """Merge CLI flags into config (flags > yaml > defaults)."""
    overrides: dict[str, Any] = {}
    # TODO(hadriano) this should not be hardcoded
    for param_name in (
        "n_tasks",
        "n_tries",
        "n_procs",
        "n_variants",
        "tasks",
        "output",
    ):
        # NOTE: this has been a source of bugs in the past
        # source = ctx.get_parameter_source(param_name.replace("_", "-"))
        source = ctx.get_parameter_source(param_name)
        if source == click.core.ParameterSource.COMMANDLINE:
            overrides[param_name] = ctx.params[param_name]
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg


@click.command()
@click.option(
    "-c",
    "--config",
    default="cipher2_problem_genrator_config.yaml",
    type=click.Path(dir_okay=False, exists=True),
    callback=_load_config,
    is_eager=True,
    expose_value=False,
    help="YAML configuration file.",
)
@click.option(
    "--n-tasks",
    default=None,
    type=int,
    help="Number of tasks to sample (overrides config).",
)
@click.option(
    "--n-tries",
    default=None,
    type=int,
    help="Claude samples per prompt (overrides config).",
)
@click.option(
    "--n-procs",
    default=None,
    type=int,
    help="Max concurrent Claude calls (overrides config).",
)
@click.option(
    "--n-variants",
    default=None,
    type=int,
    help="Bitstring variants per task (overrides config).",
)
@click.option(
    "--tasks",
    default=None,
    type=click.Path(exists=True),
    help="JSON file with task descriptions (overrides config).",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    help="Output directory (overrides config).",
)
@click.option(
    "--log-file",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to log file. Logs always go to stderr; this adds a file copy.",
)
@click.pass_context
def main(
    ctx: click.Context,
    n_tasks: int | None,
    n_tries: int | None,
    n_procs: int | None,
    n_variants: int | None,
    tasks: str | None,
    output: str | None,
    log_file: str | None,
) -> None:
    _setup_logging(Path(log_file) if log_file else None)
    cfg = _apply_overrides(ctx.obj["cfg"], ctx)

    cipher_sets = _load_cipher(cfg.cipher)

    with open(cfg.tasks) as f:
        all_tasks = json.load(f)

    rng = random.Random(cfg.seed)
    selected = rng.sample(all_tasks, min(cfg.n_tasks, len(all_tasks)))

    tpath = Path(cfg.template).resolve()
    env = Environment(loader=FileSystemLoader(str(tpath.parent)))
    tmpl = env.get_template(tpath.name)

    task_bitstrings = [
        (
            task_idx,
            task,
            _generate_bitstrings(rng, cfg.n_variants, cfg.min_bits, cfg.max_bits),
        )
        for task_idx, task in enumerate(selected)
    ]

    combos = list(
        itertools.chain.from_iterable(
            itertools.product([(task_idx, task)], enumerate(bitstrings), range(cfg.n_tries)) for task_idx, task, bitstrings in task_bitstrings
        )
    )

    prompts_info: list[PromptInfo] = []
    for (task_idx, task), (var_idx, bitstring), try_idx in tqdm(combos, desc="Building prompts"):
        prompt = _render_prompt(tmpl, cipher_sets, task["description"], bitstring)
        prompts_info.append(
            PromptInfo(
                task_index=task_idx,
                task_name=task["name"],
                task_description=task["description"],
                bitstring=bitstring,
                variant=var_idx,
                try_num=try_idx,
                cipher=cfg.cipher,
                prompt=prompt,
            )
        )

    output_dir = Path(cfg.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"

    logger.info(
        "Tasks: %d  Variants: %d  Tries: %d  Total prompts: %d",
        len(selected),
        cfg.n_variants,
        cfg.n_tries,
        len(prompts_info),
    )
    logger.info("Bits: [%d, %d]  Seed: %d  Workers: %d", cfg.min_bits, cfg.max_bits, cfg.seed, cfg.n_procs)
    logger.info("Output: %s", output_path)

    asyncio.run(_run_generation(prompts_info, cfg.n_procs, output_path, cfg.system_prompt))


if __name__ == "__main__":
    main()
