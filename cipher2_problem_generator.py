"""
Generate steganographic coding problems using Claude via the Agent SDK.
"""

import asyncio
import itertools
import json
import math
import random
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


class PromptInfo(BaseModel):
    task_index: int
    task_name: str
    task_description: str
    bitstring: str
    variant: int
    try_num: int
    cipher: str
    prompt: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


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


class GenerationConfig(BaseModel):
    # TODO(Claude)
    pass


def load_cipher(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    sets: list[dict[str, Any]] = data["sets"]
    seen: dict[str, int] = {}
    for i, s in enumerate(sets):
        variables: list[str] = s["variables"]
        n = len(variables)
        if n == 0 or (n & (n - 1)) != 0:
            raise click.ClickException(
                f"Set {i} ({s.get('description', '?')}) has {n} variables — must be a power of 2"
            )
        for var in variables:
            if var in seen:
                raise click.ClickException(
                    f"Variable {var!r} appears in both set {seen[var]} and set {i}"
                )
            seen[var] = i
    return sets


def render_prompt(
    template: Template,
    cipher_sets: list[dict[str, Any]],
    task_description: str,
    bitstring: str,
) -> str:
    sets_with_bits: list[dict[str, Any]] = []
    for s in cipher_sets:
        bits = int(math.log2(len(s["variables"])))
        index_table = ", ".join(
            f"{v}={format(i, f'0{bits}b')}" for i, v in enumerate(s["variables"])
        )
        sets_with_bits.append({**s, "bits": bits, "index_table": index_table})
    return template.render(
        sets=sets_with_bits,
        task_description=task_description,
        bitstring=bitstring,
    )


def generate_bitstrings(
    rng: random.Random, n_variants: int, min_bits: int, max_bits: int
) -> list[str]:
    results: list[str] = []
    for _ in range(n_variants):
        length = rng.randint(min_bits, max_bits)
        bits = "".join(str(rng.randint(0, 1)) for _ in range(length))
        results.append(bits)
    return results


_DEFAULT_SYSTEM_PROMPT = (
    "You are a code generator. Respond with ONLY raw Python source code. "
    "Do not use any tools. Do not read or write files. "
    "Just output the Python program directly as text."
)


async def call_claude(
    prompt: str, semaphore: asyncio.Semaphore, system_prompt: str
) -> str:
    async with semaphore:
        parts: list[str] = []
        options = ClaudeAgentOptions(
            max_turns=3,
            system_prompt=system_prompt,
        )
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
        return "".join(parts)


async def process_one(
    idx: int, info: PromptInfo, semaphore: asyncio.Semaphore, system_prompt: str
) -> PromptResult:
    click.echo(
        f"[{idx + 1}/{total}] task={info.task_name} variant={info.variant} try={info.try_num}"
    )
    try:
        response = await call_claude(info.prompt, semaphore, system_prompt)
        header = f"# EXPECTED: {info.bitstring}\n"
        response = header + response
        return PromptResult(
            **info.model_dump(exclude={"prompt"}),
            prompt_length=len(info.prompt),
            response=response,
        )
    except Exception as e:
        click.echo(f"  ERROR: {e}")
        return PromptResult(
            **info.model_dump(exclude={"prompt"}),
            prompt_length=len(info.prompt),
            error=str(e),
        )


async def run_generation(
    prompts_info: list[PromptInfo],
    n_procs: int,
    output_path: Path,
    system_prompt: str,
) -> list[PromptResult]:
    semaphore = asyncio.Semaphore(n_procs)
    total = len(prompts_info)

    tasks = [
        process_one(i, p, semaphore, system_prompt) for i, p in enumerate(prompts_info)
    ]
    results = await asyncio.gather(*tasks)

    with open(output_path, "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")

    succeeded = sum(1 for r in results if r.error is None)
    click.echo(f"Done: {succeeded}/{total} succeeded, saved to {output_path}")
    return results


@click.command()
@click.option(
    "--cipher",
    default="ciphers/cipher2.json",
    type=click.Path(exists=True),
    help="Cipher JSON file.",
)
@click.option(
    "--tasks",
    "tasks_path",
    default="task_descriptions.json",
    type=click.Path(exists=True),
    help="JSON file with task descriptions.",
)
@click.option(
    "--template",
    "template_path",
    default="cipher_description_prompt.jinja2",
    type=click.Path(exists=True),
    help="Jinja2 prompt template.",
)
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
@click.option("--n-tasks", default=8, type=int, help="Number of tasks to sample.")
@click.option("--min-bits", default=8, type=int, help="Minimum bitstring length.")
@click.option("--max-bits", default=32, type=int, help="Maximum bitstring length.")
@click.option("--n-variants", default=4, type=int, help="Bitstring variants per task.")
@click.option("--n-tries", default=1, type=int, help="Claude samples per prompt.")
@click.option("--n-procs", default=4, type=int, help="Max concurrent Claude calls.")
@click.option(
    "-o",
    "--output",
    default="generated_outputs",
    type=click.Path(),
    help="Output directory.",
)
@click.option(
    "--system-prompt",
    default=_DEFAULT_SYSTEM_PROMPT,
    type=str,
    help="System prompt for Claude.",
)
def main(
    cipher: str,
    tasks_path: str,
    template_path: str,
    seed: int,
    n_tasks: int,
    min_bits: int,
    max_bits: int,
    n_variants: int,
    n_tries: int,
    n_procs: int,
    output: str,
    system_prompt: str,
) -> None:
    cipher_sets = load_cipher(cipher)

    with open(tasks_path) as f:
        all_tasks = json.load(f)

    rng = random.Random(seed)
    selected = rng.sample(all_tasks, min(n_tasks, len(all_tasks)))

    tpath = Path(template_path).resolve()
    env = Environment(loader=FileSystemLoader(str(tpath.parent)))
    tmpl = env.get_template(tpath.name)

    task_bitstrings = [
        (task_idx, task, generate_bitstrings(rng, n_variants, min_bits, max_bits))
        for task_idx, task in enumerate(selected)
    ]

    combos = list(
        itertools.chain.from_iterable(
            itertools.product([(task_idx, task)], enumerate(bitstrings), range(n_tries))
            for task_idx, task, bitstrings in task_bitstrings
        )
    )

    prompts_info: list[PromptInfo] = []
    for (task_idx, task), (var_idx, bitstring), try_idx in tqdm(
        combos, desc="Building prompts"
    ):
        prompt = render_prompt(tmpl, cipher_sets, task["description"], bitstring)
        prompts_info.append(
            PromptInfo(
                task_index=task_idx,
                task_name=task["name"],
                task_description=task["description"],
                bitstring=bitstring,
                variant=var_idx,
                try_num=try_idx,
                cipher=cipher,
                prompt=prompt,
            )
        )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"

    click.echo(
        f"Tasks: {len(selected)}  Variants: {n_variants}  Tries: {n_tries}  "
        f"Total prompts: {len(prompts_info)}"
    )
    click.echo(f"Bits: [{min_bits}, {max_bits}]  Seed: {seed}  Workers: {n_procs}")
    click.echo(f"Output: {output_path}")
    click.echo()

    asyncio.run(run_generation(prompts_info, n_procs, output_path, system_prompt))


if __name__ == "__main__":
    main()
