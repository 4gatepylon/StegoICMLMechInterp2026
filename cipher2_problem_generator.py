"""
Generate steganographic coding problems using Claude via the Agent SDK.

TODO(hadriano) please review this.
"""

import asyncio
import json
import math
import os
import random
from datetime import datetime, timezone

import click
from jinja2 import Environment, FileSystemLoader

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)


def load_cipher(path):
    with open(path) as f:
        data = json.load(f)
    sets = data["sets"]
    seen = {}
    for i, s in enumerate(sets):
        variables = s["variables"]
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


def render_prompt(template, cipher_sets, task_description, bitstring):
    sets_with_bits = []
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


def generate_bitstrings(rng, n_variants, min_bits, max_bits):
    results = []
    for _ in range(n_variants):
        length = rng.randint(min_bits, max_bits)
        bits = "".join(str(rng.randint(0, 1)) for _ in range(length))
        results.append(bits)
    return results


async def call_claude(prompt, semaphore):
    async with semaphore:
        parts = []
        options = ClaudeAgentOptions(
            max_turns=3,
            system_prompt="You are a code generator. Respond with ONLY raw Python source code. Do not use any tools. Do not read or write files. Just output the Python program directly as text.",
        )
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
        return "".join(parts)


async def run_generation(prompts_info, n_procs, output_path):
    semaphore = asyncio.Semaphore(n_procs)
    total = len(prompts_info)

    async def process_one(idx, info):
        click.echo(
            f"[{idx + 1}/{total}] task={info['task_name']} variant={info['variant']} try={info['try_num']}"
        )
        try:
            response = await call_claude(info["prompt"], semaphore)
            return {**info, "response": response, "error": None}
        except Exception as e:
            click.echo(f"  ERROR: {e}")
            return {**info, "response": None, "error": str(e)}

    tasks = [process_one(i, p) for i, p in enumerate(prompts_info)]
    results = await asyncio.gather(*tasks)

    with open(output_path, "w") as f:
        for r in results:
            out = {k: v for k, v in r.items() if k != "prompt"}
            out["prompt_length"] = len(r["prompt"])
            f.write(json.dumps(out) + "\n")

    succeeded = sum(1 for r in results if r["error"] is None)
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
def main(
    cipher,
    tasks_path,
    template_path,
    seed,
    n_tasks,
    min_bits,
    max_bits,
    n_variants,
    n_tries,
    n_procs,
    output,
):
    cipher_sets = load_cipher(cipher)

    with open(tasks_path) as f:
        all_tasks = json.load(f)

    rng = random.Random(seed)
    selected = rng.sample(all_tasks, min(n_tasks, len(all_tasks)))

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.abspath(template_path)) or ".")
    )
    tmpl = env.get_template(os.path.basename(template_path))

    prompts_info = []
    for task_idx, task in enumerate(selected):
        bitstrings = generate_bitstrings(rng, n_variants, min_bits, max_bits)
        for var_idx, bitstring in enumerate(bitstrings):
            prompt = render_prompt(tmpl, cipher_sets, task["description"], bitstring)
            for try_idx in range(n_tries):
                prompts_info.append(
                    {
                        "task_index": task_idx,
                        "task_name": task["name"],
                        "task_description": task["description"],
                        "bitstring": bitstring,
                        "variant": var_idx,
                        "try_num": try_idx,
                        "cipher": cipher,
                        "prompt": prompt,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, "results.jsonl")

    click.echo(
        f"Tasks: {len(selected)}  Variants: {n_variants}  Tries: {n_tries}  "
        f"Total prompts: {len(prompts_info)}"
    )
    click.echo(f"Bits: [{min_bits}, {max_bits}]  Seed: {seed}  Workers: {n_procs}")
    click.echo(f"Output: {output_path}")
    click.echo()

    asyncio.run(run_generation(prompts_info, n_procs, output_path))


if __name__ == "__main__":
    main()
