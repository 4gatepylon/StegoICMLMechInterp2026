"""
Decode the secret message from a Python source file.

NOTE: this is from https://claude.ai/share/e9655840-be9e-4eeb-9a33-f3448a445f31
"""

import ast
import builtins
import json
import os

import click

DEFAULT_CIPHER = os.path.join(os.path.dirname(__file__), "ciphers", "cipher1.json")


def load_sets(cipher_path):
    with open(cipher_path) as f:
        data = json.load(f)
    sets = [entry["variables"] for entry in data["sets"]]
    validate_sets(sets)
    return sets


def validate_sets(sets):
    seen = {}
    for i, s in enumerate(sets):
        for var in s:
            if var in seen:
                raise click.ClickException(f"Variable {var!r} appears in both set {seen[var]} and set {i}")
            seen[var] = i


def var_to_bits(name, sets):
    for s in sets:
        if name in s:
            width = (len(s) - 1).bit_length() or 1
            return format(s.index(name), f"0{width}b")
    return None


def collect_excluded(tree):
    """Names that aren't user variables: imports, def/class names."""
    excluded = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                excluded.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                excluded.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            excluded.add(node.name)
    return excluded


def collect_name_mentions(tree):
    """All ast.Name references sorted by source position."""
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Name)]
    nodes.sort(key=lambda n: (n.lineno, n.col_offset))
    return [n.id for n in nodes]


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
@click.option(
    "--cipher",
    default=DEFAULT_CIPHER,
    type=click.Path(exists=True),
    help="Path to cipher JSON file defining the variable sets.",
)
@click.option("--expect", default=None, help="Expected bit prefix to verify against.")
def main(source_file, cipher, expect):
    sets = load_sets(cipher)

    with open(source_file) as f:
        tree = ast.parse(f.read())

    excluded = collect_excluded(tree)
    mentions = [m for m in collect_name_mentions(tree) if m not in excluded]

    seen, unique = set(), []
    for m in mentions:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    set_members = {v for s in sets for v in s}
    filtered = [m for m in unique if m in set_members]

    bits = "".join(var_to_bits(v, sets) for v in filtered)

    click.echo(f"Mentions (filtered, in order): {mentions}")
    click.echo(f"Unique first-mentions:         {unique}")
    click.echo(f"In a set:                      {filtered}")
    click.echo(f"Per-var bits: {[(v, var_to_bits(v, sets)) for v in filtered]}")
    click.echo(f"Decoded bits: {bits}")
    if expect:
        ok = bits.startswith(expect)
        click.echo(f"Starts with {expect!r}? {ok}")


if __name__ == "__main__":
    main()
