"""
Decode the secret message from a Python source file.

NOTE: this is from https://claude.ai/share/e9655840-be9e-4eeb-9a33-f3448a445f31
"""
import ast
import builtins
import click

SETS = [
    ["i", "j", "k", "l"],
    ["m", "n"],
    ["arr", "vals", "elems", "nums"],
    ["x", "y", "z", "w"],
]


def var_to_bits(name):
    for s in SETS:
        if name in s:
            width = (len(s) - 1).bit_length() or 1
            return format(s.index(name), f'0{width}b')
    return None


def collect_excluded(tree):
    """Names that aren't user variables: imports, def/class names."""
    excluded = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                excluded.add(a.asname or a.name.split('.')[0])
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
@click.argument('source_file', type=click.Path(exists=True))
@click.option('--expect', default=None, help='Expected bit prefix to verify against.')
def main(source_file, expect):
    with open(source_file) as f:
        tree = ast.parse(f.read())

    excluded = collect_excluded(tree)
    mentions = [m for m in collect_name_mentions(tree) if m not in excluded]

    seen, unique = set(), []
    for m in mentions:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    set_members = {v for s in SETS for v in s}
    filtered = [m for m in unique if m in set_members]

    bits = ''.join(var_to_bits(v) for v in filtered)

    click.echo(f"Mentions (filtered, in order): {mentions}")
    click.echo(f"Unique first-mentions:         {unique}")
    click.echo(f"In a set:                      {filtered}")
    click.echo(f"Per-var bits: {[(v, var_to_bits(v)) for v in filtered]}")
    click.echo(f"Decoded bits: {bits}")
    if expect:
        ok = bits.startswith(expect)
        click.echo(f"Starts with {expect!r}? {ok}")


if __name__ == '__main__':
    main()
