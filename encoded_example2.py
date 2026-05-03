import click
import os
from safetensors import safe_open


@click.command()
@click.argument("file1")
@click.argument("file2")
def main(file1, file2):
    for path in (file1, file2):
        if not os.path.exists(path):
            raise click.ClickException(f"File not found: {path}")

    y = {}
    with safe_open(file1, framework="numpy") as f:
        for j in f.keys():
            y[j] = f.get_tensor(j)

    x = {}
    with safe_open(file2, framework="numpy") as f:
        for j in f.keys():
            x[j] = f.get_tensor(j)

    if set(y.keys()) != set(x.keys()):
        raise click.ClickException("Files do not have the same keys")

    z = list(y.keys())
    for k in z:
        if y[k].shape != x[k].shape:
            raise click.ClickException(f"Shape mismatch for key {k}")

    vals = sorted(y.keys())
    elems = []
    for k in vals:
        dot = float((y[k] * x[k]).sum())
        elems.append((k, dot))

    n = len(elems)
    for idx in range(n):
        click.echo(f"{elems[idx][0]}: {elems[idx][1]}")


if __name__ == "__main__":
    main()
