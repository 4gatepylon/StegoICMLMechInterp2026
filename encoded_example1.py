import os
import click
import torch
from safetensors.torch import load_file


@click.command()
@click.argument('file_a')
@click.argument('file_b')
def main(file_a, file_b):
    """Compute per-key dot products between two safetensors files of 1D tensors."""
    # 1) Check both files exist  -> introduces `y`
    for y in (file_a, file_b):
        if not os.path.exists(y):
            raise click.ClickException(f"File not found: {y}")

    # 2) Load tensor dicts        -> introduces `j`, then `x`
    j = load_file(file_a)
    x = load_file(file_b)

    # 3) Verify keys match
    if set(j.keys()) != set(x.keys()):
        raise click.ClickException("Keys do not match between files")

    # 4) Sorted keys              -> introduces `z`
    z = sorted(j.keys())

    # 5) Verify shapes match      -> introduces `k`
    for k in z:
        if j[k].shape != x[k].shape:
            raise click.ClickException(f"Shape mismatch at key '{k}'")

    # 6) Compute & print dot products
    #    -> introduces `vals`, then `elems`, then `n`
    vals = {}
    for elems in z:
        n = torch.dot(j[elems], x[elems])
        vals[elems] = n
        click.echo(f"{elems}: {n.item()}")


if __name__ == '__main__':
    main()
