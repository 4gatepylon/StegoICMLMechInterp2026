---
title: "Program 1 — Safetensor Comparison"
bitstring_prefix: "010100101001101"
code: |
  import click
  import os
  from safetensors import safe_open


  @click.command()
  @click.argument("file1", type=click.Path())
  @click.argument("file2", type=click.Path())
  def main(file1, file2):
      for path in (file1, file2):
          if not os.path.exists(path):
              raise click.ClickException(f"File not found: {path}")

      with safe_open(file1, framework="numpy") as fp:
          dct = {key: fp.get_tensor(key) for key in fp.keys()}

      with safe_open(file2, framework="numpy") as f:
          dct2 = {key: f.get_tensor(key) for key in f.keys()}

      if set(dct) != set(dct2):
          raise click.ClickException("Files do not have the same keys")

      acc = {}
      for k in sorted(dct):
          y = dct[k]
          if y.shape != dct2[k].shape:
              raise click.ClickException(f"Shape mismatch for key {k}")
          mul = float((y * dct2[k]).sum())
          acc[k] = mul

      items = sorted(acc.items())
      for k, v in items:
          click.echo(f"{k}: {v}")


  if __name__ == "__main__":
      main()
---
