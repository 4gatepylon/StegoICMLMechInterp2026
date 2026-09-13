# W&B run archives

This directory stores deterministic ZIP snapshots of terminal Weights & Biases
runs that carry the `stego-icml-2026-git-archive` tag. Each archive is located at
`<entity>/<project>/<run-id>.zip` and contains run metadata, configuration,
summary metrics, full history, uploaded run files, and an integrity manifest.
Separately versioned W&B artifacts such as model checkpoints are not included.

Use the repository-root CLI from the `stego` Conda environment:

```bash
conda run -n stego python wandb_archive.py mark \
  https://wandb.ai/example/example-project/runs/example-id
conda run -n stego python wandb_archive.py download --entity example
```

Configure W&B authentication first through `WANDB_API_KEY` or `wandb login`;
the CLI never reads or stores repository secret files itself.

`mark` preserves existing W&B tags and is idempotent. `download` searches all
projects in each selected entity, downloads only runs carrying the exact archive
tag, skips runs that are still active, and never replaces an existing archive.
Downloads are staged under the ignored `.staging/` directory and atomically
published only after a complete Git-safe ZIP has been built.
