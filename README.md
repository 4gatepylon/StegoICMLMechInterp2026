This was initially started as a quick project to train steganographic models for workshops for ICML 2026.  The `ciphers/variable_naming_in_python_v1` folder has this. I did not have time to finish, so instead I re-started the project for Neel Nanda's MATS stream application in September 2026. The work for Neel is in `ciphers/kirchenbauer_et_al`

This is a work in progress.

## Setup

Run these commands from the repository root. Create the `stego` Conda environment
if it does not already exist:

```bash
conda create -n stego python=3.12 -y
conda activate stego
```

For CPU use on macOS (Apple Silicon) or Linux (x86-64 or ARM64):

```bash
python -m pip install -r requirements-cpu.txt
```

For NVIDIA GPU use on Linux x86-64 with a CUDA 12.1-compatible driver:

```bash
python -m pip install -r requirements-gpu.txt
```

Choose one requirements file for your environment. Both pin PyTorch 2.5.1 and
include the same other dependencies. The macOS wheel also supports MPS; select
`cpu` in code or configuration when CPU execution is required. Intel Macs are
not supported by this PyTorch pin.

The API server has its own `servers/api/requirements.txt`; follow its
[setup instructions](servers/api/README.md#local-setup) for the standalone service.

TODO(hadriano) improve AGENTS.md with some of the stuff from https://github.com/4gatepylon/SAEScoping/blob/main/AGENTS.md (via trial and error).


Also, make sure you set up a `.env` file with contents like this:
```
OPENAI_API_KEY=sk-proj-...
HF_TOKEN_V2=hf_...
CUDA_VISIBLE_DEVICES=""
WANDB_API_KEY=wandb_v1_...
WANDB_PROJECT=deleteme
PYTHONPATH=/path/to/repo/
OPENROUTER_API_KEY=sk-or-v1-...
ANTHROPIC_API_KEY=sk-ant-api03-...
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
STEGO_ARTIFACTS_DIR=/path/to/repo/artifacts/
```

> NOTE: if you get issues that you cannot connect to modal due to something involving a "Proxy" it _might_ be because of your usage of a VPN or other tool. Adriano had this issue and fixed it by disabling "Freedom" (a tool that blocks distractions like reddit, twitter, hacker news, etc...).