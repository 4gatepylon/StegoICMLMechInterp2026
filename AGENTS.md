# Agent conventions

1. All imports must be relative to the repo root (for example `from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import load_fineweb`).
2. All filesystem paths must be relative to either:
   1. the repo root, for code and other files that live in this repository
   2. `os.environ["STEGO_ARTIFACTS_DIR"]`, for artifacts such as weights, data, outputs, and similar generated files