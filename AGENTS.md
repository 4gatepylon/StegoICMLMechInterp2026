# Agent conventions

## General best practices

1. All imports must be relative to the repo root (for example `from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import load_fineweb`).
2. All agents must add their name as a suffix to the title of each PR and/or commit they create (for example, `Add evaluation metrics [Codex]`).

## Paths in code

1. All filesystem paths must be relative to either:
   1. the repo root, for code and other files that live in this repository
   2. `os.environ["STEGO_ARTIFACTS_DIR"]`, for artifacts such as weights, data, outputs, and similar generated files

## Documentation conventions

1. Write for readers who have no prior conversation context.
2. Docstrings for non-trivial functions must explain why the function exists, how to use each argument, the exact return structure (including dictionary keys and tensor shapes), how callers use each return value, and important invariants or required companion components.
3. When a dictionary crosses a component boundary, document its schema where it is consumed. Enumerate every required key and explain its value and purpose.
4. Inline comments must explain non-obvious causality, invariants, or framework behavior rather than restating the code.
5. Distinguish guarantees enforced by code from conventions followed by current callers. If a producer is replaceable, document the required contract for replacements.
6. Explain magic or sentinel values in terms of who consumes them and what behavior they trigger. When custom and default behavior differ, state both succinctly.
7. Name variables by their concrete representation and role (for example, `prefixed_model_inputs` rather than `student`).
8. Keep detailed interface documentation in docstrings; inline comments may point to that docstring rather than duplicating its schema.
9. Before documenting a data flow or framework behavior, identify its producer and consumer and verify the actual contract in code. Align names and validation with that contract.
