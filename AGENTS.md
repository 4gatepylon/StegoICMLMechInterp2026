# Agent conventions

## General best practices

1. All imports must be relative to the repo root (for example `from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import load_fineweb`).
2. All agents must add their name as a suffix to the title of each PR and/or commit they create (for example, `Add evaluation metrics [Codex]`).
3. Use the `stego` Conda environment. If it does not exist, stop and ask the user to create it with `conda create -n stego python=3.12 -y`.
4. Use Pydantic or pydantic-yaml for configuration schemas. Do not implement ad hoc dictionary merging or validators such as `_check_no_reserved`.
5. Use Click instead of argparse for new command-line interfaces when available. Tell the user if a significant portion of existing code still uses argparse.
6. Do not create trivial modules under 100 lines unless necessary.
7. Do not create complex modules over 750 lines unless necessary.

## Interface and testing

1. Define and propose the interface before implementing it, and ask the user to clarify the interface.
2. Based on the clarified interface, propose the tests and ask the user to clarify them before implementation.
3. State exactly what the tests cover and what they omit.
4. Tests must describe their test space and chosen partitions clearly and succinctly.
5. Tests are meant to reduce the chance of bugs. Do not add tests that merely assert that a constant or default argument equals its declared value; test behavior or a meaningful contract instead.

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
10. Use `@override` when overriding methods. The overriding method's docstring must state how its behavior differs from the parent implementation.
