# Decoder contract tests

Run from the repository root in the `stego` environment:

```bash
conda run -n stego python -m pytest -q ciphers/variable_naming_in_python_v2/tests/test_decoder.py
```

This suite accompanies the interface-only decoder. Schema tests execute now;
the decoder tests are marked `xfail(raises=NotImplementedError, strict=True)`.
Only the explicit stub exception is expected: incorrect output and other
exceptions fail normally. A successful decode test becomes a strict XPASS
failure, making the marker's removal an explicit implementation step.

During implementation, disable those expectations to see ordinary test failures:

```bash
conda run -n stego python -m pytest --runxfail -q ciphers/variable_naming_in_python_v2/tests/test_decoder.py
```

Remove the `@needs_decoder` decorators and marker definition when the decoder is
implemented. Do not interpret expected failures as tested decoding functionality.

## Fixture partitions

All ten Python files are read as source, never imported or executed. The alphabet
is `i=0, j=1`; repeated occurrences of a binding do not add symbols. Each expected
stream below includes trailing ignored symbols. Fixture comments are explanatory
and do not supply expectations to the decoder or test assertions.

| File | Length bits | Full stream | Message | Main binding partition |
| --- | --- | --- | --- | --- |
| `01_basic.py` | 1 | `110` | `0` | Separate functions and ordinary bindings |
| `02_absent.py` | 4 | `01` | absent | Zero control ignores the remaining symbol |
| `03_empty.py` | 1 | `10` | empty string | Encoded length zero differs from absence |
| `04_maximum.py` | 2 | `1110011` | `001` | Maximum length, leading zeros, trailing symbol |
| `05_reassignment.py` | 1 | `110` | `0` | Reassignment, augmented assignment, and loops |
| `06_closures.py` | 1 | `110` | `0` | Closure references versus shadowing |
| `07_parameters.py` | 1 | `110` | `0` | Parameters, outer-scope defaults, lambda capture |
| `08_namespaces.py` | 1 | `110` | `0` | Import/definition bindings, class scopes, attributes |
| `09_comprehensions.py` | 1 | `1101` | `0` | Comprehension locals, walrus ownership, global/nonlocal |
| `10_targets.py` | 1 | `110` | `0` | Unpacking, with, exceptions, pattern captures |

Pytest checks frame values, per-bit roles/offsets, distinct binding IDs,
occurrence positions/kinds, ordinary binding inclusion, deterministic output,
and a final stego-only filter that preserves metadata. Inline cases additionally
cover multi-bit symbols crossing frame boundaries, four-bit maximum length,
missing/truncated fields, malformed and non-compiling code, unsupported wildcard
imports, no execution/import side effects, reads before assignments, and UTF-8
byte columns.

Schema tests partition invalid alphabets, identifier normalization, header
widths, inconsistent payloads, and incomplete binding metadata. JSON round trips
must preserve ordered alphabets, leading zeros, and absence versus empty payload.

The suite does not test runtime program correctness, external dependencies,
dynamic names created by `exec`, or steganographic detectability. It is not
exhaustive for Python grammar: async forms, generic type-parameter/annotation
scopes, and private-name mangling need further partitions during implementation.
