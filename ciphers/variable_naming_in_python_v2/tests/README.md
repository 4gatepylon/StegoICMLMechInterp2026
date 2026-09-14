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

## JSON fixtures and automatic discovery

The runner discovers every immediate subfolder of `fixtures/` containing
`cipher.json`. Add Python files and matching entries in `expected_decodes.json`
to extend a suite; add a folder with both JSON files to create a suite. No Python
case list needs updating. The runner checks that the manifest keys match exactly
the sibling `*.py` filenames and compiles source without importing or executing it.

- **`cipher.json`** is loaded directly with `CipherConfig.model_validate_json()`.
  There are no per-file overrides. Each valid suite uses one control bit and two
  length bits; a maximum-length payload therefore contains three bits.
- **`expected_decodes.json`** is validated with `ExpectedDecodes` in
  [`test_decoder.py`](test_decoder.py). Its keys are exact Python filenames, such
  as `01_basic.py`. Each successful value is a complete `DecodedMessage`:
  `is_encoding`, `length`, `message_bits`, and `bindings`, including ordinary
  bindings, all occurrence spans/kinds, group membership, bits, offsets, and roles.
  See [`decoder.py`](../decoder.py) for the public field contracts.
- **IDs in expected results** use `binding_0`, `binding_1`, ... in binding order
  and `scope_0`, `scope_1`, ... in first-seen scope order. The runner normalizes
  opaque decoder IDs to these labels before comparing the entire result. Reused
  scope labels assert shared ownership; IDs need not match a future decoder's
  internal naming scheme. No other output fields are normalized.
- **Expected invalid ciphers** use an `ExpectedCipherError` value with `error`
  (`"ValidationError"`), `match` (diagnostic substring), `valid_cipher_folder`
  (relative path from this folder), and `decoded` (the complete result under that
  valid companion cipher). Config rejection runs now; decoding the same source
  under the companion cipher is a separate expected failure until implementation.

Fixture comments are explanatory and never supply test expectations. JSON results
are authored expectations, not captured output from the unimplemented decoder.

## Fixture partitions

> NOTE: This README _might_ be outdated. The ground truth is in the fixture
> folders: for a Python file, load its sibling `cipher.json` and look up its exact
> filename in `expected_decodes.json`. An error entry describes cipher rejection;
> its `decoded` field gives the result under `valid_cipher_folder`'s cipher.

| Folder under `fixtures/` | Cipher | Python files |
| --- | --- | --- |
| [`codex_generated_1_group`](fixtures/codex_generated_1_group/) | `index: i=0, j=1` | All ten partitions below |
| [`codex_generated_2_groups`](fixtures/codex_generated_2_groups/) | `index: i=0, j=1`; `state: a=0, b=1` | Same ten partitions; every file uses both groups |
| [`codex_generated_2_groups_invalid_cipher`](fixtures/codex_generated_2_groups_invalid_cipher/) | `index: [i, j]`; `state: [a, j]` overlap at `j` | `01_basic.py`, valid under the disjoint two-group cipher |

The following table summarizes both valid suites. Repeated occurrences of a
binding do not add symbols; full streams include ignored trailing symbols.

> NOTE: These summaries _might_ be outdated; use each folder's `cipher.json` and
> the filename entry in `expected_decodes.json` as the ground truth.

| File | Full stream | Message | Main binding partition |
| --- | --- | --- | --- |
| `01_basic.py` | `1010` | `0` | Separate functions and ordinary bindings |
| `02_absent.py` | `01` | absent | Zero control ignores the remaining symbol |
| `03_empty.py` | `100` | empty string | Encoded length zero differs from absence |
| `04_maximum.py` | `1110011` | `001` | Maximum length, leading zeros, trailing symbol |
| `05_reassignment.py` | `1010` | `0` | Reassignment, augmented assignment, and loops |
| `06_closures.py` | `1010` | `0` | Closure references and separate nested bindings |
| `07_parameters.py` | `1010` | `0` | Parameters, outer-scope defaults, lambda capture |
| `08_namespaces.py` | `1010` | `0` | Import/definition bindings, class scopes, attributes |
| `09_comprehensions.py` | `10101` | `0` | Comprehension locals, walrus ownership, global/nonlocal |
| `10_targets.py` | `1010` | `0` | Unpacking, with, exceptions, pattern captures |

Full result comparisons check frame values, per-bit roles/offsets, distinct
binding IDs, scope ownership, occurrence positions/kinds, and ordinary bindings.
The runner also checks deterministic output and a final stego-only filter that
preserves metadata. The two-group suite interleaves groups by source position;
the single-group suite retains same-spelling shadowing cases.

Focused inline cases additionally cover multi-bit symbols crossing frame
boundaries, four-bit maximum length, missing/truncated fields, malformed and
non-compiling code, unsupported wildcard imports, no execution/import side
effects, reads before assignments, and UTF-8 byte columns. Every decoder call
loads its cipher from JSON; these edge cases use the one-group cipher or
`fixtures/multibit_cipher.json` and `fixtures/four_length_bits_cipher.json`.
Schema unit tests still construct intentional valid/invalid model inputs directly.

Schema tests partition invalid alphabets, identifier normalization, header
widths, inconsistent payloads, and incomplete binding metadata. JSON round trips
must preserve ordered alphabets, leading zeros, and absence versus empty payload.

The suite does not test runtime program correctness, external dependencies,
dynamic names created by `exec`, or steganographic detectability. It is not
exhaustive for Python grammar: async forms, generic type-parameter/annotation
scopes, and private-name mangling need further partitions during implementation.
