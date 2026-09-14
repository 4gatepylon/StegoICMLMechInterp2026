# Decoder contract tests

Run from the repository root in the `stego` environment:

```bash
conda run -n stego python -m pytest -q ciphers/variable_naming_in_python_v2/tests/test_decoder.py
```

Decoder checks expect the stub's `NotImplementedError`; schema checks run now.
Use `--runxfail` during implementation, then remove `@needs_decoder`.

## Fixture partitions

> NOTE: This README _might_ be outdated. The ground truth is in the fixture
> folders: load `cipher.json` and find the Python filename in
> `expected_decodes.json`. The runner automatically discovers these pairs and
> validates them with Pydantic (`CipherConfig` and `ExpectedDecodes`). Successful
> entries contain a complete `DecodedMessage`; error entries specify rejection
> plus a `decoded` result under the referenced `valid_cipher_folder`.

| Folder under `fixtures/` | Cipher | Python files |
| --- | --- | --- |
| [`codex_generated_1_group`](fixtures/codex_generated_1_group/) | `index: i=0, j=1` | All ten partitions below |
| [`codex_generated_2_groups`](fixtures/codex_generated_2_groups/) | `index: i=0, j=1`; `state: a=0, b=1` | Same ten partitions; every file uses both groups |
| [`codex_generated_2_groups_invalid_cipher`](fixtures/codex_generated_2_groups_invalid_cipher/) | `index: [i, j]`; `state: [a, j]` overlap at `j` | `01_basic.py`, valid under the disjoint two-group cipher |

Both valid suites use two length bits. Streams include ignored trailing symbols.

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

## Test-space decision tree

```text
Cipher valid?
├── No → reject before decoding
│   ├── JSON fixture: two groups overlap; same source has a valid-cipher control
│   └── Schema cases: missing/empty/singleton/non-power-of-two groups,
│       duplicate names, invalid/keyword/non-normalized names, invalid widths
└── Yes → one group OR multiple groups (represented by two)
    └── Encoding status? Apply this partition to each group-count branch:
        ├── No encoding → zero control; remaining bits ignored (both suites)
        ├── Invalid encoding / undecodable source
        │   ├── Missing control, truncated length, truncated payload
        │   ├── Parse errors, illegal scopes, null byte
        │   └── Unsupported wildcard import
        │       Coverage: one-group inline cases; two-group cases still missing
        └── Valid encoding
            ├── Empty → present control, length 0, empty payload (both suites)
            └── Non-empty → partition by payload length and binding syntax
                ├── Length 1: basic plus syntax cases 05–10 (both suites)
                ├── Length 3: maximum for two length bits; leading zeros
                │   and trailing ignored bits (both suites)
                └── Additional one-group inline cases:
                    length 2 with multi-bit symbols crossing frame boundaries;
                    length 15, maximum for four length bits

Shared checks
├── Fixture results: frame, all bindings, occurrence spans/kinds, group/bit
│   metadata, scope ownership, deterministic output, final stego-only filtering
│   (opaque IDs normalized to first-seen binding_N / scope_N labels)
├── Additional source cases: reads before assignment, UTF-8 byte columns,
│   strings excluded, no source execution or dependency imports
└── Result schemas: absence vs. empty, leading-zero round trips, inconsistent
    payloads, invalid spans, incomplete contributions, occurrence ordering

Outside coverage: runtime correctness, dynamic exec-created names,
steganographic detectability, exhaustive Python grammar (including async,
type-parameter/annotation scopes, and private-name mangling).
```
