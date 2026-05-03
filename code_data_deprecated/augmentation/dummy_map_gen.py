from __future__ import annotations

import orjson
import tqdm
from pathlib import Path


def generate_map_file():
    # All possible keys from DatasetEntry that can be included/excluded
    all_keys = [
        "starter_code",
        "things_to_know",
        "source_description",
        "question",
        "answers",
        "expected_inputs_outputs",
        "difficulty",
        "url",
    ]

    # Generate all possible combinations (2^8 = 256)
    # This includes the empty set
    all_combinations = []

    # Generate all subsets of keys
    for n in tqdm.tqdm(
        range(0, 2 ** len(all_keys)), desc="Generating all combinations"
    ):
        includes = [((n >> i) & 1) == 1 for i in range(len(all_keys))]
        all_combinations.append(
            [all_keys[i] for i in range(len(all_keys)) if includes[i]]
        )
    assert len(set(map(tuple, all_combinations))) == len(all_combinations)  # all combos
    assert len(all_combinations) == 2 ** len(all_keys)

    # Create the map structure
    map_data = []

    for key_combination in tqdm.tqdm(all_combinations, desc="Generating map.json"):
        # Each entry is [list_of_keys, {template_name: probability}]
        # Since we only have dummy.j2, it gets 100% probability
        entry = [sorted(key_combination), {"dummy.j2": 1.0}]  # sort for determinism
        map_data.append(entry)

    # Sort by length of key combination and then alphabetically for consistency
    map_data.sort(key=lambda x: (len(x[0]), x[0]))

    # Write to file
    output_path = (
        Path(__file__).parent / "rephrase_templates" / "map.json"
    )  # copy it l8r
    assert output_path.parent.exists()
    assert not output_path.exists()

    output_path.write_bytes(orjson.dumps(map_data))


if __name__ == "__main__":
    generate_map_file()
