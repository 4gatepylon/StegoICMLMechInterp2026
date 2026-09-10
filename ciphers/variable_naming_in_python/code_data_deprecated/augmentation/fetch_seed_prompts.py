from __future__ import annotations
import io
from pathlib import Path
from typing import Optional, Set, List, Literal, Iterator, Dict, Any
import json
import orjson
from datasets import DatasetDict, Dataset, concatenate_datasets
from nemi_mvp.dataset_lib.data_entry import DatasetEntry, DatasetMerger
import tqdm


class DatasetEntrySeeder:
    def __init__(
        self,
        folder: Path = Path(__file__).parent.parent.parent / "merged_seed_prompt_datasets",  # output_folder
        save_mode: Literal["jsonl"] = "jsonl",
        skip_files_too_large: bool = False,
    ):
        self.folder = folder
        self.save_mode = save_mode
        if self.save_mode != "jsonl":
            raise NotImplementedError(f"Save mode {self.save_mode} not implemented yet")

    def save_seeds(
        self,
        dataset_dict_folder: Optional[Path] = Path(__file__).parent.parent.parent / "merged_code_datasets",  # fmt: skip
        max_file_size_bytes: int = 50 * 1e6,
        avoid_splits: List[str] = ["test", "validation"],
        max_file_n: Optional[int] = None,
    ) -> None:
        """
        Load ALL of the datasets and ALL of their items (instead of only certain difficulties)
        and then save the prompts to the output folder.

        Args:
            dataset_dict_folder: The folder containing the dataset dicts that we want to
                avoid the prompts for (specifically, this is a DatasetDict and we want to
                avoid the prompts for the "test" and "validation" splits).
            max_file_size_bytes: The maximum file size in bytes. Default is 50MB for github.
            avoid_splits: The splits to avoid the prompts for. Default is ["test", "validation"].
        """
        output_folder = self.folder
        if output_folder.exists() and len(list(output_folder.iterdir())) > 0:
            raise FileExistsError(f"Output folder {output_folder} already contains files")
        # 1. Load the avoid prompts
        avoid_prompts: Set[str] = set()
        if dataset_dict_folder is not None:
            if not dataset_dict_folder.exists():
                raise FileNotFoundError(f"Dataset dict folder {dataset_dict_folder} does not exist")
            dd = DatasetDict.load_from_disk(dataset_dict_folder)
            if not set(avoid_splits).issubset(set(dd.keys())):
                raise ValueError(f"Avoid splits {avoid_splits} are not a subset of the dataset dict keys {dd.keys()}")
            for split_name in dd.keys():
                if split_name in avoid_splits:
                    assert "prompt" in dd[split_name].column_names, f"Prompt column not found in {split_name} split"
                    for entry in dd[split_name]:
                        avoid_prompts.add(entry["prompt"])
        print(f"Found a total of {len(avoid_prompts)} prompts to avoid from folder: {dataset_dict_folder}")
        print("=" * 100)
        # 2. Collect all available dataset entries without these prompts
        merger = DatasetMerger()  # Dummy
        print("Loading datasets for difficulties column analysis...")
        print("=" * 100)
        dataset_name2dataset: Dict[str, Dataset] = {}
        print("Loading BAAI/TACO...")
        dataset_name2dataset["BAAI/TACO"] = merger._get_baai_taco_raw_datset()
        print("=" * 100)
        print("Loading deepmind/code_contests...")
        dataset_name2dataset["deepmind/code_contests"] = merger._get_deepmind_code_contests_raw_dataset()
        print("=" * 100)
        print("Loading livecodebench/code_generation_lite...")
        dataset_name2dataset["livecodebench/code_generation_lite"] = merger._get_live_code_bench_raw_dataset()
        print("=" * 100)
        print("Loading codeparrot/apps...")
        dataset_name2dataset["codeparrot/apps"] = merger._get_codeparrot_apps_raw_dataset()
        print("=" * 100)
        print("[OK] Loaded datasets!")
        print("=" * 100)
        print("Loading columns of difficulties...")
        dataset_name2_all_difficulties_set: Dict[str, Set[str | int]] = {}
        for dataset_name, dataset in tqdm.tqdm(dataset_name2dataset.items(), desc="Loading difficulties columns..."):
            assert "difficulty" in dataset.column_names, f"Dataset {dataset_name} does not have a difficulty column"
            dataset_name2_all_difficulties_set[dataset_name] = sorted(list(set([x["difficulty"] for x in dataset])))
        print("Columns:")
        print(json.dumps(dataset_name2_all_difficulties_set, indent=4))
        print("[OK] Got all difficulties!")
        print("=" * 100)
        print("Merging datasets...")
        merger = DatasetMerger(
            # Default split fractions OK since we will merge, BUT
            # we want ALL problems so we must filter for ALL the difficulties
            name2difficulties=dataset_name2_all_difficulties_set,
            # [BEGIN]
            # NOTE these are COPIED From `script1_datasets_merge.py` in `nemi_mvp`
            splits_fracs={
                "train": 0.965,
                "test": (1 - 0.965) / 2,
                "validation": (1 - 0.965) / 2,
            },
            splits_fracs_min_n={
                "train": 18_000,
                "test": 20,
                "validation": 20,
            },
            # [END]
        )
        dataset_dict: DatasetDict = merger.create_combined_dataset()
        print("[OK] Merged datasets!")
        print("=" * 100)
        print("Ensuring lengths...")
        assert set(dataset_dict.keys()) == {
            "train",
            "test",
            "validation",
        }, f"Got {dataset_dict.keys()} keys, but expected {['train', 'test', 'validation']}"
        dataset_combined: Dataset = concatenate_datasets([dataset_dict[split_name] for split_name in dataset_dict.keys()])
        assert len(dataset_combined) == len(dataset_dict["train"]) + len(dataset_dict["test"]) + len(dataset_dict["validation"]), (
            f"Dataset combined has {len(dataset_combined)} entries, but should have {len(dataset_dict['train']) + len(dataset_dict['test']) + len(dataset_dict['validation'])} entries"
        )
        print("[OK] Ensured lengths!")
        print("=" * 100)
        print("Filtering prompts to avoid...")
        _old_len = len(dataset_combined)
        dataset_combined: List[Dict[str, Any]] = [
            z for z in tqdm.tqdm(dataset_combined, desc="Filtering for prompts to avoid...") if z["question"] not in avoid_prompts
        ]
        print("[OK] Filtered for prompts to avoid!")
        print(f"Num that do have the prompts to avoid: {_old_len - len(dataset_combined)}")
        print(f"From num prompts to avoid: {len(avoid_prompts)}")
        print(f"New size: {len(dataset_combined)}")
        print("=" * 100)
        print("Shuffling...")
        dataset_combined: Dataset = Dataset.from_list(dataset_combined).shuffle(seed=1026576343452)
        print("[OK] Shuffled!")
        print("=" * 100)

        # 3. Save the prompts to the output folder
        output_folder.mkdir(parents=True, exist_ok=True)
        entry_buff: io.BytesIO = io.BytesIO()
        entry_buff_n: int = 0
        entry_buff_n_bytes_written: int = 0
        entry_buff_n_count_in_buff: int = 0
        for entry in tqdm.tqdm(dataset_combined, desc="Saving seeds..."):
            _serialized_entry: bytes = orjson.dumps(entry) + b"\n"
            # If overflow, then write and clear buffer.
            assert entry_buff_n_bytes_written == len(entry_buff.getvalue())  # debug, -O this
            if entry_buff_n_bytes_written + len(_serialized_entry) > max_file_size_bytes:
                file_path = output_folder / f"{entry_buff_n}.jsonl"
                assert not file_path.exists()
                file_path.write_bytes(entry_buff.getvalue())
                entry_buff = io.BytesIO()
                entry_buff_n += 1
                entry_buff_n_bytes_written = 0
                entry_buff_n_count_in_buff = 0
                if max_file_n is not None and entry_buff_n >= max_file_n:
                    raise ValueError(f"Max file n {max_file_n} reached")
            # Now we can just write to the buffer
            entry_buff_n_bytes_written += entry_buff.write(_serialized_entry)
            entry_buff_n_count_in_buff += 1
            if entry_buff_n_bytes_written > max_file_size_bytes:
                raise ValueError(
                    f"Entry buffer size {entry_buff_n_bytes_written} is greater "
                    + f"than the max file size {max_file_size_bytes},"
                    + f"n_in_buff={entry_buff_n_count_in_buff}"
                )
        # Write at the end if possible
        if entry_buff_n_bytes_written > 0:
            file_path = output_folder / f"{entry_buff_n}.jsonl"
            assert not file_path.exists()
            assert entry_buff_n_bytes_written <= max_file_size_bytes, (
                f"Entry buffer size {entry_buff_n_bytes_written} is greater than the max file size {max_file_size_bytes}, n_in_buff={entry_buff_n_count_in_buff}"
            )
            file_path.write_bytes(entry_buff.getvalue())

        assert len(self.load_seeds(deserialize=False)) == len(dataset_combined)  # debug; -O this

    def load_seeds_stream(self, deserialize: bool = True) -> Iterator[DatasetEntry | Dict[str, Any]]:
        if not self.folder.exists() or len(list(self.folder.iterdir())) == 0:
            raise FileNotFoundError(f"Output folder {self.folder} does not exist or is empty")
        if self.save_mode == "jsonl":
            for file in sorted(list(self.folder.glob("*.jsonl"))):
                with open(file, "rb") as f:
                    for line in f.readlines():
                        if len(line.strip()) == 0:
                            continue
                        _loaded_entry = orjson.loads(line)
                        if deserialize:
                            loaded_entry = DatasetEntry.deserialize(_loaded_entry)
                        else:
                            loaded_entry = _loaded_entry
                        yield loaded_entry
        else:
            raise NotImplementedError(f"Load mode {self.save_mode} not implemented yet")

    def load_seeds(self, *args, **kwargs) -> List[DatasetEntry | Dict[str, Any]]:
        return list(self.load_seeds_stream(*args, **kwargs))


if __name__ == "__main__":
    loader = DatasetEntrySeeder(
        folder=Path(__file__).parent.parent.parent / "merged_seed_prompt_datasets",
        save_mode="jsonl",
    )
    loader.save_seeds(
        # NOTE this location choice is for the align machines
        dataset_dict_folder=Path(__file__).parent.parent.parent / "merged_code_datasets",
        avoid_splits=["test", "validation"],
        max_file_size_bytes=500 * 1e6,  # github limit is 100MB but we rnt storing :P
        max_file_n=500,
    )
