from __future__ import annotations

import itertools
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import jinja2
import orjson
import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets
from nemi_mvp.dataset_lib.load_augmented_j2s import (
    get_augmenting_templates,
    get_hydrating_dict,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

"""
Provide simple utilities for loading chat datasets from disk. They are expected to be
saved as either JSONL or JSON files where each entry/line is either one chat or a list
of chats (and we want to basically combine everything into a big list of chats)

A chat is defind by the OpenAI API format/schema.

NOTE: these are mainly meant to be used for:
1. SAE-training
2. SFT post-SAE training (recovery)
3. NOT RL, at least not necessarily...
4. DEFINATELY NOT BENCHMARKING/EVALUATIONS (these are meant for TRAINING)

The main entrypoint (usage) is to just pas the dataset name you want in `load_chats`
(and you can also ask for deduplication and exclusion of validation/test-set prompts).
"""
# First one is always user which is why we take [0] in the dedup mode
ROLE_SCORE = {"user": 0, "system": 1}
assert sorted(["user", "system"], key=lambda x: ROLE_SCORE[x]) == ["user", "system"]
assert sorted(["system", "user"], key=lambda x: ROLE_SCORE[x]) == ["user", "system"]

NAME2ARGS = {
    # We have multiple datasets at differnet paths; this is hardcoded for the lab machines
    # NOTE that there are two main classes: (1) chat/input datasets and (2) chat/output datasets
    # (the 1st is meant for chats that have not yet had inference and the 2nd is meant for
    # chats that have)
    #
    ######################## Output Datasets ########################
    #### Original Generations ####
    "api_generations": {  # the easy/medium no-thinking answers by api (4.1, 5, etc...)
        "path": Path(__file__).parent.parent / "code_api_generations/",
        "globs": ["*.jsonl"],
        "force_0turn": False,
        "must_be_0turn": False,
        "must_be_1turn": True,
    },
    "qwen_generations": {  # the easy/medium no-thinking answers by qwen
        "path": Path(__file__).parent.parent / "code_qwen_generations/",
        "globs": ["**/conversation_*.jsonl"],
        "force_0turn": False,
        "must_be_0turn": False,
        "must_be_1turn": True,
    },
    #### Augmented Prompts' (from ^) Generations ####
    "code_augmented_qwen_generations": {
        "path": Path(__file__).parent.parent / "code_augmented_qwen_generations/",
        "globs": ["*.json"],
        "force_0turn": False,
        "must_be_0turn": True,
        "must_be_1turn": False,
    },
    "code_augmented_api_generations": {
        "path": Path(__file__).parent.parent / "code_augmented_api_generations/",
        "globs": ["*.json"],
        "force_0turn": False,
        "must_be_0turn": False,
        "must_be_1turn": True,
    },
    #### Huggingface Generations ####
    "default_generations": {  # Generations/chats w/ answers acc. to huggingface
        "path": Path(__file__).parent.parent / "code_dataset_default_generations/",
        "globs": ["*.json"],
        "force_0turn": False,
        "must_be_0turn": False,
        "must_be_1turn": True,
    },
    "default_qwen_generations": {  # ^ but the answers r placed by qwen's answers
        "path": Path(__file__).parent.parent / "code_dataset_default_qwen_generations/",
        "globs": ["*.json"],
        "force_0turn": False,
        "must_be_0turn": False,
        "must_be_1turn": True,
    },
    "default_api_generations": {  # ^ but the answers r placed by api's answers
        "path": Path(__file__).parent.parent / "code_dataset_default_api_generations/",
        "globs": ["*.json"],
        "force_0turn": False,
        "must_be_0turn": False,
        "must_be_1turn": True,
    },
    ######################## Input Datasets ########################
    #### Original Prompts ####
    "prompts": {  # the easy/medium no-thinking answers by api (4.1, 5, etc...)
        "path": Path(__file__).parent.parent / "code_qwen_generations/",
        "globs": ["**/conversation_*.jsonl"],
        "force_0turn": True,  # NOTE: we force it DOWN by turning 1turn -> 0turn
        "must_be_0turn": False,
        "must_be_1turn": True,
    },
    "prompts_from_api": {  # same as ^ but we get it from the API folder instead
        "path": Path(__file__).parent.parent / "code_api_generations/",
        "globs": ["*.jsonl"],
        "force_0turn": True,  # NOTE: we force it DOWN by turning 1turn -> 0turn
        "must_be_0turn": False,
        "must_be_1turn": True,
    },
    #### Augmented Prompts' (from ^) Prompts ####
    "augmented_prompts": {
        "path": Path(__file__).parent.parent / "code_augmented_prompts/",
        "globs": ["*.json"],
        "force_0turn": False,
        "must_be_0turn": True,  # NOTE: default is actually 0turn
        "must_be_1turn": False,
    },
    #### Huggingface Prompts ####
    "default_prompts": {  # Huggingface prompts
        "path": Path(__file__).parent.parent / "code_dataset_default_generations/",
        "globs": ["*.json"],
        "force_0turn": True,  # NOTE: we force to turn them back into PROMPTS
        "must_be_0turn": False,  # (huggingface dataset has this stuff setup here)
        "must_be_1turn": True,
    },
}


class EdgeCase:
    @staticmethod
    def _is_prompt_generations(chat: List[Dict[str, str]]) -> bool:
        if not isinstance(chat, list):
            return False
        if not all(isinstance(item, dict) for item in chat):
            return False
        if not all(set(item.keys()) == {"prompt", "generation"} for item in chat):
            return False
        if not all(all(isinstance(v, str) for v in item.values()) for item in chat):
            return False
        return True

    @staticmethod
    def _prompt_generation_to_chat(chat: Dict[str, str]) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": chat["prompt"]},
            {"role": "assistant", "content": chat["generation"]},
        ]

    @staticmethod
    def _prompt_generations_to_chats(
        chats: List[Dict[str, str]],
    ) -> List[List[Dict[str, str]]]:
        return [EdgeCase._prompt_generation_to_chat(chat) for chat in chats]


def is_valid_chat(chat: List[Dict[str, str]]) -> bool:
    if not isinstance(chat, list):
        return False
    if not all(isinstance(item, dict) for item in chat):
        return False
    if not all(set(item.keys()) == {"role", "content"} for item in chat):
        return False
    if not all(item["role"] in ["system", "user", "assistant"] for item in chat):
        return False
    if not all(isinstance(item["content"], str) for item in chat):
        return False
    return True


# 0 turn => just a prompt (optionally WITH or WITHOUT sysprompt OK)
# (must by system then user or just user)
# 1 turn => a 0turn chat followed by an assistant message
def is_valid_0turn_chat(chat: List[Dict[str, str]]) -> bool:
    if not is_valid_chat(chat):
        return False

    # Must be either [user] or [system, user]
    if len(chat) == 1:
        return chat[0]["role"] in {"user", "system"}
    elif len(chat) == 2:
        return chat[0]["role"] == "system" and chat[1]["role"] == "user"
    else:
        return False


def is_valid_1turn_chat(chat: List[Dict[str, str]]) -> bool:
    if not is_valid_chat(chat):
        return False

    # Must be either [user, assistant] or [system, user, assistant]
    if len(chat) == 2:
        return chat[0]["role"] in {"user", "system"} and chat[1]["role"] == "assistant"
    elif len(chat) == 3:
        return chat[0]["role"] == "system" and chat[1]["role"] == "user" and chat[2]["role"] == "assistant"
    else:
        return False


def convert_1turn_to_0turn(chat: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove the assistant message and return the 0turn chat"""
    if not is_valid_1turn_chat(chat):
        raise ValueError("Input chat is not a valid 1turn chat")

    # Remove the last message (assistant)
    return chat[:-1]


def is_array_of_chats(contents: List[Any]) -> bool:
    if not isinstance(contents, list):
        return False
    if not all(is_valid_chat(x) for x in contents):
        return False
    return True


def is_array_of_arrays_of_chats(contents: List[Any]) -> bool:
    if not isinstance(contents, list):
        return False
    if not all(is_array_of_chats(x) for x in contents):
        return False
    return True


def _load_chats(
    folder: Path,
    globlist: List[str],
    force_0turn: bool = False,
    must_be_0turn: bool = False,
    must_be_1turn: bool = False,
    debug_flags: Dict[str, Any] = {
        # max_load_files -> only one supported
    },
) -> List[List[Dict[str, str]]]:
    """
    Generic method that can load chats, whether they be jsonl, json, or a mix of both
    as well as whether you store as list of chats or list of list of chats.
    """
    if force_0turn and must_be_0turn:
        raise ValueError("force_0turn and must_be_0turn cannot both be True")
    if must_be_0turn and must_be_1turn:
        raise ValueError("must_be_0turn and must_be_1turn cannot both be True")
    if not folder.exists():
        raise ValueError(f"Folder {folder} does not exist")

    if len(globlist) == 0:
        print("WARNING USING DEFAULT JSON AND JSONL SHALLOW GLOBS")
        globlist = ["*.json", "*.jsonl"]
    files = []
    for glob in globlist:
        files.extend(sorted(list(folder.glob(glob))))
    n_max_files = debug_flags.get("max_load_files", len(files))
    files = files[:n_max_files]
    assert len(files) > 0
    all_chats: List[List[Dict[str, str]]] = []
    for file in tqdm.tqdm(files, desc="Loading chats' files in order..."):
        contents = None
        load_jsonl = file.suffix == ".jsonl"
        if file.suffix == ".json":
            try:
                contents = orjson.loads(file.read_bytes())
            except Exception:
                # Sometimes we renamed incorrectly
                load_jsonl = True
        if load_jsonl:  # this might even work for json since orjson stuffs into one line
            contents = [orjson.loads(line) for line in file.read_bytes().splitlines() if line.strip()]
        assert contents is not None
        # Now we have an array of chats OR an array of arrays of chats
        _is_array_of_chats = is_array_of_chats(contents)
        _is_array_of_arrays_of_chats = is_array_of_arrays_of_chats(contents)
        _edge_case_is_prompt_generations = EdgeCase._is_prompt_generations(contents)
        assert _is_array_of_chats or _is_array_of_arrays_of_chats or _edge_case_is_prompt_generations, (
            f"type={type(contents)}\n\ntypes={set(type(x) for x in contents)}\n{contents[0].keys()}"
        )
        if _is_array_of_chats:
            contents = [contents]  # now should be array of arrays of chats
        elif _edge_case_is_prompt_generations:
            contents = EdgeCase._prompt_generations_to_chats(contents)
            assert is_array_of_chats(contents)
            contents = [contents]  # NOW also array of arrays of chats
        assert is_array_of_arrays_of_chats(contents)
        for these_chats in contents:
            all_chats.extend(these_chats)  # this is an array of chats

    # Make sure they are all indeed chats
    assert all(is_valid_chat(chat) for chat in all_chats)
    if must_be_0turn:
        assert all(is_valid_0turn_chat(chat) for chat in all_chats), f"is_valids: {100 *sum(is_valid_0turn_chat(chat) for chat in all_chats) / len(all_chats)}% chats[0]: {[c for c in all_chats if not is_valid_0turn_chat(c)][:1]}"  # fmt: skip
    if must_be_1turn:
        assert all(is_valid_1turn_chat(chat) for chat in all_chats), f"is_valids: {100 *sum(is_valid_1turn_chat(chat) for chat in all_chats) / len(all_chats)}% chats[0]: {[c for c in all_chats if not is_valid_1turn_chat(c)][:1]}"  # fmt: skip
    if force_0turn:
        assert all(is_valid_1turn_chat(chat) for chat in all_chats)
        all_chats = [convert_1turn_to_0turn(chat) for chat in all_chats if is_valid_1turn_chat(chat)]
    return all_chats


def load_exclusion_prompts(
    dd_path: Path = Path(__file__).parent.parent / "merged_code_datasets/",
    dd_splits: List[str] = ["test", "validation"],  # exclude these splits
    exclusion_mode: Literal[
        "substring",
        "substring_lower",
        "substring_strip",
        "substring_lower_strip",
        "exact" | "exact_lower" | "exact_strip" | "exact_lower_strip",
    ] = "substring",
) -> List[str]:
    if not dd_path.exists():
        raise ValueError(f"Dataset path {dd_path} does not exist")
    assert all(split in ["test", "validation"] for split in dd_splits)
    dd = DatasetDict.load_from_disk(dd_path)
    dataset: Dataset = concatenate_datasets([dd[split] for split in dd_splits])
    # prompt will lead to exactly the prompts matching in the conversation being excl.
    # but question will lead to a slower substring check search where even prompts that
    # are slight modifications will not appear
    key = "question" if exclusion_mode == "substring" else "prompt"
    qs = list(set([di[key] for di in dataset]))
    return qs


def should_exclude_chat(
    chat: List[Dict[str, str]],
    exclusion_list: List[str] | Set[str],
    exclusion_mode: Literal[
        "substring",
        "substring_lower",
        "substring_strip",
        "substring_lower_strip",
        "exact" | "exact_lower" | "exact_strip" | "exact_lower_strip",
    ] = "substring",
) -> bool:
    contents = [item["content"] for item in chat]
    if "strip" in exclusion_mode:
        contents = list(map(str.strip, contents))
    if "lower" in exclusion_mode:
        contents = list(map(str.lower, contents))
    if "exact" in exclusion_mode:
        assert isinstance(exclusion_list, set), f"Expected exclusion_list to be a set, got {type(exclusion_list)} (are you sure you want this? it will be slower if not a set)"
        return any(content in exclusion_list for content in contents)  # O(|C|)
    else:
        assert exclusion_mode in [f"substring{x}" for x in ["", "_lower", "_strip", "_lower_strip"]], f"Invalid exclusion mode: {exclusion_mode}"  # fmt: skip
        return any(exclusion in content for exclusion in exclusion_list for content in contents)  # O(|C| * |E|)


def load_exclusion_prompts_from_dataset(
    chats: List[List[Dict[str, str]]],
    exclusion_dd_path: Path = Path(__file__).parent.parent / "merged_code_datasets/",
    exclusion_dd_splits: List[str] = ["test", "validation"],
    exclusion_mode: Literal[
        "substring",
        "substring_lower",
        "substring_strip",
        "substring_lower_strip",
        "exact" | "exact_lower" | "exact_strip" | "exact_lower_strip",
    ] = "substring",
) -> List[str]:
    assert isinstance(exclusion_dd_path, Path)
    exclusion_list = load_exclusion_prompts(
        dd_path=exclusion_dd_path,
        dd_splits=exclusion_dd_splits,
        exclusion_mode=exclusion_mode,
    )
    if "lower" in exclusion_mode:
        exclusion_list = list(map(str.lower, exclusion_list))
    if "strip" in exclusion_mode:
        exclusion_list = list(map(str.strip, exclusion_list))
    if "exact" in exclusion_mode:
        exclusion_list = set(exclusion_list)  # Hashing will make this MUCH faster
    return [chat for chat in tqdm.tqdm(chats, desc="Excluding chats...") if not should_exclude_chat(chat, exclusion_list, exclusion_mode)]


def select_from_duplicates_list(
    duplicates_list: List[List[Dict[str, str]]],
    selection_mode: Literal["random"] = "random",
) -> List[List[Dict[str, str]]]:
    if selection_mode == "random":
        s = random.choice(duplicates_list)
        assert is_valid_chat(s)
        return s
    else:
        raise ValueError(f"Selection mode {selection_mode} not supported")


def dedup_chats(
    chats: List[List[Dict[str, str]]],
    selection_mode: Literal["random"] = "random",
    dedup_mode: Literal["user_dict", "full_template", "system_user_template"] = "user_dict",
    tokenizer: Optional[str | AutoTokenizer] = None,
) -> List[List[Dict[str, str]]]:
    """
    Here are the supported dedup moes and their meanings:
    (selection mode means which of the duplicates to select and random means uniformly random)
    - user_dict: equality IFF same user message (must be only one)
    - full_template: equality IFF same full template (i.e. the user's message + the system's message)
    - system_user_template: equality IFF the role in {"system", "user"} messages are 1:1
        (the subsequence)
    """
    if "template" in dedup_mode:
        if tokenizer is None:
            raise ValueError("tokenizer is required for template dedup")
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # <hashables>
    # Get a 1:1 list of hashables that then we will use to calculate whether these are duplicates or not
    hashables = None
    if dedup_mode == "user_dict":
        assert all(len([c for c in chat if c["role"] in {"user", "system"}]) <= 2 for chat in chats)
        assert all(len([c for c in chat if c["role"] in {"user"}]) <= 1 for chat in chats)
        assert all(len([c for c in chat if c["role"] in {"user"}]) <= 1 for chat in chats)
        hashables = [
            sorted(
                [c for c in chat if c["role"] in {"user", "system"}],
                key=lambda x: ROLE_SCORE[x["role"]],
            )[0]["content"]
            for chat in tqdm.tqdm(chats, desc="Filtering user messages")
        ]
    elif dedup_mode == "full_template":
        hashables = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in tqdm.tqdm(chats, desc="Applying chat template (full)")]
    elif dedup_mode == "system_user_template":
        hashables = [
            tokenizer.apply_chat_template([c for c in chat if c["role"] in {"system", "user"}], tokenize=False)
            for chat in tqdm.tqdm(chats, desc="Applying chat template (system_user)")
        ]
    else:
        raise ValueError(f"Invalid dedup mode: {dedup_mode}")
    assert hashables is not None
    assert isinstance(hashables, list)
    assert all(isinstance(hashable, str) for hashable in hashables)
    assert len(hashables) == len(chats)
    # DONE
    # </hashables>
    #
    # dupliates_dict is from hashable to list of chats under that hashable
    duplicates_dict: Dict[str, List[List[Dict[str, str]]]] = {}
    for hashable, chat in tqdm.tqdm(
        zip(hashables, chats),
        desc="Creating duplicates dictionary",
        total=len(hashables),
    ):
        if hashable not in duplicates_dict:
            duplicates_dict[hashable] = []
        duplicates_dict[hashable].append(chat)
    selections: List[List[Dict[str, str]]] = [
        select_from_duplicates_list(duplicates_dict[hashable], selection_mode)
        for hashable in tqdm.tqdm(
            duplicates_dict,
            desc="Selecting from duplicates",
            total=len(duplicates_dict),
        )
    ]
    assert all(is_valid_chat(selection) for selection in selections)
    return selections


# TODO(Adriano) add support for loading OOD chats and also add support for loading
# fewer chats faster (i.e. some kind of pre-randomized streaming)
def load_chats(
    dataset_name: str,
    # None means no exclusion
    exclusion_mode: Optional[
        Literal[
            "substring",
            "substring_lower",
            "substring_strip",
            "substring_lower_strip",
            "exact" | "exact_lower" | "exact_strip" | "exact_lower_strip",
        ]
    ] = "substring",
    dedup_mode: Optional[Literal["user_dict", "full_template", "system_user_template"]] = "user_dict",
    tokenizer: Optional[str | AutoTokenizer] = None,
    verbose: bool = True,
    debug_flags: Dict[str, Any] = {
        # max_load_files -> passthrough to _load_chats
        # max_length -> supported to dedup/etc... less chats
        # max_length_apply_time -> supported to dedup/etc... less chats
    },
) -> List[List[Dict[str, str]]]:
    if dataset_name not in NAME2ARGS:
        raise ValueError(f"Dataset name {dataset_name} not found in NAME2ARGS")
    path = NAME2ARGS[dataset_name]["path"]
    globlist = NAME2ARGS[dataset_name]["globs"]
    force_0turn = NAME2ARGS[dataset_name]["force_0turn"]
    must_be_0turn = NAME2ARGS[dataset_name]["must_be_0turn"]
    must_be_1turn = NAME2ARGS[dataset_name]["must_be_1turn"]
    chats: List[List[Dict[str, str]]] = _load_chats(
        path,
        globlist,
        force_0turn,
        must_be_0turn,
        must_be_1turn,
        debug_flags,
    )
    debug_max_length = debug_flags.get("max_length", None)
    debug_max_length_apply_time = debug_flags.get("max_length_apply_time", "")
    if debug_max_length_apply_time not in [
        "",
        "before_exclusion",
        "before_dedup",
        "before_return",
    ]:
        raise ValueError(f"Invalid debug flag: max_length_apply_time: {debug_max_length_apply_time}")
    if debug_max_length is not None and debug_max_length_apply_time == "before_exclusion":
        print(f"Clipping the chats from length {len(chats)} to length{debug_max_length} @ before_exclusion")
        chats = chats[: debug_flags["max_length"]]
    if exclusion_mode is not None:
        # Use default paths since they should be correct for align machines
        chats_before: int = len(chats)
        chats = load_exclusion_prompts_from_dataset(chats, exclusion_mode=exclusion_mode)
        chats_after: int = len(chats)
        if verbose:
            print(f"Excluded {chats_before - chats_after} chats")
    if debug_max_length is not None and debug_max_length_apply_time == "before_dedup":
        print(f"Clipping the chats from length {len(chats)} to length{debug_max_length} @ before_dedup")
        chats = chats[: debug_flags["max_length"]]
    if dedup_mode is not None:
        chats_before: int = len(chats)
        chats = dedup_chats(chats, dedup_mode=dedup_mode, tokenizer=tokenizer)
        chats_after: int = len(chats)
        if verbose:
            print(f"Deduped {chats_before - chats_after} chats (now there are {chats_after} chats)")
    if debug_max_length is not None and debug_max_length_apply_time == "before_return":
        print(f"Clipping the chats from length {len(chats)} to length{debug_max_length} @ before_return")
        chats = chats[: debug_flags["max_length"]]
    assert all(is_valid_chat(chat) for chat in chats)
    return chats


def load_split_chats_0turn(
    folder: Path = Path(__file__).parent.parent / "merged_code_datasets",
    verbose: bool = True,
    include_all_prompts: bool = False,
    split: str = "validation",
    minimum_samples: int = 20,
    maximum_samples: int = 20,
    difficulties_allowed: Set[str] = {
        # Excludes MEDIUM and MEDIUM_HARD as well as medium, 6, etc...
        "easy",
        "EASY",
        "introductory",
        "0",
        "1",
        "2",
        0,
        1,
        2,
    },
    random_seed: int = 888,
    exclude_regexes: List[str] = [r".*<image>.*"],  # This is usually missing information
) -> List[List[Dict[str, str]]]:
    """
    Support loading validation/test/etc... sets of chats possibly with or withot
    data augmentation. These are 0-turn chats meaning they have only an input and adding
    system prompt is not supported yet.
    """
    dd: DatasetDict = DatasetDict.load_from_disk(folder)
    d_validation = dd[split]
    d_validation = [x for x in d_validation if x["difficulty"] in difficulties_allowed]
    for exclude_regex in exclude_regexes:
        d_validation = [x for x in d_validation if not re.match(exclude_regex, x["question"])]
    only_keep_prompts_validation = set(di["prompt"] for di in tqdm.tqdm(d_validation, desc=f"Loading {split} set prompts"))
    prompts = list(only_keep_prompts_validation)
    if include_all_prompts:
        hydrating_dicts = [get_hydrating_dict(di) for di in tqdm.tqdm(d_validation, desc="Getting hydrating dicts")]
        j2_templates: List[jinja2.Template] = get_augmenting_templates(verbose=False)
        assert len(j2_templates) > 0
        for hydrating_dict, j2_template in tqdm.tqdm(
            itertools.product(hydrating_dicts, j2_templates),
            desc="Rendering hydrating dicts",
            total=len(hydrating_dicts) * len(j2_templates),
        ):
            prompt = j2_template.render(hydrating_dict)
            prompts.append(prompt)
    assert isinstance(prompts, list)
    assert all(isinstance(prompt, str) for prompt in prompts)
    assert len(prompts) > 0
    random.seed(random_seed)
    random.shuffle(prompts)
    if len(prompts) < minimum_samples:
        raise ValueError(f"Found {len(prompts)} prompts, but minimum_samples is {minimum_samples}")
    prompts = prompts[:maximum_samples]
    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    assert all(is_valid_chat(chat) for chat in chats), f"chats[0]: {chats[0]}"
    return chats


def load_validation_chats_0turn(
    **kwargs,
) -> List[List[Dict[str, str]]]:
    assert "split" not in kwargs
    kwargs["split"] = "validation"
    return load_split_chats_0turn(**kwargs)


def load_test_chats_0turn(
    **kwargs,
) -> List[List[Dict[str, str]]]:
    assert "split" not in kwargs
    kwargs["split"] = "test"
    return load_split_chats_0turn(**kwargs)


if __name__ == "__main__":
    ################ [BEGIN] DEBUGGING THE DATASETS VAL/TEST [BEGIN] ################
    # TODO(Adriano) in the future run a more assiduous data contamination validation
    # analysis to make SURE that there is no train/test contamination
    # TODO(Adriano) make sure to deal with non-solvable questions (our best solution to
    # this issue right now is to get rid of the ones which regex match known missing
    # information, such as "<image>")
    difficulties = {
        # Easiest! I manually inspect below (testing) that they are like seriously easy
        "easy",
        "EASY",
        "introductory",
        "0",
        0,
    }
    vchats = load_validation_chats_0turn(maximum_samples=1_000_000, difficulties_allowed=difficulties)
    tchats = load_test_chats_0turn(maximum_samples=1_000_000, difficulties_allowed=difficulties)
    print(f"Found up to {len(vchats)} validation chats and {len(tchats)} test chats")
    random.seed(57032)
    random.shuffle(vchats)
    assert all(is_valid_chat(chat) for chat in vchats), f"vchats[0]: {vchats[0]}"
    for chat in vchats[:1]:
        print(chat[0]["content"])
        print("=" * 100)
        input()
    print("+" * 100)
    vchats_aug = load_validation_chats_0turn(
        maximum_samples=1_000_000,
        difficulties_allowed=difficulties,
        include_all_prompts=True,
    )
    assert len(vchats_aug) > len(vchats), f"vchats_aug: {len(vchats_aug)}, vchats: {len(vchats)}"
    assert len(vchats_aug) % len(vchats) == 0, f"vchats_aug: {len(vchats_aug)}, vchats: {len(vchats)}"
    print(f"Found up to {len(vchats_aug)} validation chats with augmentation")
    random.seed(57032)
    random.shuffle(vchats_aug)
    for chat in vchats_aug[:3]:
        print(chat[0]["content"])
        print("=" * 100)
        input()
    print("+" * 100)
    ################ [END] DEBUGGING THE DATASETS VAL/TEST [END] ################
