from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict
import tqdm
from pathlib import Path
import json
import traceback
import random
from transformers import AutoTokenizer
import pandas as pd
import os
from typing import Dict, Set, Any, List, Optional
import pydantic

"""
This library provides utilities for creating a ready-to-use dataset for code SFT, PeFT,
SAE, and SAE-recovery (SFT, PeFT) training (as well as inference, etc...).

It generally allows you to:
1. Load contents from huggingface hub and local storage
2. Canonicalize the datsets into common entry formats
3. Generate prompts for the LLM from the dataset using hierarchical 
    templates/hyperparameters.
4. Get out a clean Dataset or DatasetDict object that you can use for the above purposes.
    NOTE: these returend objects are CLEAN by which we mean that they have the prompt already
    in the dataset in the `prompt` entry. This makes it easy to simply comperhension with
    `tokenizer.apply_chat_template` for your favorite models (we are using modern models,
    such as Qwen3, so they have the template built in).

Partly by Claude.

These are meant for the ORIGINAL dataset version.
"""


def json_loads_safe(s: str) -> Any:
    try:
        return json.loads(s)
    except ValueError as e:
        e_msg = str(e)
        if "Exceeds the limit" in e_msg and "for integer string conversion" in e_msg:
            return s  # string of digits
        raise e


def get_live_code_bench_dataset() -> Dataset:
    live_code_bench_location = Path("/Users/4gate/Downloads/live_code_bench")
    if not live_code_bench_location.exists():
        live_code_bench_location = Path("/mnt/align4_drive2/adrianoh/live_code_bench/")
        if not live_code_bench_location.exists():
            raise NotImplementedError("Live code bench location does not exist in two known locations... patch code? lmao")  # fmt: skip
    all_js = []
    locations = list(live_code_bench_location.glob("text*.jsonl"))
    for i, file in enumerate(locations):
        lines = file.read_bytes().splitlines()
        js = [json_loads_safe(line) for line in tqdm.tqdm(lines, desc=f"Loading {file.name} ({i+1}/{len(locations)})")]  # fmt: skip
        all_js.extend(js)
    return Dataset.from_list(all_js)


def load_partial_parquet_blobs_from_local(dataset_name: str, remove_columns: Optional[List[str]] = None) -> Dataset:
    dataset_name = "datasets/" + dataset_name
    dataset_name = dataset_name.replace("/", "--")
    datasets = []
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    dataset_dir = hf_home / "hub" / dataset_name / "snapshots"
    if not dataset_dir.exists():
        raise ValueError(f"You might not have downloaded the dataset? Dataset Directory={dataset_dir.resolve().as_posix()}")  # fmt: skip
    if len(list(dataset_dir.iterdir())) != 1:
        raise NotImplementedError("Only one snapshot at a time is supported...")
    dataset_dir = dataset_dir / next(dataset_dir.iterdir()).name / "data"
    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    print("\n".join(map(lambda x: x.name, parquet_files)))
    if len(parquet_files) == 0:
        raise ValueError("Cannot find any parquet files.Dataset Directory={dataset_dir.resolve().as_posix()}")  # fmt: skip
    for file in tqdm.tqdm(parquet_files, desc="Loading (partially loaded) parquets from local HF cache."):  # fmt: skip
        # Note: for soe reasson `Dataset.from_pandas` has some sort of parallelism
        # issue (or perhaps it was pandas concat) for large datasets. That's why instead
        # we convert to dataset and concat datasets instead of dataframes.
        datasets.append(Dataset.from_pandas(pd.read_parquet(file.resolve().as_posix())))
    # for dataset in datasets:
    #     print(dataset.column_names)
    column_names = set(datasets[0].column_names)
    assert all(set(dataset.column_names) == column_names for dataset in datasets), "All datasets must have the same column names"  # fmt: skip
    if remove_columns is not None:
        for dataset in datasets:
            # Some columns lead to merge issues :/
            dataset.remove_columns(remove_columns)
    return concatenate_datasets(datasets)


class DatasetEntry(pydantic.BaseModel):
    """
    A dataset entry will always be formatted like below for the prompt:

    # Overview
    <description of what to do, the problem, etc...>

    # Things to know
    <things to know about the problem, such as tags, etc...>

    # Starter code
    <starter code if available>

    # Problem statement
    <problem statement>

    # Output format
    <output format>

    => Then, this will be put into the prompt for our Qwen3 model.

    NOTE that the following things must be deduplicated:
    1. `question`
    2. `url`
    3. `question_id`
    """

    ################ Pre-preprocessing ################
    starter_code: Optional[str]
    things_to_know: Optional[str]
    source_description: str  # something like "codeforces.com" or whatever this offers
    source_dataset: str  # something like BAAI/TACO, codeparrot/apps, ... (HF dataset)
    question_id: str  # (modified/new) global question id (must 1:1 with dedup.)
    question: str | List[str]  # Can be a list of questions too! (for data augmentation)
    answers: List[str]
    # {
    #   'inputs': [...],
    #   'outputs': [...],
    # }
    # (meant for stdin/from stdin, string-checked
    # fn_type... => can store other values for keys
    expected_inputs_outputs: Optional[Dict[str, List[Any] | str]]

    # NOTE: this will depend on the dataset, to an extent!
    # (it could be EASY or easy or 0, 1, ... (etc...))
    difficulty: str
    url: Optional[str]

    ################ Post-preprocessing/generation/etc... ################
    full_generation: Optional[str]  # prompt + response if necessary
    # Define possible optoins on how to render the fields into a prompt
    prompt_hyperparamaters: Optional[Dict[str, Any]]
    prompt: Optional[str]
    response: Optional[str]
    parsed_response_code: Optional[str]
    gotten_inputs_outputs: Optional[Dict[str, List[Any]]]
    # {
    #   'inputs': [...],
    #   'outputs': [...],
    # }
    # (meant for stdin/from stdin, string-checked)
    metadata: Optional[Dict[str, Any]] = None

    def serialize(self) -> Dict[str, Any]:
        """
        HOTFIX For `pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values`
        error. Should be 1:1 with `deserialize` below.

        To solve problem, any list of dictionary object gets turned into a string (even
        if None; json serialize of None is 'null' which also deserializes to None)
        """
        return {
            "starter_code": self.starter_code,
            "things_to_know": self.things_to_know,
            "source_description": self.source_description,
            "source_dataset": self.source_dataset,
            "question_id": self.question_id,
            "question": self.question,
            "answers": json.dumps(self.answers),  # !!!
            "expected_inputs_outputs": json.dumps(self.expected_inputs_outputs),  # !!!
            "difficulty": self.difficulty,
            "url": self.url,
            "full_generation": self.full_generation,
            "prompt_hyperparamaters": json.dumps(self.prompt_hyperparamaters),  # !!!
            "prompt": self.prompt,
            "response": self.response,
            "parsed_response_code": self.parsed_response_code,
            "gotten_inputs_outputs": json.dumps(self.gotten_inputs_outputs),  # !!!
        }

    @staticmethod
    def deserialize(d: Dict[str, Any]) -> "DatasetEntry":
        """
        Undo the actions of `serialize` above.
        """
        return DatasetEntry(
            starter_code=d["starter_code"],
            things_to_know=d["things_to_know"],
            source_description=d["source_description"],
            source_dataset=d["source_dataset"],
            question_id=d["question_id"],
            question=d["question"],
            answers=json.loads(d["answers"]),  # !!!
            expected_inputs_outputs=json.loads(d["expected_inputs_outputs"]),  # !!!
            difficulty=d["difficulty"],
            url=d["url"],
            full_generation=d["full_generation"],
            prompt_hyperparamaters=json.loads(d["prompt_hyperparamaters"]),  # !!!
            prompt=d["prompt"],
            response=d["response"],
            parsed_response_code=d["parsed_response_code"],
            gotten_inputs_outputs=json.loads(d["gotten_inputs_outputs"]),  # !!!
        )


class PromptCreator:
    """
    This static class merely encapsulates the methods we use to create prompts. As is
    defined in `DatasetEntry`,

    # Overview
    <description of what to do, the problem, etc...>

    # Things to know
    <things to know about the problem, such as tags, etc...>

    # Starter code
    <starter code if available>

    # Problem statement
    <problem statement>

    # Output format
    <output format>

    Each of these markdown headers corresponds to a single block (the overview block,
    the things to know block, etc...). Each block is technically optional or can be
    hyperparameterized or may depend on the specific dataset (i.e. APPS vs. BAAI/TACO).
    """

    @staticmethod
    def dataset2prompt_baai_taco(entry: Dict[str, Any]) -> str:
        # NOTE: here is an example!
        #    "starter_code": "#User function Template for python3\n\nclass Solution:\n    def minTime (self, arr, n, k):\n        #code here\n        ",
        #     "input_output": "{\"inputs\": [\"n = 5\\nk = 3\\narr[] = {5,10,30,20,15}\", \"n = 4\\nk = 2\\narr[] = {10,20,30,40}\"], \"outputs\": [\"35\", \"60\"]}",
        #     "difficulty": "MEDIUM_HARD",
        #     "raw_tags": "['Algorithms', 'Searching', 'Binary Search', 'Divide and Conquer', 'Dynamic Programming']",
        #     "name": null,
        #     "source": "geeksforgeeks",
        #     "tags": "['Dynamic programming', 'Sorting', 'Divide and conquer', 'Complete search']",
        #     "skill_types": "['Dynamic programming', 'Sorting', 'Complete search']",
        #     "url": "https://practice.geeksforgeeks.org/problems/the-painters-partition-problem1535/1",
        #     "Expected Auxiliary Space": "O(1)",
        #     "time_limit": null,
        #     "date": null,
        #     "picture_num": "0",
        #     "memory_limit": null,
        #     "Expected Time Complexity": "O(n log m) , m = sum of all boards' length"
        # }

        # General information for these...
        tags = entry.get("tags", "")
        assert isinstance(tags, str), f"Tags must be a string, got {type(tags)}"  # fmt: skip
        tags = tags.strip()
        if len(tags) == 0:
            tags = entry.get("raw_tags", "")
            assert isinstance(tags, str), f"Tags must be a string, got {type(tags)}"  # fmt: skip
            tags = tags.strip()
            if len(tags) == 0:
                tags = entry.get("skill_types", "")
                assert isinstance(tags, str), f"Tags must be a string, got {type(tags)}"  # fmt: skip
                tags = tags.strip()
                if len(tags) == 0:
                    tags = "No tags available for this problem."
        tags_information = f"Some problems contain `tags` which elucidate the _types_ of solutions you might use (such as dynamic programming, binary search, greedy, etc...). This problem stipulates the following tags: {tags}"

        exp_aux_space = entry.get("Expected Auxiliary Space", "unknown")
        if exp_aux_space is None:
            exp_aux_space = "unknown"
        time_limit = entry.get("time_limit", "uknown")
        if time_limit is None:
            exp_aux_space = "unknown"
        memory_limit = entry.get("memory_limit", "unknown")
        if memory_limit is None:
            exp_aux_space = "unknown"
        exp_time_complex = entry.get("Expected Time Complexity", "unknown")
        if exp_time_complex is None:
            exp_aux_space = "unknown"
        time_space_complexity_prompt = f"""Remember to focus on making a correct solution without caring about performance. However, some problems do provide information about what sorts of runtimes and memory usages are possible. Use these _only_ as a hint to implement a functional and correct solution even if it is slow. The expected auxiliary space possible is {exp_aux_space}, and the expected runtime complexity is {exp_time_complex}. For the specific parameters stipulated below, the memory limit is: {memory_limit} and the time limit is: {time_limit}."""
        source = entry.get("source", "Unknown")
        name_information = entry.get("name", "This problem has no name.")

        # Starter code
        starter_code = entry.get("starter_code", "")
        assert isinstance(starter_code, str), f"Starter code must be a string, got {type(starter_code)}"  # fmt: skip
        starter_code = starter_code.strip()
        is_starter_code: bool = len(starter_code) > 0
        starter_code_prompt = f"""There {"is not any starter code." if not is_starter_code else 'is starter code in "starter code" markdown section. You may, but are not obligated to, use it as a starting point if you find it helpful.'}"""

        # Craft the prompt
        question = entry.get("question", "")
        assert isinstance(question, str), f"Question must be a string, got {type(question)}"  # fmt: skip
        question = question.strip()
        assert len(question) > 0
        prompt = f"""Below is a programming algorithms problem. Please solve it in python without using external tools nor a length explanation. Just write the code it, focusing on correctness (these should be fairly easy). Write clean code.

Below you will see:
1. Things to know about this problem (which could help you implement it quickly and effectively)
2. (Optionally) some starter code if evailable
3. The problem statement
4. Output format

Make sure to follow the instructions and use the hints for each section in your answer.

# Things to know
- Tags: {tags_information}
- Name: {name_information}
- Time/Space complexity information: {time_space_complexity_prompt}
- Problem Source: {source}
- {starter_code_prompt}
- Remember: while the time complexities might be stated above, they are only meant as a _hint_. Focus on implementing your function CORRECTLY and in simple, concise code without explaining or thinking for too long.

"""
        prompt += f"""# Starter Code
{starter_code if len(starter_code) > 0 else "Starter code is not available."}

"""
        prompt += f"""# Problem Statement
{question}

"""
        prompt += """# Output format
Please write your output delineated by triple tic-marks, like so in this pseudocode:

def <my function>(...) -> <annotations>:
    <my code>
<more code here>
etc... 

The user will use the python `.split` method to split your code on the triple ticmarks "" so make sure to use them properly to delineate your code once at the end of your output.

Implement the function and code in python. Do NOT think too much. Just implement the problem in the most simple way possible.
"""
        return prompt

    @staticmethod
    def dataset2prompt_codeparrot_apps(entry: Dict[str, Any]) -> str:
        # Extract metadata
        difficulty = entry.get("difficulty", "unknown")
        assert isinstance(difficulty, str), f"Difficulty must be a string, got {type(difficulty)}"  # fmt: skip
        difficulty = difficulty.strip()
        url = entry.get("url", "")
        assert isinstance(url, str), f"URL must be a string, got {type(url)}"  # fmt: skip
        url = url.strip()
        source = PromptCreator._get_codeparrot_apps_source(entry)

        # Parse inputs/outputs
        inputs_outputs_str = entry.get("input_output", None)
        if inputs_outputs_str is None:
            inputs_outputs_str = entry.get("input_outputs", None)
            if inputs_outputs_str is None:
                inputs_outputs_str = entry.get("inputs_outputs", None)
                if inputs_outputs_str is None:
                    inputs_outputs_str = entry.get("inputs_output", None)
        assert inputs_outputs_str is not None, f"Inputs outputs must be a dict, got {type(inputs_outputs)} =>\n\n{inputs_outputs}"  # fmt: skip
        try:
            inputs_outputs = json_loads_safe(inputs_outputs_str) if isinstance(inputs_outputs_str, str) else inputs_outputs_str
            assert isinstance(inputs_outputs, dict), f"Inputs outputs must be a dict, got {type(inputs_outputs)} =>\n\n{inputs_outputs}"  # fmt: skip
            assert "inputs" in inputs_outputs and "outputs" in inputs_outputs, f"Inputs outputs must have 'inputs' and 'outputs' keys, got {inputs_outputs.keys()}\n\n{entry}"  # fmt: skip
            assert isinstance(inputs_outputs["inputs"], list) and isinstance(inputs_outputs["outputs"], list), f"Inputs and outputs must be lists, got {type(inputs_outputs['inputs'])} and {type(inputs_outputs['outputs'])}"  # fmt: skip
            assert len(inputs_outputs["inputs"]) == len(inputs_outputs["outputs"]), f"Inputs and outputs must have the same length, got {len(inputs_outputs['inputs'])} and {len(inputs_outputs['outputs'])}"  # fmt: skip
        except Exception as e:
            raise e
            inputs_outputs = {"inputs": [], "outputs": []}

        # Get question
        question = entry.get("question", "")
        assert isinstance(question, str), f"Question must be a string, got {type(question)}"  # fmt: skip
        question = question.strip()
        assert len(question) > 0, "Question cannot be empty"

        # Build prompt
        prompt = f"""Below is a programming algorithms problem. Please solve it in python without using external tools nor a lengthy explanation. Just write the code, focusing on correctness. Write clean code.

Below you will see:
1. Things to know about this problem (which could help you implement it quickly and effectively)
2. The problem statement
3. Output format

Make sure to follow the instructions and use the hints for each section in your answer.

# Things to know
- Difficulty: {difficulty}
- Problem Source: {source}
- URL: {url if url else "No URL available"}
- This problem has {len(inputs_outputs.get("inputs", []))} test case(s) available for validation
- Remember: Focus on implementing your solution CORRECTLY and in simple, concise code without explaining or thinking for too long.

"""
        prompt += f"""# Problem Statement
{question}

"""
        prompt += """# Output format
Please write your output delineated by triple backticks (```), like so:

```python
def solution(...):
    # your code here
    pass

# any additional code if needed
```

The user will parse the code between the triple backticks, so make sure to use them properly to delineate your code.

Implement the solution in python. Do NOT think too much. Just implement the problem in the most simple way possible.
"""
        return prompt

    @staticmethod
    def dataset2prompt_deepmind_code_contests(entry: Dict[str, Any]) -> str:
        # Extract metadata
        name = entry.get("name", "Unnamed problem")
        assert isinstance(name, str), f"Name must be a string, got {type(name)}"  # fmt: skip
        name = name.strip()
        # Source is an int, not sure why---ignore!
        difficulty = str(entry.get("difficulty", "unknown"))

        # Extract Codeforces-specific info
        cf_tags = entry.get("cf_tags", [])
        cf_rating = entry.get("cf_rating", "unknown")
        time_limit = entry.get("time_limit", "unknown")
        memory_limit_bytes = entry.get("memory_limit_bytes", "unknown")

        # Format tags
        tags_str = ", ".join(cf_tags) if cf_tags else "No tags available"

        # Format limits
        memory_limit_mb = memory_limit_bytes / (1024 * 1024) if isinstance(memory_limit_bytes, (int, float)) else "unknown"

        # Get problem description
        description = entry.get("description", "").strip()
        assert len(description) > 0, "Description cannot be empty"

        # Build prompt
        prompt = f"""Below is a programming algorithms problem. Please solve it in python without using external tools nor a lengthy explanation. Just write the code, focusing on correctness. Write clean code.

Below you will see:
1. Things to know about this problem (which could help you implement it quickly and effectively)
2. The problem statement
3. Output format

Make sure to follow the instructions and use the hints for each section in your answer.

# Things to know
- Problem Name: {name}
- Difficulty Level: {difficulty}
- Codeforces Rating: {cf_rating}
- Tags: {tags_str}
- Time Limit: {time_limit}
- Memory Limit: {memory_limit_mb} MB
- Remember: While the complexity hints are provided, focus on implementing your solution CORRECTLY first.

"""
        prompt += f"""# Problem Statement
{description}

"""
        prompt += """# Output format
Please write your output delineated by triple backticks (```), like so:

```python
def solution(...):
    # your code here
    pass

# any additional code if needed
```

The user will parse the code between the triple backticks, so make sure to use them properly to delineate your code.

Implement the solution in python. Do NOT think too much. Just implement the problem in the most simple way possible.
"""
        return prompt

    @staticmethod
    def dataset2prompt_live_code_bench(entry: Dict[str, Any]) -> str:
        # Extract metadata
        question_title = entry.get("question_title", "Untitled").strip()
        platform = entry.get("platform", "unknown").strip()
        difficulty = entry.get("difficulty", "unknown").strip()
        question_id = entry.get("question_id", "unknown").strip()
        contest_id = entry.get("contest_id", "").strip()

        # Get starter code
        starter_code = entry.get("starter_code", "").strip()
        has_starter_code = len(starter_code) > 0

        # Get question content
        question_content = entry.get("question_content", "").strip()
        assert len(question_content) > 0, "Question content cannot be empty"

        # Get test cases count
        public_test_cases = entry.get("public_test_cases", [])
        test_count = len(public_test_cases)

        # Build prompt
        prompt = f"""Below is a programming algorithms problem. Please solve it in python without using external tools nor a lengthy explanation. Just write the code, focusing on correctness. Write clean code.

Below you will see:
1. Things to know about this problem (which could help you implement it quickly and effectively)
2. {"Starter code" if has_starter_code else "(No starter code available)"}
3. The problem statement
4. Output format

Make sure to follow the instructions and use the hints for each section in your answer.

# Things to know
- Problem Title: {question_title}
- Platform: {platform}
- Difficulty: {difficulty}
- Contest ID: {contest_id if contest_id else "No contest ID"}
- Question ID: {question_id}
- This problem has {test_count} public test case(s) available
- {"Starter code is provided below" if has_starter_code else "No starter code is provided for this problem"}
- Remember: Focus on implementing your solution CORRECTLY and in simple, concise code.

"""
        if has_starter_code:
            prompt += f"""# Starter Code
{starter_code}

"""
        prompt += f"""# Problem Statement
{question_content}

"""
        prompt += """# Output format
Please write your output delineated by triple backticks (```), like so:

```python
def solution(...):
    # your code here
    pass

# any additional code if needed
```

The user will parse the code between the triple backticks, so make sure to use them properly to delineate your code.

Implement the solution in python. Do NOT think too much. Just implement the problem in the most simple way possible.
"""
        return prompt

    @staticmethod
    def _get_codeparrot_apps_source(entry: Dict[str, Any]) -> str:
        # Extract source URL from entry
        url = entry.get("url", "")
        if not url:
            return "Unknown source"

        # Clean up the URL to get just the domain
        cleaned_url = url.replace("https://", "").replace("http://", "")
        domain = cleaned_url.split("/", 1)[0]
        expected_domains = set(
            [
                "leetcode.com",
                "atcoder.jp",
                "www.hackerrank.com",
                "codeforces.com",
                "www.codewars.com",
                "www.codechef.com",
                "open.kattis.com",
            ]
        )
        assert domain in expected_domains, f"Unknown domain: {domain}"
        return domain


class DatasetMerger:
    """
    This will merge the datasets. To do so it will canonicalize these problems into the
    datset entry class format above.
    """

    def __init__(
        self,
        name2difficulties: Dict[str, Set[str | int]] = {
            "livecodebench/code_generation_lite": {"easy", "medium"},
            "deepmind/code_contests": set(range(7)),
            "codeparrot/apps": {"introductory"},
            "BAAI/TACO": {"EASY", "MEDIUM", "MEDIUM_HARD"},
        },
        splits_fracs: Dict[str, float] = {
            "train": 0.8,
            "test": 0.1,
            "validation": 0.1,
        },
        splits_fracs_min_n: Dict[str, int] = {
            "train": 18_000,
            "test": 20,
            "validation": 20,
        },
    ):
        self.name2difficulties = name2difficulties
        self.splits_fracs = splits_fracs
        self.splits_fracs_min_n = splits_fracs_min_n
        assert set(self.splits_fracs.keys()) == set(self.splits_fracs_min_n.keys()), f"All splits must have a minimum number of samples, got {self.splits_fracs.keys()} and {self.splits_fracs_min_n.keys()}"  # fmt: skip
        assert abs(sum(self.splits_fracs.values()) - 1.0) < 1e-6, f"All splits must sum to 1.0, got {self.splits_fracs.values()}"  # fmt: skip

    ################ [BEGIN] Getters for raw datasets be4 merge [BEGIN] ################
    def _get_baai_taco_raw_datset(self) -> Dataset:
        """
        # Relevant Columns:
        ## Relevant identifiers and metadata:
        - url: Problem URL
        - starter_code:  Template code to start with
        - difficulty: Problem difficulty (e.g., "EASY", "MEDIUM", "MEDIUM_HARD")
            (this is used to filter problems for the dataset)

        ### Identification Hints
        These are like `url` but they can be used by the LLM to better identify what the
        solution might look like (i.e. some sources tend to skew towards certain types of
        problems/solutions).
        - name: Problem name (can be null)
        - source: Source platform (e.g., "geeksforgeeks")

        ### Tags Columns
        These give hints about the types of algorithms to try:
        - raw_tags: List of algorithm types as string
        - tags: Processed algorithm tags
        - skill_types: Alternative tag format

        ### Complexity Columns
        These give hints about basically what data structures/etc... you may use, what
        is possible, etc...
        - Expected Auxiliary Space: Space complexity hint
        - time_limit: Time constraint (can be null)
        - memory_limit: Memory constraint (can be null)
        - Expected Time Complexity: Time complexity hint

        ## Actual contents:
        - question: The actual problem statement
        - solutions: List of solutions to the problem (list of strings)
        - input_output: JSON string with inputs and expected outputs (should be a dict)

        `input_output` is formatted like for Codeparrot (and also json-serialized):

        {
            'inputs': ['<input1>', '<input2>', '<input3>', ...],
            'outputs': ['<output1>', '<output2>', '<output3>', ...]
        }


        These input/output pairs are what you would feed into the comandline of the
        LLM's generated code's process (you would run it in a process).
        """
        return concatenate_datasets(
            [
                load_dataset("BAAI/TACO", split="train", trust_remote_code=True),
                load_dataset("BAAI/TACO", split="test", trust_remote_code=True),
            ]
        )

    def _get_codeparrot_apps_raw_dataset(self) -> Dataset:
        """
        # Relevant Columns:
        ## Relevant identifiers and metadata:
        - url: Problem URL (identifies the problem across datasets when merging, helps
            identify the source, etc...)
        - problem_id: Problem ID (used for identification only)
        - difficulty: Problem difficulty (e.g., "introductory") (used to filter problems
            for the dataset)

        ## Contents
        - question: The actual problem statement
        - solutions: List of solutions to the problem (list of strings)
        - inputs_outputs: JSON string with inputs and expected outputs (should be a dict)
            (use this to verify LLM-generated output correctness)

        `inputs_outputs` will be formatted like:

        {
            'inputs': ['<input1>', '<input2>', '<input3>', ...],
            'outputs': ['<output1>', '<output2>', '<output3>', ...]
        }

        (where these inputs/outputs should be lists). Moreover, these need to be json
        decoded/deserialized from string into python objects.
        """
        return concatenate_datasets(
            [
                load_dataset("codeparrot/apps", split="train", trust_remote_code=True),
                load_dataset("codeparrot/apps", split="test", trust_remote_code=True),
            ]
        )

    def _get_live_code_bench_raw_dataset(self) -> Dataset:
        """
        # Relevant Columns:
        ## Metadata and identification
        - question_title: Title of the coding problem (may or may not be informative)
        - platform: Source platform (e.g., "codeforces", "atcoder")
            (we can use this platform value to inform the LLM about the likely style)
        - question_id: Unique identifier for the problem
        - contest_id: Contest identifier where the problem appeared
        - contest_date: Date of the contest (not really used but identifier)
        - difficulty: Problem difficulty, usually 'easy' or 'medium' or 'hard' etc...
        - metadata: Additional metadata about the problem

        ## Useful for LLM
        - starter_code: Initial code template or boilerplate (if available), string

        ## Contents
        - question_content: The actual problem statement/description, string
        - public_test_cases: Test cases for validation (note: private test cases not
            used)

        NOTE: This dataset does not contain reference solutions.

        The public test cases are formatted as follows (by example):

        [
            {'input': '5 2 3\n', 'output': '1 3 2 4 5\n', 'testtype': 'stdin'},
            {'input': '7 1 1\n', 'output': '1 2 3 4 5 6 7\n', 'testtype': 'stdin'},
            {'input': '10 1 10\n', 'output': '10 9 8 7 6 5 4 3 2 1\n','testtype': 'stdin'}
        ]

        It will be reformatted into the format of the others:

        {
            'inputs': [...],
            'outputs': [...],
        }

        (all tests are for stdin into the docker container's running process)
        """
        return get_live_code_bench_dataset()

    def _get_deepmind_code_contests_raw_dataset(self) -> Dataset:
        """
        # Relevant Columns:
        ## Metadata and identification
        - name: Problem name/title
        - source: Source platform (e.g., "codeforces")
        - difficulty: Problem difficulty rating/level

        ### Definite identifiers
        - cf_contest_id: (Codeforces) contest identifier
        - cf_index: (Codeforces) problem index within contest
        - cf_points: (Codeforces) point value for the problem

        ### Informational hints
        - cf_rating: (Codeforces) problem rating (difficulty metric)
        - cf_tags: (Codeforces) problem tags/categories (may or may not be useful to the LLM)
        - time_limit: time limit for execution
        - memory_limit_bytes: memory limit in bytes

        ### Not clear what these are and they are ignored...
        - input_file: Input file specification (ignored for now)
        - output_file: Output file specification (ignored for now)
        (TODO(Adriano) maybe just filter out for the non-'' ones here...)

        ## Contents
        - description: Problem statement/description (btw there may or may not be "Translation"
            possibly from another language?? we ignore 'is_description_translated' and
            'untranslated_description')
        - public_tests: Public test cases for validation (private tests not used, generated_tests not used)
        - solutions: Reference solutions (incorrect solutions provided and not used)
        """
        # Unfort time limit can yield problems :/
        # ValueError: The features can't be aligned because the key nanos of features
        #   {'nanos': Value(dtype='float64', id=None), 'seconds': Value(dtype='float64',
        #   id=None)} has unexpected type - Value(dtype='float64', id=None) (expected
        #   either Value(dtype='int64', id=None) or Value("null").
        # return load_partial_parquet_blobs_from_local(
        #     "deepmind/code_contests",
        #     remove_columns=[
        #         "time_limit",
        #     ],
        # )
        data = concatenate_datasets(
            [
                load_dataset("deepmind/code_contests", split="train", trust_remote_code=True),
                load_dataset("deepmind/code_contests", split="test", trust_remote_code=True),
                load_dataset("deepmind/code_contests", split="valid", trust_remote_code=True),
            ]
        )
        data.remove_columns(["time_limit"])
        return data

    ################ [END] Getters for raw datasets be4 merge [END] ################

    ################ [BEGIN] Helpers for different steps [BEGIN] ################
    def _filter_difficulty(self, dataset: Dataset, dataset_name: str) -> Dataset:
        return dataset.filter(lambda x: x["difficulty"] in self.name2difficulties[dataset_name])

    def _convert_to_dataset_entry(
        self,
        dataset: Dataset,
        dataset_name: str,
        prompt_creator: PromptCreator,
    ) -> Dataset:
        """Convert raw dataset entries to DatasetEntry format"""
        entries = []
        n_missed = 0
        for idx, entry in enumerate(tqdm.tqdm(dataset, desc=f"Converting {dataset_name} to DatasetEntry format")):
            try:
                if dataset_name == "BAAI/TACO":
                    # Parse inputs/outputs
                    inputs_outputs_str = entry.get("input_output", "{}")
                    try:
                        inputs_outputs = json_loads_safe(inputs_outputs_str) if isinstance(inputs_outputs_str, str) else inputs_outputs_str
                        assert isinstance(inputs_outputs, dict), f"Inputs outputs must be a dict, got {type(inputs_outputs)} =>\n\n{inputs_outputs}"  # fmt: skip
                        assert "inputs" in inputs_outputs and "outputs" in inputs_outputs, f"Inputs outputs must have 'inputs' and 'outputs' keys, got {inputs_outputs.keys()}"  # fmt: skip
                        assert isinstance(inputs_outputs["inputs"], list) and isinstance(inputs_outputs["outputs"], list), f"Inputs and outputs must be lists, got {type(inputs_outputs['inputs'])} and {type(inputs_outputs['outputs'])}"  # fmt: skip
                        assert len(inputs_outputs["inputs"]) == len(inputs_outputs["outputs"]), f"Inputs and outputs must have the same length, got {len(inputs_outputs['inputs'])} and {len(inputs_outputs['outputs'])}"  # fmt: skip
                    except Exception as e:
                        raise e
                        inputs_outputs = {"inputs": [], "outputs": []}

                    dataset_entry = DatasetEntry(
                        starter_code=entry.get("starter_code", "").strip() or None,
                        things_to_know=f"Tags: {entry.get('tags', [])}",
                        source_description=str(entry.get("source", "unknown")),
                        source_dataset=dataset_name,
                        question_id=f"{dataset_name}_{entry.get('url', f'idx_{idx}')}",
                        question=entry["question"].strip(),
                        answers=json_loads_safe(entry.get("solutions", "[]")),
                        expected_inputs_outputs=inputs_outputs,
                        difficulty=str(entry.get("difficulty", "unknown")),
                        url=entry.get("url", None),
                        # Inputs to models
                        prompt=prompt_creator.dataset2prompt_baai_taco(entry),
                        # Outputs from models (not yet computed)
                        full_generation=None,
                        response=None,
                        parsed_response_code=None,
                        gotten_inputs_outputs=None,
                        prompt_hyperparamaters=None,
                    )

                elif dataset_name == "codeparrot/apps":
                    # Parse inputs/outputs
                    inputs_outputs_str = entry.get("input_output", "{'inputs': [], 'outputs': []}")
                    if isinstance(inputs_outputs_str, str):
                        inputs_outputs_str = inputs_outputs_str.strip()
                        if inputs_outputs_str == "":
                            inputs_outputs_str = "{}"
                        else:
                            inputs_outputs_str = json_loads_safe(inputs_outputs_str)
                    try:
                        inputs_outputs = json_loads_safe(inputs_outputs_str) if isinstance(inputs_outputs_str, str) else inputs_outputs_str
                        assert isinstance(inputs_outputs, dict), f"Inputs outputs must be a dict, got {type(inputs_outputs)} =>\n\n{inputs_outputs}"  # fmt: skip
                        assert "inputs" in inputs_outputs and "outputs" in inputs_outputs, f"Inputs outputs must have 'inputs' and 'outputs' keys, got {inputs_outputs.keys()}"  # fmt: skip
                        assert isinstance(inputs_outputs["inputs"], list) and isinstance(inputs_outputs["outputs"], list), f"Inputs and outputs must be lists, got {type(inputs_outputs['inputs'])} and {type(inputs_outputs['outputs'])}"  # fmt: skip
                        assert len(inputs_outputs["inputs"]) == len(inputs_outputs["outputs"]), f"Inputs and outputs must have the same length, got {len(inputs_outputs['inputs'])} and {len(inputs_outputs['outputs'])}"  # fmt: skip
                    except Exception as e:
                        raise e
                        inputs_outputs = {"inputs": [], "outputs": []}

                    dataset_entry = DatasetEntry(
                        starter_code=None,  # APPS doesn't have starter code
                        things_to_know=f"Difficulty: {entry.get('difficulty', 'unknown')}",
                        source_description=prompt_creator._get_codeparrot_apps_source(entry),
                        source_dataset=dataset_name,
                        question_id=f"{dataset_name}_{entry.get('url', f'idx_{idx}')}",
                        question=entry["question"].strip(),
                        answers=json_loads_safe(entry.get("solutions", "[]")),
                        expected_inputs_outputs=inputs_outputs,
                        difficulty=str(entry.get("difficulty", "unknown")),
                        url=entry.get("url", None),
                        # Inputs to models
                        prompt=prompt_creator.dataset2prompt_codeparrot_apps(entry),
                        # Outputs from models (not yet computed)
                        full_generation=None,
                        response=None,
                        parsed_response_code=None,
                        gotten_inputs_outputs=None,
                        prompt_hyperparamaters=None,
                    )

                elif dataset_name == "livecodebench/code_generation_lite":
                    # Convert test cases format
                    public_test_cases = json_loads_safe(entry.get("public_test_cases", "[]"))
                    inputs_outputs = {
                        "inputs": [tc["input"] for tc in public_test_cases],
                        "outputs": [tc["output"] for tc in public_test_cases],
                    }

                    dataset_entry = DatasetEntry(
                        starter_code=entry.get("starter_code", "").strip() or None,
                        things_to_know=f"Platform: {entry.get('platform', 'unknown')}",
                        source_description=str(entry.get("platform", "unknown")),
                        source_dataset=dataset_name,
                        question_id=f"{dataset_name}_{entry.get('question_id', f'idx_{idx}')}",
                        question=entry.get("question_content", "").strip(),
                        answers=[],  # No solutions in this dataset
                        expected_inputs_outputs=inputs_outputs,
                        difficulty=str(entry.get("difficulty", "unknown")),
                        url=None,  # No URL in this dataset
                        # Inputs to models
                        prompt=prompt_creator.dataset2prompt_live_code_bench(entry),
                        # Outputs from models (not yet computed)
                        full_generation=None,
                        response=None,
                        parsed_response_code=None,
                        gotten_inputs_outputs=None,
                        prompt_hyperparamaters=None,
                    )

                elif dataset_name == "deepmind/code_contests":
                    # Convert test cases format
                    inputs_outputs = entry.get("public_tests", {"inputs": [], "outputs": []})

                    # Build URL if codeforces
                    url = None
                    if entry.get("source") == "codeforces" and entry.get("cf_contest_id"):
                        cf_contest_id = entry.get("cf_contest_id")
                        cf_index = entry.get("cf_index", "")
                        url = f"https://codeforces.com/contest/{cf_contest_id}/problem/{cf_index}"

                    answers = []
                    solutions = entry.get("solutions", {"solution": [], "language": []})
                    assert len(solutions['solution']) == len(solutions['language']), f"Solutions and languages must have the same length, got {len(solutions['solution'])} and {len(solutions['language'])}"  # fmt: skip
                    assert "description" in entry and isinstance(entry["description"], str), f"Description must be a string, got {type(entry['description'])}"  # fmt: skip
                    for solution, language in zip(solutions["solution"], solutions["language"]):
                        if language == 1:
                            answers.append(solution)
                    dataset_entry = DatasetEntry(
                        starter_code=None,  # Code contests doesn't have starter code
                        things_to_know=f"CF Tags: {entry.get('cf_tags', [])}",
                        source_description=str(entry.get("source", "unknown")),
                        source_dataset=dataset_name,
                        question_id=f"{dataset_name}_{entry.get('name', f'idx_{idx}')}",
                        question=entry["description"].strip(),
                        answers=answers,
                        expected_inputs_outputs=inputs_outputs,
                        difficulty=str(entry.get("difficulty", "unknown")),
                        url=url,
                        # Inputs to models
                        prompt=prompt_creator.dataset2prompt_deepmind_code_contests(entry),
                        # Outputs from models (not yet computed)
                        full_generation=None,
                        response=None,
                        parsed_response_code=None,
                        gotten_inputs_outputs=None,
                        prompt_hyperparamaters=None,
                    )

                else:
                    raise ValueError(f"Unknown dataset name: {dataset_name}")

                entries.append(dataset_entry.serialize())

            except Exception as e:
                print(f"Error converting entry {idx} from {dataset_name}: {e}")
                traceback.print_exc()
                n_missed += 1
                continue
        assert len(entries) + n_missed == len(dataset), f"Total entries must be the same as the dataset, got {len(entries) + n_missed} and {len(dataset)}"  # fmt: skip
        print(f"Total missed entries: {n_missed} / {len(dataset)} = {n_missed / len(dataset)}")

        assert isinstance(entries, list), f"Entries must be a list, got {type(entries)}"  # fmt: skip
        assert all(isinstance(entry, dict) for entry in entries), f"All entries must be dicts, got {type(entries[0])}"  # fmt: skip
        # print(entries[0])
        return Dataset.from_list(entries)

    def _deduplicate_dataset_entries(self, dataset: Dataset) -> Dataset:
        """Deduplicate dataset entries based on question, url, and question_id"""
        seen_questions = set()
        seen_urls = set()
        seen_question_ids = set()

        deduplicated_entries = []

        for entry in tqdm.tqdm(dataset, desc="Deduplicating dataset entries"):
            question = entry.get("question", "").strip()
            url = entry.get("url", None)
            question_id = entry.get("question_id", "")

            # Check if we've seen this before
            if question in seen_questions:
                continue
            if url and url in seen_urls:
                continue
            if question_id in seen_question_ids:
                continue

            # Add to seen sets
            if question:
                seen_questions.add(question)
            if url:
                seen_urls.add(url)
            if question_id:
                seen_question_ids.add(question_id)

            # Keep this entry
            deduplicated_entries.append(entry)

        print(f"Deduplicated from {len(dataset)} to {len(deduplicated_entries)} entries")
        return Dataset.from_list(deduplicated_entries)

    def _split_dataset(
        self,
        dataset: Dataset,
        splits_fracs: Optional[Dict[str, float]] = None,
        splits_fracs_min_n: Optional[Dict[str, int]] = None,
    ) -> DatasetDict:
        """Split dataset into train/test/validation sets"""
        split_fracs = splits_fracs or self.splits_fracs
        split_fracs_min_n = splits_fracs_min_n or self.splits_fracs_min_n

        # Validate inputs
        assert set(split_fracs.keys()) == set(split_fracs_min_n.keys()), "All splits must have a minimum number of samples"
        assert abs(sum(split_fracs.values()) - 1.0) < 1e-6, f"All splits must sum to 1.0, got {sum(split_fracs.values())}"

        # Shuffle dataset
        dataset = dataset.shuffle(seed=82048)
        total_size = len(dataset)

        # Calculate split sizes
        split_sizes = {}
        cumulative = 0
        for split_name, frac in split_fracs.items():
            size = int(total_size * frac)
            if size < split_fracs_min_n[split_name]:
                raise ValueError(
                    f"Split {split_name} is too small ({size} samples) for minimum requirements ({split_fracs_min_n[split_name]} needed)"
                )
            split_sizes[split_name] = size
            cumulative += size

        # Check if we have enough data
        if cumulative > total_size:
            raise ValueError(
                f"Dataset too small ({total_size} samples) for minimum requirements ({cumulative} needed); split sizes:\n\n{json.dumps(split_sizes, indent=4)}"
            )

        # Adjust last split to use all remaining data
        split_names = list(split_fracs.keys())
        if cumulative < total_size:
            split_sizes[split_names[-1]] += total_size - cumulative

        # Create splits
        dataset_dict = {}
        start_idx = 0
        for split_name in split_names:
            end_idx = start_idx + split_sizes[split_name]
            dataset_dict[split_name] = dataset.select(range(start_idx, min(end_idx, total_size)))
            start_idx = end_idx

        # Print split info
        for split_name, split_dataset in dataset_dict.items():
            print(f"{split_name}: {len(split_dataset)} samples")

        return DatasetDict(dataset_dict)

    ################ [END] Helpers for different steps [END] ################

    ################ [BEGIN] Main method for this class [BEGIN] ################

    def create_combined_dataset(self) -> DatasetDict:
        """
        Creates a combined dataset and return a DatasetDict that you can save to disk
        for future usage. It is designed to be split among many files so that it can fit
        in github commits if necessary.

        This should follow roughly the following steps:
        1. Load each dataset (just combine the test, train, valid into one mega-dataset
            like above here... we will split out the canonicalized/merged dataset later
            into better splits).
        2. For each dataset, filter for the difficulty level in `self.name2difficulties`
        3. For each dataset, convert it into a list of `DatasetEntry` objects. You can
            just call the abstractions in `DatasetEntry` and PromptCreator class to do
            this.
        4. Deduplicate (basically iterates through and just keeps only one out of the
            entries that share any of the "deduplication" fields (question, url,
            question_id). By convention, always pick the first available entry.
        5. Turn this into a dataset of Dict objects by model_dumping the entries and
            return the new `Dataset` object.
        6. Randomly shuffle it and select the fractions in `self.splits_fracs` and then
            assert that at least `splits_fracs_min_n[i]` are in each of the splits [i].
        7. Return the DatasetDict.
        """
        prompt_creator = PromptCreator()
        all_dataset_entries = []

        # Define dataset loaders and their names
        dataset_loaders = {
            "BAAI/TACO": self._get_baai_taco_raw_datset,
            "codeparrot/apps": self._get_codeparrot_apps_raw_dataset,
            "livecodebench/code_generation_lite": self._get_live_code_bench_raw_dataset,
            "deepmind/code_contests": self._get_deepmind_code_contests_raw_dataset,
        }
        # TODO(Adriano) what is desired behavior?
        if set(dataset_loaders.keys()) != set(self.name2difficulties.keys()):
            print("WARNING: Dataset loaders keys do not match name2difficulties keys")
        dataset_loaders = {k: v for k, v in dataset_loaders.items() if k in self.name2difficulties}

        # Process each dataset
        for dataset_name, loader_fn in dataset_loaders.items():
            print(f"\n{'=' * 50}")
            print(f"Processing {dataset_name}")
            print(f"{'=' * 50}")

            try:
                # Step 1: Load dataset
                print(f"Loading {dataset_name}...")
                raw_dataset = loader_fn()
                print(f"Loaded {len(raw_dataset)} raw entries")

                # Step 2: Filter by difficulty
                assert dataset_name in self.name2difficulties, f"Dataset name {dataset_name} not in {self.name2difficulties.keys()}"  # fmt: skip
                print(f"Filtering dataset {dataset_name} by difficulty: {self.name2difficulties[dataset_name]}")
                filtered_dataset = self._filter_difficulty(raw_dataset, dataset_name)
                print(f"Filtered dataset {dataset_name} to {len(filtered_dataset)} entries")

                # Step 3: Convert to DatasetEntry format
                print("Converting to DatasetEntry format...")
                converted_dataset = self._convert_to_dataset_entry(filtered_dataset, dataset_name, prompt_creator)
                print(f"Converted {len(converted_dataset)} entries")

                all_dataset_entries.append(converted_dataset)

            except Exception as e:
                # TODO
                # print(f"Error processing {dataset_name}: {e}")
                # import traceback

                # traceback.print_exc()
                # continue
                raise e

        # Step 4: Concatenate all datasets
        print(f"\n{'=' * 50}")
        print("Concatenating all datasets...")
        combined_dataset = concatenate_datasets(all_dataset_entries)
        print(f"Combined dataset has {len(combined_dataset)} entries")

        # Step 5: Deduplicate
        print("\nDeduplicating...")
        print(f"Size of combined dataset: {len(combined_dataset)}")
        deduplicated_dataset = self._deduplicate_dataset_entries(combined_dataset)

        # Step 6: Split into train/test/validation
        print("\nSplitting dataset...")
        print(f"Size of deduplicated dataset: {len(deduplicated_dataset)}")
        dataset_dict = self._split_dataset(deduplicated_dataset)

        print(f"\n{'=' * 50}")
        for split_name, split_dataset in dataset_dict.items():
            print(f"{split_name}: {len(split_dataset)} samples")
        print("Dataset creation complete!")
        print(f"{'=' * 50}")

        return dataset_dict

    ################ [END] Main method for this class [END] ################


if __name__ == "__main__":
    #### TESTING ####
    creator = PromptCreator()  # all defaults ehe
    dataset = load_dataset("BAAI/TACO", split="train")
    prompts = [creator.dataset2prompt_baai_taco(entry) for entry in tqdm.tqdm(dataset, desc="Converting to prompts for LLM.")]
    random.shuffle(prompts)
    for i in range(10):
        print("=" * 100)
        print(prompts[i])
        print("\n\n")
        print("=" * 100)

    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    chat_templatted_prompts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    print(chat_templatted_prompts[0])
