from __future__ import annotations


from typing import List, Dict
import random
import json
import copy
from pathlib import Path
from utils.llm_judge.api_generate import load_jinja_template
import jinja2

"""
Thisty module is menat to help you basically load a bunch of additional prompts for
any single prompt/dataset entry you have.
"""


def get_augmenting_templates(verbose: bool = True) -> List[jinja2.Template]:
    files = list((Path(__file__).parent.parent / "dataset_lib" / "augmentation" / "initial_templates").glob("*.j2"))
    files += list((Path(__file__).parent.parent / "dataset_lib" / "augmentation" / "initial_templates").glob("*.jinja2"))
    files = [f for f in files if f.is_file() and f.stem != "dummy"]
    if verbose:
        print(f"Found the following ({len(files)}) files: for j2 templates")
        print(" - " + "\n - ".join(f.name for f in files))
    templates = [load_jinja_template(file) for file in files]
    return templates


def get_hydrating_dict(dataset_entry: Dict[str, str]) -> Dict[str, str]:
    assert isinstance(dataset_entry, dict)
    assert "answers" in dataset_entry
    assert isinstance(dataset_entry["answers"], (list, str)), f"type={type(dataset_entry['answers'])}\n\n{dataset_entry['answers']}"
    if isinstance(dataset_entry["answers"], str):
        assert dataset_entry["answers"].startswith("[")
        dataset_entry["answers"] = json.loads(dataset_entry["answers"])
    assert isinstance(dataset_entry["answers"], list), f"type={type(dataset_entry['answers'])}\n\n{dataset_entry['answers']}"
    answer = random.choice(dataset_entry["answers"]) if len(dataset_entry["answers"]) > 0 else "No answer was provided... (oops!)"
    assert isinstance(answer, str), f"type={type(answer)}\n\n{answer}"
    inputs_outputs = dataset_entry["expected_inputs_outputs"]
    # BEGIN FIX
    if isinstance(inputs_outputs, str):
        inputs_outputs = json.loads(inputs_outputs)
    if "inputs" not in inputs_outputs:
        assert "input" in inputs_outputs
        inputs_outputs["inputs"] = inputs_outputs["input"]
        del inputs_outputs["input"]
    if "outputs" not in inputs_outputs:
        assert "output" in inputs_outputs
        inputs_outputs["outputs"] = inputs_outputs["output"]
        del inputs_outputs["output"]
    # END FIX
    assert isinstance(inputs_outputs, dict), f"inputs_outputs is not a dict: {type(inputs_outputs)}\n\n{inputs_outputs}"
    assert {"inputs", "outputs"}.issubset(set(inputs_outputs.keys())), f"inputs_outputs keys is {set(inputs_outputs.keys())}"
    assert len(inputs_outputs["inputs"]) == len(inputs_outputs["outputs"])
    inputs = [str(x) for x in inputs_outputs["inputs"]]
    outputs = [str(x) for x in inputs_outputs["outputs"]]
    indices = list(range(len(inputs)))
    random.shuffle(indices)
    max_examples_per_prompt = 10
    inputs = [inputs[i] for i in indices[:max_examples_per_prompt]]
    outputs = [outputs[i] for i in indices[:max_examples_per_prompt]]
    inputs_outputs_string = "\n".join(json.dumps({"input": i, "output": o}, indent=4) for i, o in zip(inputs, outputs))
    output = copy.deepcopy(dataset_entry)
    assert "answer" not in output
    assert "expected_inputs_outputs" in output  # overwrite
    output["answer"] = answer
    output["expected_inputs_outputs"] = inputs_outputs_string
    keys_to_keep = [
        "things_to_know",
        "source_description",
        "question",
        "answer",
        "expected_inputs_outputs",
    ]
    assert set(keys_to_keep).issubset(set(output.keys())), f"output keys is {set(output.keys())}"
    returnmeplz = {k: output[k] for k in keys_to_keep}
    _types_dict = {k: type(v) for k, v in returnmeplz.items()}
    assert all(isinstance(v, str) for v in returnmeplz.values()), f"types dict: {_types_dict}"
    return returnmeplz
