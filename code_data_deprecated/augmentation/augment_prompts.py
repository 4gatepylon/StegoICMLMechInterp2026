from __future__ import annotations

from typing import List, Dict, Any, Tuple, Iterator
import shutil
from nemi_mvp.dataset_lib.data_entry import DatasetEntry
from pathlib import Path
import pydantic
import orjson
import numpy as np
import json
from utils.llm_judge.api_generate import load_jinja_template
import jinja2
import tqdm
from utils.jsonl_folder_reader_writer import JSONLFolderReaderWriter
from utils.llm_judge.api_generate import APIGenerator

"""
This module defines a hierarchical model (in the forms of stream that sample at each
step/level) to overall sample the synthetic data.
"""


class TemplatedEntry(pydantic.BaseModel):
    string: str
    j2name: str  # this determiens which set of rephraseables are OK
    keylist: List[str]  # this determiens which set of rephraseables are OK


# TODO(Adriano) modify this to use readerwriter and be mem. limited
# XXX todos include:
#   (3) generic stream chain => Move this to utils and define a stream class with chaining utilities, that we we just instantiate a streamchain from
#       the relevant streams here in the `load_dataset_entry` method
#   (4) base class for keylist mappable (also support mappable via a specific key-gen)
#   (5) implement the pre-filter thing --- stream onto batched oai requests, rpetty obvious
#   (6) test the overall chain with the dummy prompts
#   (7) create real prompts!
class CachedStream:  # NOTE: stream internals must be json serializable
    def __init__(
        self,
        cache_filepath: Path,
        clobber_cache: bool = False,
        write_kwargs: Dict[str, Any] = {},
    ) -> None:
        self.cache_filepath = cache_filepath
        self.clobber_cache = clobber_cache
        if self.cache_filepath.exists() and not self.clobber_cache:
            raise FileExistsError(f"Cache file {self.cache_filepath} already exists")
        elif self.cache_filepath.exists() and self.clobber_cache:
            shutil.rmtree(self.cache_filepath.resolve().as_posix())
        self.cache_filepath.mkdir(parents=True, exist_ok=True)
        self.reader_writer = JSONLFolderReaderWriter(self.cache_filepath)

        # NOTE: these are the defaults!
        # max_n_files: int | float = float("inf"),
        # max_lines_per_file: int | float = float("inf"),
        # max_size_per_file: int | float = float("inf"),
        # max_size_overall: int | float = float("inf"),
        # max_lines_overall: int | float = float("inf"),
        self.write_kwargs = write_kwargs

    def call_inner(self) -> Iterator[Any]:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Iterator[Dict[str, Any]]:
        g = self.reader_writer.write_stream(self.call_inner(*args, **kwargs), **self.write_kwargs)
        # item is passthrough, _ is (bytes_written, lines_written, files_written)
        for item, _ in g:
            yield item


class DatasetEntryCombinationsGenerator(CachedStream):
    def __init__(
        self,
        weights: Path | Dict[str, float] = Path(__file__).parent / "sample_combinations_weights.json",
        cache_filepath: Path = Path(__file__).parent / ".cache" / "dataset_entry_combinations",
        clobber_cache: bool = False,
    ) -> None:
        super().__init__()
        self.weights = orjson.loads(weights.read_bytes()) if isinstance(weights, Path) else weights
        assert isinstance(self.weights, dict)
        assert set(self.weights.keys()) == {
            "starter_code",
            "things_to_know",
            "source_description",
            "question",
            "answers",
            "expected_inputs_outputs",
            "difficulty",
            "url",
        }
        # probability of selection in the weights i.e. {key: probbility it is kept/selected}
        assert all(isinstance(v, float) for v in self.weights.values())
        assert all(0.0 <= v <= 1.0 for v in self.weights.values())
        self.disallowed_arrays = [
            [],  # no empty keys
        ]
        self.cache_filepath = cache_filepath
        self.clobber_cache = clobber_cache

    @staticmethod
    def _sample_array(options2weights: Dict[str, float], n_samples: int) -> List[List[str]]:
        """
        Utility to sample random keys from a dictionary of independent probabilities for
        each key showing up.
        """
        # Sanity and sort
        assert all(isinstance(v, float) for v in options2weights.values())
        assert all(0.0 <= v <= 1.0 for v in options2weights.values())
        keys_values = sorted(list(options2weights.items()), key=lambda x: x[0])
        keys = [k for k, _ in keys_values]
        values = np.array([v for _, v in keys_values]).reshape(1, -1)
        assert values.shape == (1, len(options2weights))  # batch, n_options
        # Sample
        samples = np.random.random((n_samples, len(options2weights)))  # batch, n_options
        select = samples < values
        ret = [[k for k, s in zip(keys, row) if s] for row in select]
        assert len(ret) == n_samples
        assert all(len(l) >= 0 for l in ret)
        return ret

    @staticmethod
    def generate_entry_combos(
        # How to sample
        disallowed_arrays: List[List[str]],
        weights: Dict[str, float],
        # What to sample for/from + constraints on sampling and outputting
        dataset_entry: DatasetEntry | Dict[str, Any],
        n_samples: int = 100,
        allow_duplicates: bool = False,
        deserialize: bool = False,
        # Operational parameters for runtime/etc...
        insurance_multiplier: int = 2,
    ) -> DatasetEntry | Dict[str, Any]:
        """
        Sample realistic combinations of the dataset entry's components without
        """
        # Impossibility sans
        # TODO(Adriano) check for all disallowed arrays that our weights don't screw
        # it over (the way to do this is not totally clear, but it shouldn't be tooooo
        # hard)
        if 2 ** len(weights) - len(disallowed_arrays) < n_samples:
            raise ValueError(f"Not enough weights to sample {n_samples} combinations")
        if deserialize:
            raise NotImplementedError("Not implemented yet")  # Not clear how to do
        # preproc. for format and runtime
        disallowed_arrays = [tuple(t) for t in disallowed_arrays]
        dataset_entry = dataset_entry.serialize() if isinstance(dataset_entry, DatasetEntry) else dataset_entry
        sampled_arrays: List[List[str] | Tuple[str, ...]] = []
        while len(sampled_arrays) < n_samples:
            sampled_arrays = DatasetEntryCombinationsGenerator._sample_array(weights, n_samples * insurance_multiplier)
            sampled_arrays = [arr for arr in sampled_arrays if tuple(arr) not in disallowed_arrays]
            if not allow_duplicates:
                sampled_arrays = [list(a) for a in set(tuple(arr) for arr in sampled_arrays)]
            sampled_arrays = sampled_arrays[:n_samples]
        assert isinstance(sampled_arrays, list)
        assert all(isinstance(arr, list) for arr in sampled_arrays)
        assert all(all(isinstance(s, str) for s in arr) for arr in sampled_arrays)
        assert len(sampled_arrays) == n_samples
        entry_dicts_made = [{key: dataset_entry[key] for key in array} for array in sampled_arrays]
        return entry_dicts_made

    def call_inner(
        self,
        dataset_entries: Iterator[DatasetEntry],
        # Kwargs for the per-entry specific sampling
        n_samples_per_entry: int = 100,  # greedy FYI
        n_samples_overall: int = 10000,
        allow_duplicates: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        n_samples_per_entry = min(n_samples_per_entry, n_samples_overall)
        n_samples_yielded: int = 0
        if n_samples_overall == 0:
            return
        for dataset_entry in dataset_entries:
            sampled_arrays = self.generate_entry_combos(
                disallowed_arrays=self.disallowed_arrays,
                weights=self.weights,
                dataset_entry=dataset_entry,
                n_samples=n_samples_per_entry,
                allow_duplicates=allow_duplicates,
                deserialize=False,  # wnat dicts
                insurance_multiplier=2,  #
            )
            assert len(sampled_arrays) == n_samples_per_entry
            assert all(isinstance(d, dict) for d in sampled_arrays)
            for _dict in sampled_arrays:
                yield _dict
                n_samples_yielded += 1
                if n_samples_yielded >= n_samples_overall:  # exit early
                    return
            # paranoid programming but whatever
            if n_samples_yielded >= n_samples_overall:  # exit for good
                return  #


class DictionaryTemplateRender:
    """
    Provide utilities to, given a set of j2 templates (passed in via a folder or even
    literally as a parameter), stream out randomly sampled template choices, rendered,
    based on a distribution defined by the folder or a parameter.

    In stream, incoming dictionaries must have:
    1. Keys to render under the "render" key (pointing to items to render, so a dict)
    2. Optional "identifier" key whose value is a string and which is reserved for
        `RenderStrategy` (a future abstraction to support non-random-sampling template +
        render choices)
    3.

    NOTE: it is not supported to do programmatic (python) "rendering" of a more advanced
    variety, nor is it supported BY DEFAULT to use as a CachedStream. (Though, to allow
    inheritors to use CachedStream, the actual logic is in `call_inner` such that when
    __call__ is overwritten it will work).
    """

    def __init__(self) -> None:
        pass

    def call_inner(self, stream: Iterator[Dict[str, str]], *args, **kwargs) -> Iterator[Dict[str, str]]:
        raise NotImplementedError

    def __call__(self, stream: Iterator[Dict[str, str]], *args, **kwargs) -> Iterator[Dict[str, str]]:
        yield from self.call_inner(stream, *args, **kwargs)


class DatasetEntryInitialRender(CachedStream):
    def __init__(
        self,
        initial_templates_j2s_folder: Path = Path(__file__).parent / "initial_templates",
        cache_filepath: Path = Path(__file__).parent / ".cache" / "dataset_entry_renders",
        clobber_cache: bool = False,
        # folder has "map.json" and a bunch of *.j2 files
        # map is used to map from the keys above (combinations supported---all2**8=256)
        # to supported templates and _weights_
    ) -> None:
        self.initial_templates_j2s_folder = initial_templates_j2s_folder
        self.cache_filepath = cache_filepath
        self.clobber_cache = clobber_cache

        j2_files = []
        for ext in [".j2", ".jinja2"]:
            for glob_prefix in ["*", "**/*"]:
                j2_files.extend(list(initial_templates_j2s_folder.glob(f"{glob_prefix}{ext}")))
        # NOTE: these two are order-zipped
        j2_files = list(set(j2_files))
        if len(set(j2f.name for j2f in j2_files)) != len(j2_files):
            raise ValueError(f"Duplicate template file NAMES found in {initial_templates_j2s_folder} (not allowed sorry)")
        self.templates = [load_jinja_template(j2_file) for j2_file in j2_files]
        assert all(isinstance(template, jinja2.Template) for template in self.templates)

        # Load the mapfile
        mapfile = initial_templates_j2s_folder / "map.json"
        assert mapfile.exists()
        map = orjson.loads(mapfile.read_bytes())
        # NOTE that because no tuples in json and we cannot have keys as lists, we have
        # a list of 2-element lists:
        # [keys-list (template key), {template filenames that support it: probability of selection}]
        #
        # List of 2-element lists
        assert isinstance(map, list)
        assert all(isinstance(item, list) for item in map)
        assert all(len(item) == 2 for item in map)
        # List of stirngs as keys
        assert all(isinstance(item[0], list) for item in map)
        assert all(all(isinstance(key, str) for key in item[0]) for item in map)
        # List of these dicts and they must be weights over the templates
        assert all(isinstance(item[1], dict) for item in map)
        assert all(isinstance(item[1][key], float) for item in map for key in item[1])
        assert all(0.0 <= item[1][key] <= 1.0 for item in map for key in item[1])
        assert all(abs(sum(item[1].values()) - 1.0) < 1e-6 for item in map)
        map2 = [(tuple(item[0]), item[1]) for item in map]
        assert len(map2) == len(set(k for k, _ in map2))
        map3 = {k: v for k, v in map2}
        # Map from name2template and map from keylist to names
        self.name2template = {j2f.name: template for j2f, template in zip(j2_files, self.templates)}
        # (make sure that all the names refer to SOME template)
        assert all(all(vk in self.name2template for vk in v.keys()) for v in map3.values())
        self.keylist2name = map3

    def call_inner(self, combinations: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:  # output is of a `TemplatedEntry` type
        for combination in combinations:
            keylist = tuple(sorted(combination.keys()))  # sort for determinism
            names_dist = sorted(self.keylist2name[keylist].items(), key=lambda x: x[0])
            probabilities = np.array([v for _, v in names_dist])
            probabilities = probabilities / probabilities.sum()  # just in case
            name = np.random.choice([n for n, _ in names_dist], p=probabilities)
            template = self.name2template[name]
            yield TemplatedEntry(string=template.render(combination), j2name=name, keylist=keylist).model_dump()


class LLMRephrasePromptRender(CachedStream):
    def __init__(
        self,
        cache_filepath: Path = Path(__file__).parent / ".cache" / "llm_rephrase_prompt_renders",
        clobber_cache: bool = False,
        # also has a map like above with rephraseables
        rephrase_prompts_j2s_folder: Path = Path(__file__).parent / "rephrase_templates",
    ) -> None:
        self.cache_filepath = cache_filepath
        self.clobber_cache = clobber_cache
        self.rephrase_prompts_j2s_folder = rephrase_prompts_j2s_folder

    def __call__(self, combinations: Iterator[DatasetEntry]) -> Iterator[str]:
        raise NotImplementedError


class APIGeneratorStream(CachedStream):
    def __init__(
        self,
        cache_filepath: Path = Path(__file__).parent / ".cache" / "api_generator_stream",
        clobber_cache: bool = False,
        # also has a map like above with rephraseables
        rephrase_prompts_j2s_folder: Path = Path(__file__).parent / "rephrase_templates",
        prompt_key: str = "prompt",
        response_key: str = "response",
        model: str = "gpt-4.1-nano",
        outer_batch_size: int = 4096,
        generate_kwargs: Dict[str, Any] = {},
    ) -> None:
        self.cache_filepath = cache_filepath
        self.clobber_cache = clobber_cache
        self.rephrase_prompts_j2s_folder = rephrase_prompts_j2s_folder
        self.generator = APIGenerator()
        self.prompt_key = prompt_key
        self.response_key = response_key

    def call_inner(self, prompts: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        prompts_buffer: List[str] | List[List[Dict[str, str]]] = []
        # NOTE: this will duplicate!
        for prompt in prompts:
            # 1. Generate if necessary
            if len(prompts_buffer) >= self.outer_batch_size:
                responses_inner_stream = self.generator.api_generate_json_mode_streaming(
                    prompts=prompts_buffer,
                    model=self.model,
                    **self.generate_kwargs,
                )
                for response in responses_inner_stream:
                    yield {self.response_key: response, **prompt}
                prompts_buffer = []
            # 2. Add to prompts buffer
            prompts_buffer.append(prompt[self.prompt_key])
        # 3. Generate if necessary (at end...)
        if len(prompts_buffer) > 0:
            responses_inner_stream = self.generator.api_generate_json_mode_streaming(
                prompts=prompts_buffer,
                model=self.model,
                **self.generate_kwargs,
            )
            for response in responses_inner_stream:
                yield {self.response_key: response, **prompt}


class LLMQualiyFilter:
    """
    Calculate the quality of the LLM queries by accumulating them into a buffer and
    then in blocks picking the ones that pass certain thresholds, are top-k 'best',
    etc...

    TODO(Adriano) actually implement this instead of passthrough. For now it's simply
    an abstraction layer for future work.

    Specifically we might use some of the following criteria:
    - They are not too close (using a DP algorithm I think; it's like a knapsack)
        - String distance
        - Strict equality check
        - Bleu score
        - Emedding similarity (maybe)
    - Coherence filtering with prompts
    """

    def __call__(self, prompts: Iterator[str]) -> Iterator[str]:
        yield from prompts


class SyntheticDataGenerationPipeline:  # XXX generic stream plz
    def __init__(
        self,
        # TODO(Adriano) refactor to streams and then make this generic
        dataset_entry_combination_kwargs: Dict[str, Any] = {},
        dataset_entry_initial_render_kwargs: Dict[str, Any] = {},
        llm_rephrase_kwargs: Dict[str, Any] = {},
        llm_quality_filter_kwargs: Dict[str, Any] = {},
    ) -> None:
        # self.dataset_entry_combination_kwargs = dataset_entry_combination_kwargs
        # self.dataset_entry_initial_render_kwargs = dataset_entry_initial_render_kwargs
        # self.llm_rephrase_kwargs = llm_rephrase_kwargs
        # self.llm_quality_filters = llm_quality_filters
        self.dataset_entry_combination = DatasetEntryCombinationsGenerator(**dataset_entry_combination_kwargs)
        self.dataset_entry_initial_render = DatasetEntryInitialRender(**dataset_entry_initial_render_kwargs)
        self.llm_rephrase = LLMRephrasePromptRender(**llm_rephrase_kwargs)
        self.llm_quality = LLMQualiyFilter(**llm_quality_filter_kwargs)

    def __call__(self, dataset_entries: Iterator[DatasetEntry]) -> Iterator[str]:
        chained_iterator = self.llm_quality(self.llm_rephrase(self.dataset_entry_initial_render(self.dataset_entry_combination(dataset_entries))))
        yield from chained_iterator


def load_dataset_entries(
    path: Path = Path(__file__).parent.parent.parent / "merged_code_datasets",
    deserialize: bool = False,
) -> Iterator[DatasetEntry]:
    for file in sorted(path.glob("*.jsonl")):
        content = file.read_bytes()
        for line in content.split(b"\n"):
            if len(line.strip()) > 0:
                object = orjson.loads(line)
                try:
                    entry = DatasetEntry.deserialize(object) if deserialize else object
                    yield entry
                except Exception:
                    assert isinstance(object, list)
                    for entry in object:
                        yield DatasetEntry.deserialize(entry) if deserialize else object


def create_synthetic_data(  # this is our main entrypoint that you import and use basically
    self,
    output_folder=Path(__file__).parent.parent.parent / "merged_augmented_code_datasets",
    n_items: int = 10,  # debug
    max_file_size: int = 500 * 1e6,  # 500MB
    max_n_files: int | float = float("inf"),
    max_overall_Size: int = 100 * 1e9,  # 100 GB should be WAY more than enough
    path: Path = Path(__file__).parent.parent.parent / "merged_code_datasets",
    **kwargs,  # for the __call__ streams above
) -> None:
    input_stream = load_dataset_entries(path)
    tf_stream_creator = SyntheticDataGenerationPipeline(**kwargs)
    tf_stream = tf_stream_creator(input_stream)

    reader_writer = JSONLFolderReaderWriter(output_folder)  # TODO insert max file size etc... (n times etc...)
    reader_writer.write(tf_stream)


if __name__ == "__main__":
    # Dummy generated by Claude
    # Debugging code basically
    test_entry = DatasetEntry(
        starter_code="def solution(n):\n    # Your code here\n    pass",
        things_to_know="This is a dynamic programming problem",
        source_description="leetcode.com",
        source_dataset="test_dataset",
        question_id="test_001",
        question="Given an integer n, return the nth Fibonacci number.",
        answers=["def solution(n):\n    if n <= 1:\n        return n\n    return solution(n-1) + solution(n-2)"],
        expected_inputs_outputs={"inputs": ["5", "10"], "outputs": ["5", "55"]},
        difficulty="medium",
        url="https://leetcode.com/problems/fibonacci",
        prompt=None,
        full_generation=None,
        response=None,
        parsed_response_code=None,
        gotten_inputs_outputs=None,
        prompt_hyperparamaters=None,
        metadata=None,
    )
    weights = {  # weights to include EVERYTHING
        "starter_code": 0.5,
        "things_to_know": 0.5,
        "source_description": 0.5,
        "question": 0.5,
        "answers": 0.0,
        "expected_inputs_outputs": 0.0,
        "difficulty": 0.5,
        "url": 0.5,
    }

    def debug_dataset_entry_combinations_generator():
        generator = DatasetEntryCombinationsGenerator(weights=weights)
        entries = [test_entry for _ in range(20_000)]
        for entry in tqdm.tqdm(
            # TODO(Adriano) for some reason a performance issue (I think? infinite while loop?
            # causes freezing for large numbers of samples per entry)
            generator(entries, n_samples_per_entry=50, n_samples_overall=1_000_000),
            desc="Generating entries...",
            total=1_000_000,
        ):
            # print(json.dumps(entry, indent=4))
            pass

    # debug_dataset_entry_combinations_generator()
    def debug_dataset_entry_initial_render():
        generator1_creator = DatasetEntryCombinationsGenerator(weights=weights)
        generator2_creator = DatasetEntryInitialRender()
        entries = iter([test_entry for _ in range(20_000)])
        generator1 = generator1_creator(entries, n_samples_per_entry=50, n_samples_overall=1_000_000)
        generator2 = generator2_creator(generator1)
        for i, entry in enumerate(
            tqdm.tqdm(
                generator2,
                desc="Generating entries...",
                total=1_000_000,
            )
        ):
            if i < 3:
                print(json.dumps(entry, indent=4))
            pass

    debug_dataset_entry_initial_render()
