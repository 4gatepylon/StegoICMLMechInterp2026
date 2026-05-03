# Augmentation
The purpose of the Augmentation library is to provide utilities to turn our four datasets (BAAI, deepcode code contests, etc...) into larger, realistic datasets if questions. It is assumed that the answers will be acquired in some other way.

# High level view
The generation of data is created using a pipeline of data that uses both heuristics and LLMs to convert the datasets into a large dataset.

# Dataset Entry
This module provides multiple prompt "templates" (really functions here) that take in
a DatasetEntry like so:
```
{
    starter_code: Optional[str]
    things_to_know: Optional[str]
    source_description: str  # something like "codeforces.com" or whatever this offers
    source_dataset: str  # something like BAAI/TACO, codeparrot/apps, ... (HF dataset)
    question: str # a good description of the question
    answers: List[str] # examples of solutions (note: not all may read stdin/out corrrectly)

    #### Expected inputs/outputs (can be used as examples) ####
    # {
    #   'inputs': [...],
    #   'outputs': [...],
    # }
    expected_inputs_outputs: Optional[Dict[str, List[Any] | scr]]
    difficulty: str
    url: str | None
}
```
(dataset entry defined in `../data_entry.py`)

These dataset entries are used to create prompts via a template-based algorithm.

# Implementation Details
## AI Suggestions
- https://chatgpt.com/share/68b6464e-125c-8010-9510-ea144378750b
- https://claude.ai/share/fac7b7f4-9f79-4e37-a628-0978a27530fa

## Axes of variance
We aim to vary the following things:
1. What information is given or withheld. We may provide or withold some of the following:
    - The metadata such as things_to_know (string), source, difficulty, url, etc...
    - The answers (we may give partial answer, starter code, or something else)
    - Examples of inputs and outputs
2. The format. Specifically these are the main axes of variances
    - Markdown vs. Spaces/newlines-speeration only vs. seperation with long lines like
        "=====..." vs. XML
    - Code formatting requests (i.e. please be concise vs. document clearly vs. etc...)
    - Prose vs. bullet-point request style
    - Task framing variations:
        - Direct instruction ("Implement a solution for...")
        - Collaborative tone ("Let's work together to solve...")
        - Expert assumption ("As an experienced programmer, solve...")
        - Teaching context ("Explain your approach while solving...")
        - Debugging scenario ("Fix and complete this partial solution...")
        - Code review ("Improve this working but inefficient solution...")
    - Setting: no setting vs. made up setting vs. etc... (similar to above)
    - Thinking/explanation request (also ask for a description/thinking/etc... vs.
        to just ask for the answer and no thinking vs. multiple solution vs. test-cases
        too (first) vs. etc...) => This is what claude refers to as "Solution Depth
        Variations"
    - We basically ask an LLM to re-phrase this in multiple different ways and then add
        that to the dataset (with tracking across these).
    - Adding typos vs. not in the request (we just use a typo library)
    - Amount of specification (some specification will be very clear, and sometimes it
        will be a bit vague).
    - TODO(Adriano) add code that is broken in subtle ways. We won't do this for now,
        but in the future we can use the AST to randomly break stuff, or ask an LLM to
        add a subtle bug (this is here just as a note/reminder for future todos).

We cover the following 10 rough categories of prompts:
1. The default prompt.
    - Mentions each of the things given above except the expected inputs/outputs and
        answers (those are used for testing, verification, etc...)
    - Formats everything in markdown format.
2. Tiny prompt (i.e. just the question) in markdown format
3. Tiny prompt which prompts the return format (also in markdown)
4. Tiny prompt which prompts for the return format AND also gives a few of the
    expected inputs/outputs for the model to be able to see and make better choices.
5. Reformat + partial answer (i.e. just the question and answer) and a descrption of
    the input/output desired. All in markdown format.
6. Reformat + partial answer + the return format description + a few expected
    inputs/outputs for the model to be able to see and make better choices. All in
    markdown format.
7. Reformat + full answer + the return format description (markdown)
8. Reformat + full answer + the return format description + a few expected inputs/outputs
    (markdown).
9. Reformat + full answer + all the metadata (markdown).
10. Variants of all of the above with non-markdown (look at axes of variance above).

# Scope of this data
NOTE: the scope of the model this data trains is:
- Inputs:
    - English requests. No other language supported
    - Code examples, etc... supported
    - Common formatting strategies supported (newlines, seperators, markdown, xml)
- Outputs:
    - Explanations supported
    - Code snippets supported (more than one)
    - Test-cases supported
    - ONLY PURE CODE supported (that is to say, NO web-code or other such side-effect
        include code---no servers, cude, ML code etc... supported)
    - Multithreading, multiprocessing, and file IO not _really_ supported, but it's OK
        if it's in the data (we don't really care). The main idea is that the code you
        get here should run on any python runtime (i.e. for a code contest).
    - ONLY STANDARD LIBRARIES supported (no pydantic, click, other common libraries)
    - ONLY PYTHON SUPPORTED (no other languages)
    - ONLY ONE TURN supported (no multi-turn)
