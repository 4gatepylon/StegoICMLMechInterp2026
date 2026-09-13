# High level plan
Basically the high-level plan is pretty basic:
1. Define a good code cipher
2. Collect a large number of high quality coding problems with golden solutions/examples for encoding a secret message (short code) into each. This should be at least 10K easy problems and realistic code snippets. They should ideally all be scripts or runnable snippets that can be tested via STDIN/STDOUT. All the code should work as well (pass a set of pre-defined unit tests), have the correct secret message be decodeable, and have correctness tests for its specification. It should be uploaded to huggingface and easy-to-use.
3. Evaluate large models on (2). The goal is to see if prompt engineering "just works".
4. Train via SFT and/or RL to get a model that can do the task. Maybe derisk with OpenAI finetuning API? Unclear. The goal is to get open weights model that is small and works and doesn't make the cipher too, too obvious, but focus on code performance and cipher correctness.
5. Find some way to make the cipher not-too-obvious. Iterate on obfuscation. This should produce a model that can encode with the cipher but is hard to leak the cipher from.
6. Perform Mech. Interp. and/or other ablations/experiments to understand how this model works and try and leak the cipher. If it's hard and non-obvious, release the competition.

# TODOs
1. Make a good plan here.
2. Debug "credit balance too low" despite being able to use claude code. Have a way to use a big model.
3. Review the code marked `TODO(hadriano)` and make sure it's free from bugs (be able to test it).
4. Collect large dataset of coding problems in two variants: (a) prompt claude or an LLM to refactor a LEETCODE-STYLE (easy) problem to encode a secret message (short code), (b) prompt claude (max/opus) to produce useful SCRIPTS for common tasks (in ML R&D, data science, etc...). Focus entirely on python.
    - Save the dataset to disk and upload to huggingface
    - ... TBD need to write this.
... TBD after this

## TBD old deprecated TODOs
The components therefore are likely:
- Great dataset. Ideally around 10K datapoints plus filtering for SFT plus RL. Probably important to include not only short but also long programs.
- Off-the-shelf SFT + RL loop/code
- Experimemnts (each training run could take 4-8 hours I think)
- Some kind of way to try and filter away evidence of how the encoding scheme works (or maybe we can insert it mechanistically? unclear)
- Competition website
- The decoder API/website
- Paper writeup

- (Low priority) Understand how Agent SDK works: https://code.claude.com/docs/en/agent-sdk/overview
- Get a really good cipher + prompt
- Understand how to measure model usage under max and understand how many model calls I can make/how much I can programmatically extract here. More generally, get a way to get a lot of data (ideally 10K samples) as soon as possible. If I could possible modify previous problems from previous coding tasks that could help. I think I could do it with an agent and providing the working nono-cover text as an example + the decoder as verification loop for an agent. I could set caude code to autonomously try to get it work (or make a scaffold).
- Come up with a plan to avoid making it too easy to reveal (I thought I might RL against claude but not sure tbh)
- Do small model experiemnts

## Low priority things
- Review and fix website(s)
- Fix the `.cursor` and `.vscode` settings files (and ruff, etc...)
