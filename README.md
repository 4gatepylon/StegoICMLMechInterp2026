Please look at chrome browser (4gate) group StegoEvalClaude or chats:
- https://claude.ai/share/d51206ef-3544-468b-9887-7a6aff7e028c
- https://claude.ai/share/17879ca6-c9c5-4583-930a-7a69d7872a24
- https://claude.ai/share/c44ca8e4-b4a7-4dc3-89c0-8befb6b5ec5b
- https://claude.ai/share/0b6aeb8d-6122-4fbc-8e71-c649fd7476c6
- https://claude.ai/share/e9655840-be9e-4eeb-9a33-f3448a445f31
- https://claude.ai/share/106fce01-cb76-45c3-8f9d-46039fa1cecc
- https://claude.ai/share/86dc594a-8594-4168-9859-902f8cc2143d

Status:
- Prompt-based CAN be done using Claude Opus 4.7
- Some strategy for obfuscation ^ but that seems to be thet main challenge
- Pure RL is unlikely to work, but maybe it won't be too obvious? It would be a lot easier with some collaborators

Likely contribution is just:
- Show that prompting WORKS (probably partially done before) in a CODING SETTING
- Show that you CAN finetune via SFT + RL. How small can the model be?
- Observe the difficulty of removing obvious traces from the CoT/training logs. Main question we answer: can you get it to do the entire computation in latent-space?
- Maybe do some mech. interp. to observe how the model does it. Depending on the situation, we might not release some relevant facts and create a website for a competition. I'm thinking I might give away $1024 in the span of 4 days.
- Possible bonus: seeing if claude can detect what is going on, unsupervised (just text search). for example---does it work with just prompting? with agent mode? what hints does it need?

## Testing

Decode the example programs to extract their hidden bitstrings:

```bash
# cipher1
python decoder.py --cipher ciphers/cipher1.json cipher1_examples/encoded_example1.py \
  --expect "$(head -1 cipher1_examples/encoded_example1.py | sed 's/# EXPECTED: //')"
python decoder.py --cipher ciphers/cipher1.json cipher1_examples/encoded_example2.py \
  --expect "$(head -1 cipher1_examples/encoded_example2.py | sed 's/# EXPECTED: //')"

# cipher2
python decoder.py --cipher ciphers/cipher2.json cipher2_examples/encoded_example1.py \
  --expect "$(head -1 cipher2_examples/encoded_example1.py | sed 's/# EXPECTED: //')"
```


## Dataset

Build the canonical combined code-problem dataset (requires HuggingFace `datasets` and `huggingface_hub`):

```bash
python datasets/build.py --output ./my_dataset
python datasets/build.py --output hf --hf-args repo_id=myuser/my-dataset
```

Generate steganographic code samples via the Claude Agent SDK (requires `claude_agent_sdk`, `click`):

```bash
python cipher2_problem_generator.py --n-tasks 1 --n-variants 1 --n-tries 1 -o deleteme_generated_outputs
```