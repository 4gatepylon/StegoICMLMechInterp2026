Please look at chrome browser (4gate) group StegoEvalClaude or chats:
- https://claude.ai/share/d51206ef-3544-468b-9887-7a6aff7e028c
- https://claude.ai/share/17879ca6-c9c5-4583-930a-7a69d7872a24
- https://claude.ai/share/c44ca8e4-b4a7-4dc3-89c0-8befb6b5ec5b
- https://claude.ai/share/0b6aeb8d-6122-4fbc-8e71-c649fd7476c6
- https://claude.ai/share/e9655840-be9e-4eeb-9a33-f3448a445f31
- https://claude.ai/share/106fce01-cb76-45c3-8f9d-46039fa1cecc
- https://claude.ai/share/86dc594a-8594-4168-9859-902f8cc2143d

TODO(hadriano) how to use my max plan for API/programmatic querying?

Status:
- Prompt-based CAN be done
- Some strategy for obfuscation ^ but that seems to be thet main challenge
- Pure RL is unlikely to work, but maybe it won't be too obvious? It would be a lot easier with some collaborators

Likely contribution is just:
- Show that prompting WORKS (probably partially done before) in a CODING SETTING
- Show that you CAN finetune via SFT + RL. How small can the model be?
- Observe the difficulty of removing obvious traces from the CoT/training logs. Main question we answer: can you get it to do the entire computation in latent-space?
- Maybe do some mech. interp. to observe how the model does it. Depending on the situation, we might not release some relevant facts and create a website for a competition. I'm thinking I might give away $1024 in the span of 4 days.

## Testing

Decode the two example programs (encoded with cipher1) to extract their hidden bitstrings:

```bash
python decoder.py --cipher ciphers/cipher1.json cipher1_examples/encoded_example1.py
python decoder.py --cipher ciphers/cipher1.json cipher1_examples/encoded_example2.py
```

To verify against an expected prefix, use `--expect`:

```bash
python decoder.py --cipher ciphers/cipher1.json cipher1_examples/encoded_example1.py --expect "01"
```

---

The components therefore are likely:
- Great dataset. Ideally around 10K datapoints plus filtering for SFT plus RL. Probably important to include not only short but also long programs.
- Off-the-shelf SFT + RL loop/code
- Experimemnts (each training run could take 4-8 hours I think)
- Some kind of way to try and filter away evidence of how the encoding scheme works (or maybe we can insert it mechanistically? unclear)
- Competition website
- The decoder API/website
- Paper writeup
