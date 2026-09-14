# NEW TODOS
Previously, I was working on two classes of experiments: (a) encode secret messages into pretraining-like sampling, (b) encode secret messages into code. I focused on (a). I ran into some issues and training the models was taking too long so I am switching to (b). This was in `ciphers/variable_naming_in_python_v1` and `ciphers/kirchenbauer_et_al`.

The goal is to train a model that can encode one or more bits (ideally aroud one bit per line) in simple programs that you might see in the wild. The way I intend to do this is with a scheme that chooses the value of variables to something semantically "synonymous" (such as `i` and `j`) to encode bits.

Tasks to get to MVP:
- [x] 1. Define algorithm on a high-level (without harness-level details)
- [ ] 2. Acquire dataset of aroud ~1-100 good testing tasks with unit tests.
- [ ] 3. Define the interface for encoding/decoding and other parts of the stack.
- [ ] 4. Implement decoder and test it on a variety of different secret encoded messages.
- [ ] 5. Support a mock model that can work on special code files with UUID substrings (or something like that) and output canned correct/incorrect responses. This should enable us to run the harness fully (fully integration test). Ideally this mock model can run on an OpenAI API-compatible endpoint. Ideally, I can also support local (tiny) models via Ollama or something like that (even HF is fine; I could copy over my batched HF server from SAE Scoping old code---any agents working on this should ask user for that if needed; but we prefer VLLM or SGLang or Ollama or other local LLM server runtimes).
- [ ] 6. Implement evaluation setup harness without LLM Judges (only encoder/decoder)
- [ ] 7. Implement and evalute MVP v1 harness. This harness will just try 1-shot and either fail or succeed. Support both OpenRouter and Codex directly (add a TODO for OpenAI API and/or Anthropic/Claude, mentioning use of max plan; maybe support generic OpenAI API-compatible endpoints).
- [ ] 8. Define a harness API and implement and evaluate MVP v2 harness. This harness will 2-shot (first one produces the code, second one modifies it).
- [ ] 9. Clean up and generalize harnesses. All of them must take in generic multi-evaluator (intersection) stuff. Define final harness with everything exept reward-hacking judge and only using most basic "black box detection" judge (the "Basic" one). We can work with Luna for now.
- [ ] 10. Acquire yes-stego vs. no-stego data and A/B test different judges. Pick the best one for this task. Ablate the model (size).
- [ ] 11. Acquire larger dataset of aroud ~1-100K (ideally >= 10K) easy-ish python tasks (for 27b model), each with at least 10 unit tests of some kind.
- [ ] 12. Iterate on acquiring harness trajectories for ^.
- [ ] 13. TBD but implement translation. This may require some iteration on the CoT realism/quality. You might want to run some small runs on tinker while doing this to get a sense of whether it works.
- [ ] 14. TBD but upload and train on these CoTs.
- [ ] 15. TBD: Detection + Mech interp (need to define algorithm more clearly, etc...)