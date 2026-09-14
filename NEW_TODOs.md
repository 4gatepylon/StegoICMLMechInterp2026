# NEW TODOS
Previously, I was working on two classes of experiments: (a) encode secret messages into pretraining-like sampling, (b) encode secret messages into code. I focused on (a). I ran into some issues and training the models was taking too long so I am switching to (b). This was in `ciphers/variable_naming_in_python_v1` and `ciphers/kirchenbauer_et_al`.

The goal is to train a model that can encode one or more bits (ideally aroud one bit per line) in simple programs that you might see in the wild. The way I intend to do this is with a scheme that chooses the value of variables to something semantically "synonymous" (such as `i` and `j`) to encode bits.

Tasks:
- [ ] 1. Define algorithm on a high-level (without harness-level details)
- [ ] 2. Implement evaluation setup
- [ ] 3. Implement harness and evaluate quality
- [ ] 4. Finish defining and implementing "translation" (from harness trajectory to CoT-native trajectory). Iterate until the CoT looks realistic.
- [ ] 5. Train on the CoT and evaluate efficacy of the training scheme. Iterate until we have a reasonably capable steganographic model.
- [ ] 6. TBD: Detection + Mech interp (need to define algorithm more clearly, etc...)