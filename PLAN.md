# Plan
What experiments can I run to begin with?

My first batch of experiments have the purpose of answering the following question:
- Can we train a model to steganographically encode a message for at least one reasonable cipher?
- Is it easy to detect which code or content is steganographically encoded using black-box methods? (It's OK if it's possible, but it should be _harder_ than white-box testing for the MATS application/process. Also, ideally we _need_ the CoT (or something like that).)

Good experiments here:
- [x] **Question:** (A) Can we prompt Claude or another powerful model to generate plausible/correct steganographic code for a simple cipher (without access to tools)? **Answer:** Yes, it's possible. However it does not look realistic to a human (although Claude seems unable to detect it!).
- [ ] **Question:** (B) Can we setup a harness for getting Claude or another powerful model to generate that plausible/correct steganographic code for in a way that ACTUALLY looks realistic to a human?
- [ ] **Question:** (C) Can we get a smaller model with few-shot-learning, harness engineering, etc... to generate that plausible/correct steganographic code (from A) for in a way that ACTUALLY looks realistic to a human?
- [ ] **Question:** (D) Can we get that smaller model (from C) to generate that plausible/correct steganographic code (from A) without the harness via post-training/RL (probably start with SFT or SDPO using known trajectories and then move on to GRPO/other RL _if needed_)?
- [ ] **Question:** How does the model from (D) do it's task? Look at the CoT and possibly try a few plausible hypotheses (look at attention maps, ablate neurons/components, probe for quantities that might be indicative, etc...).
- [ ] **Question:** (X) Can we encode a single bit in pretraining data/generation using a similar logit-boost cipher?
- [ ] **Question:** (Y) Can we encode multiple bits in pretraining data/generation using a similar logit-boost cipher?
- [ ] **Question:** How do models from (X) or (Y) work? Can we look at attention maps, probe for quantities that might be indicative, ablate neurons/components, etc... and try to figure out what's going on?

Future batches can include:
- Can we detect it using white-box testing?
- Can we get actually really good steganography?
- Can we detect it without CoT (etc...)