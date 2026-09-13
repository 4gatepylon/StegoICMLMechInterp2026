# Plan
What experiments can I run to begin with?

My first batch of experiments have the purpose of answering the following question:
- Can we train a model to steganographically encode a message for at least one reasonable cipher?
- Is it easy to detect which code or content is steganographically encoded using black-box methods? (It's OK if it's possible, but it should be _harder_ than white-box testing for the MATS application/process. Also, ideally we _need_ the CoT (or something like that).)

Immediate next steps:
- [x] Download/cache part of the Fineweb dataset
- [x] Trainer should log nll vs. steganographic loss
- [ ] ]We should be able to measure encode/decode accuracy (we should have a module for this; BTW I'm not sure this makes sense without sampling... maybe there is a closed form solution?)
- [ ] Trainer should log encode/decode accuracy. We could add a callback.
- [x] Define YAML for main experiments (1, 2, 4, 8 bits on pretraining data)
- [x] Make sure checkpoints are saved and validation is logged (incl for the metrics ^)
- [ ] Launch experiment and let it run with wandb logging these metrics properly
- [ ] Reject very short documents (caching pipeline from ^ should support some kind of filtering, etc...)
- [x] Log data statsitics and examples via ipynb somewhere
- [ ] Add hyper-parameter sweep support for `{lrs} x {batch size} x {nll enabled alpha 1, nll enabled alpha 2, ..., nll disabled} x {modulus, chunk/block}`
- [ ] Add other useful metric logging to trainer such as expected red/green,
- [ ] Add support for sampler-based experiment with encoder/decoder accuracy and red/green distributions
- [ ] Add plotting scripts for ^ and to generate the outputs for our first 2 (X, Y) experiments.

Good experiments for the pretraining bits-encoding objective here:
- [ ] **Question:** (X) Can we encode a single bit in pretraining data/generation using a similar logit-boost cipher? `We are working this`
- [ ] **Question:** (Y) Can we encode multiple bits in pretraining data/generation using a similar logit-boost cipher? `We are working this`
- [ ] **Question:** Can we detect using black-box on outputs from (X) or (Y)?
- [ ] **Question:** Can we detect using white-box on outputs from (X) or (Y)?
- [ ] **Question:** How do models from (X) or (Y) work? Can we look at attention maps, probe for quantities that might be indicative, ablate neurons/components, etc... and try to figure out what's going on?

Good experiments for the variable renaming bits-encoding objective here:
- [x] **Question:** (A) Can we prompt Claude or another powerful model to generate plausible/correct steganographic code for a simple cipher (without access to tools)? **Answer:** Yes, it's possible. However it does not look realistic to a human (although Claude seems unable to detect it!).
- [ ] **Question:** (B) Can we setup a harness for getting Claude or another powerful model to generate that plausible/correct steganographic code for in a way that ACTUALLY looks realistic to a human?
- [ ] **Question:** (C) Can we get a smaller model with few-shot-learning, harness engineering, etc... to generate that plausible/correct steganographic code (from A) for in a way that ACTUALLY looks realistic to a human?
- [ ] **Question:** (D) Can we get that smaller model (from C) to generate that plausible/correct steganographic code (from A) without the harness via post-training/RL (probably start with SFT or SDPO using known trajectories and then move on to GRPO/other RL _if needed_)?
- [ ] **Question:** How does the model from (D) do it's task? Look at the CoT and possibly try a few plausible hypotheses (look at attention maps, ablate neurons/components, probe for quantities that might be indicative, etc...).

Future batches can include:
- Can we detect it using white-box testing?
- Can we get actually really good steganography?
- Can we detect it without CoT (etc...)
