# Plan
What experiments can I run to begin with?

My first batch of experiments have the purpose of answering the following question:
- Can we train a model to steganographically encode a message for at least one reasonable cipher?
- Is it easy to detect which code or content is steganographically encoded using black-box methods? (It's OK if it's possible, but it should be _harder_ than white-box testing for the MATS application/process. Also, ideally we _need_ the CoT (or something like that).)

Immediate next steps:
- [x] Download/cache part of the Fineweb dataset
- [x] Trainer should log nll vs. steganographic loss
- [x] Define YAML for main experiments (1, 2, 4, 8 bits on pretraining data)
- [x] Make sure checkpoints are saved and validation is logged (incl for the metrics ^)
- [x] Launch experiment and let it run with wandb logging these metrics properly
- [x] Log data statsitics and examples via ipynb somewhere
- [ ] Add support for storing wandb logs (i.e. so we can train scaling laws).
- [x] **TODO(Codex 1)** Add support to log the number of tokens consumed in the process of training (if not already present; otherwise let the user know how to see that).
- [ ] **TODO(Codex 2)** (Be able to) reject very short documents (caching pipeline from ^ should support some kind of filtering, etc...). Simply, make a flag for downloading the fineweb cache and a flag for loading from the cache. Each one can include/exclude files based on a minimum size (can be 0) or maximum size (can be None for no maximum). By default use 0 and None.
- [ ] **TODO(Codex 3)** Add hyper-parameter sweep support for `{lrs} x {batch size} x {nll enabled alpha 1, nll enabled alpha 2, ..., nll disabled} x {modulus, chunk/block}`. Specifically, add a script similar to the bash script that launches the 1 bit, 2 bit, etc... and then for each one try ^. Pick reasonable number of steps to do this for and reasonable hyperparemeters. Ideally, each attempted grid element does not run for more than ~15m and/or there is a smaller/better way to pick the right hyperparameters. The implementer of this might want to propose a scaling curve methodology to extrapolate the training performance based on ^ (it shold be simplest possible; maybe fit a power law or even a kinked linear function; idk).
- [ ] **TODO(Codex 4)** Add expected reds/expected greens by summing over P(red|logits) at each token (and vice versa for green). This should be logged separately for yes cases and no cases and seperate for 0 or 1-encoding tokens. I want to see (where `E` means expectation) the values for:
    - `E(red tokens over red-boosted token positions | yes) / Num(red-boosted token positions` (i.e. fraction of "should be red" that are red)
    - `E(red tokens overall | yes) / Num(tokens overall)` (this will help us understand if there is some bias)
    - Also both of these also for green. These four above will help us understand how much the boost affects and that it is properly unbiased.
    - I also want to see `E(red tokens overall | no) / Num(tokens overall)` and `E(red tokens on modulus positions | no) / Num(tokens on modulus positions)` and `E(red tokens on chunk positions | no) / Num(tokens on chunk positions)` (and same for green). This will help us understand that when there is no boosting, the distribution is in fact not boosted.
    - If possible to do mathematically in closed form using the probabilities at each token position, I would like to see some metric of variance or variation to see how tight the distribution around the mean likely is. In general this and the other expectation portions should have proper mathematical assumptions described, because without real sampling I'm not sure these analysis are entirely/fully correct; they may be making some IID assumptions or something like that.
- [ ] **TODO(Codex 5)** Add support for easily swapping out the model. I should be able to use `Qwen/Qwen3-1.7B-Base` and `Qwen/Qwen3-0.6B-Base` for smaller-scale experiment(s). We should then have the `[0.6B, 1.7B, 4B]` experiments each get their own experiment folder YAMLs and there is a documented template for running a chosen model size and number of bits. 
- [ ] We should be able to measure encode/decode accuracy. Add a generation script and then leverage the tool to MLE predict.
- [ ] Trainer should log encode/decode accuracy. We could add a callback.
- [ ] Add plotting scripts for ^ and to generate the outputs for our first 2 (X, Y) experiments.
- [ ] Optionally (if needed) look at scaling laws as a function of train resources to best identify the hero run.

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
