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
- [x] **TODO(Codex 2)** (Be able to) reject very short documents (caching pipeline from ^ should support some kind of filtering, etc...). Simply, make a flag for downloading the fineweb cache and a flag for loading from the cache. Each one can include/exclude files based on a minimum size (can be 0) or maximum size (can be None for no maximum). By default use 0 and None.
- [ ] **TODO(Codex 3)** Add hyper-parameter sweep support for `{lrs} x {batch size} x {nll enabled alpha 1, nll enabled alpha 2, ..., nll disabled} x {modulus, chunk/block}`. Specifically, add a script similar to the bash script that launches the 1 bit, 2 bit, etc... and then for each one try ^. Pick reasonable number of steps to do this for and reasonable hyperparemeters. Ideally, each attempted grid element does not run for more than ~15m and/or there is a smaller/better way to pick the right hyperparameters. The implementer of this might want to propose a scaling curve methodology to extrapolate the training performance based on ^ (it shold be simplest possible; maybe fit a power law or even a kinked linear function; idk).
  - Worktree: `tokyo`; branch: `4gatepylon/plan-codex-3-todo`.
  - Planning phase implemented: [manifest generation and archived-W&B timing estimates](ciphers/kirchenbauer_et_al/experiments/SWEEP_PLANNING.md). The sequential runner remains deferred until the generated grid and estimates are reviewed. Trials use 4,096 documents with no time cap; the initial 4B grid estimates 475.87 node-hours.
- [ ] **TODO(Codex 4)** Add expected reds/expected greens by summing over P(red|logits) at each token (and vice versa for green). This should be logged separately for yes cases and no cases and seperate for 0 or 1-encoding tokens. I want to see (where `E` means expectation) the values for:
    - **Owner / location:** Codex 4; branch `4gatepylon/codex-4-todo-plan`; worktree `/Users/4gate/conductor/workspaces/StegoICMLMechInterp2026/edmonton`.
    - **Status (2026-09-13):** Code review and initial plan complete; implementation not started. Awaiting clarification of model/teacher reference curves and `no` position groups, then test-plan clarification before implementation.
    - **Proposed plan:** Log token-weighted red/green expected fractions and counts in train/eval, split by yes/no and 0/1-assigned positions; add variation metrics with explicit fixed-context/independence assumptions, focused behavioral tests, and documentation.
    - `E(red tokens over red-boosted token positions | yes) / Num(red-boosted token positions` (i.e. fraction of "should be red" that are red)
    - `E(red tokens overall | yes) / Num(tokens overall)` (this will help us understand if there is some bias)
    - Also both of these also for green. These four above will help us understand how much the boost affects and that it is properly unbiased.
    - I also want to see `E(red tokens overall | no) / Num(tokens overall)` and `E(red tokens on modulus positions | no) / Num(tokens on modulus positions)` and `E(red tokens on chunk positions | no) / Num(tokens on chunk positions)` (and same for green). This will help us understand that when there is no boosting, the distribution is in fact not boosted.
    - If possible to do mathematically in closed form using the probabilities at each token position, I would like to see some metric of variance or variation to see how tight the distribution around the mean likely is. In general this and the other expectation portions should have proper mathematical assumptions described, because without real sampling I'm not sure these analysis are entirely/fully correct; they may be making some IID assumptions or something like that.
- [ ] **TODO(Codex 5)** Add support for easily swapping out the model. I should be able to use `Qwen/Qwen3-1.7B-Base` and `Qwen/Qwen3-0.6B-Base` for smaller-scale experiment(s). We should then have the `[0.6B, 1.7B, 4B]` experiments each get their own experiment folder YAMLs and there is a documented template for running a chosen model size and number of bits. **Branch:** `4gatepylon/plan-codex-5-todo`; **Worktree:** `/Users/4gate/conductor/workspaces/StegoICMLMechInterp2026/kingstown`.
- [ ] **TOY MODEL** Do an experiment with a TOY MODEL. Like SERIOUSLY TOY (i.e. 1-4 layers and fewer bits and maybe hardcoded to output uniform values or smth).
- [ ] **[BUG] L-Q vs. Q**. There is a conceptual bug where we are not using the 4096 generation length for the actual remaining tokens. Instead we do 4096 - prefix length. This means the chunking is fucked up (I think). But I'm not actually sure what is going on here. I think my agent messed it up and I'll need to look into it. **Worktree:** `/Users/4gate/conductor/workspaces/StegoICMLMechInterp2026/chengdu`; **branch:** `4gatepylon/audit-chunk-length`.
- [ ] **TODO(Codex 6)** Build standalone encode/decode and steganalysis evaluation tooling. Owner: Codex; branch: `4gatepylon/encode-decode-eval`; worktree: `/Users/4gate/conductor/workspaces/StegoICMLMechInterp2026/malabo`. PR1 design is proposed; implementation awaits interface/test clarification.
    - [ ] **PR1 — Synthetic evaluation/testing harness:** Add fast, no-LLM mocks with 2- and 4-token vocabularies, Transformers-style forward/sampling interfaces, validated existing XML control prefixes, yes/no encoding gates, block/chunk and modulo strategies, and configurable delta including obviously biased samples. Provide labelled samples for varied inputs and test that the existing one-bit estimator favors the correct bit under strong encoding.
    - [ ] **PR2 — Multi-bit extraction (independent PR):** Extend the existing estimator to return a `[N, 2]` probability tensor for a known N-bit message layout, preserving full token context while scoring individual chunks or modulo-strided positions. Add appropriate tests and a `TODO(hadriano)` noting that priors over the partitioning strategy could be supported later; the strategy is known for now.
    - [ ] **PR3 — Binary encoding detection:** Implement PyTorch `P(yes | tokens)` with configurable encoding/no-encoding priors and an equal-prior mode, using the known block/modulo layout. Include the step-by-step algebraic derivation in code comments, reuse the synthetic tests, and evaluate ROC/AUROC with threshold sweeps for different tradeoffs. Add a one-line `TODO(hadriano)` about future sampled or exhaustive candidate-message scoring under the actual learned model.
    - [ ] **PR4 — Deferred until review of PRs 1–3:** Scope TBD, probably standalone sampling and steganalysis on real models, with direct CLI flags and optional YAML configuration. The separate training-time callback TODO below remains out of scope for this agent.
- [ ] **TODO(Codex 6)** Trainer should log encode/decode accuracy. We could add a callback.
- [ ] Add plotting scripts for ^ and to generate the outputs for our first 2 (X, Y) experiments.
- [ ] Optionally (if needed) look at scaling laws as a function of train resources to best identify the hero run.
- [ ] Look into how to make runs faster via tinker (https://chatgpt.com/share/6aa72e30-c5ec-83e8-a862-1a12e928a1be), unsloth, or other tooling (worst case we can probably do some custom pytorch + compile workflow?) We might want to skyline (https://chatgpt.com/share/6aa72eac-4444-83e8-8b31-c3dd8bfa6b72) the performance.

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
