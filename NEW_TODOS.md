# NEW TODOS
## State of `ciphers/kirchenbauer_et_al`
The cipher is defined and inspired by existing Watermarking schemes. However, I have done _VERY LITTLE_ actual literature research/review to understand whether there might be a better way. **One first step before proceeding is to read about and understand the problem space better.** Should be (at the very least) [Zamir 2024](https://arxiv.org/abs/2306.09194), [Tool use Stego](https://arxiv.org/abs/2606.28425), [Minimum-entropy coupling](https://arxiv.org/abs/2210.14889). Also, probably some of these from what my friend said:
```
https://www.lesswrong.com/posts/EEvsL9cpgDAxAhTzt/some-negative-steganography-results?utm_source=chatgpt.com

Yeah, David recommended this paper: https://arxiv.org/pdf/2601.22184
https://arxiv.org/abs/2410.08948
https://arxiv.org/abs/2607.00233
https://arxiv.org/html/2607.11363v1
https://arxiv.org/abs/2606.28425
```

The motivation is here: https://docs.google.com/document/d/1W2iJCvNQpCY7OmkK-4zfR88gEcm2MCExs3-dVq1qBGs/edit?usp=drive_link (and it is possibly not the strongest setup/argument). I think it is worth ideating whether the questions we care about actually might be answered in a simpler way (i.e. entirely with prompting, if the goal is to evaluate/compare black-box to white-box or even if the goal is to understand if models could do general-purpose stego in the forward pass).

Then, it's TBD. However, if we want to keep the exact original task plan the things to be done are:
1. Finish collecting good data to finetune GPT-20B-OSS on Tinker for the variable-naming cipher. Some questions remain here:
    - How much guidance should we provide?
    - Do we want to use CoT or not?
    - If we use CoT, do we use real or fake CoT?
    - Do we want GPT-20B-OSS's actual outputs or will Codex's outputs be enough?
    - Do we want to perform SFT, SDPO, or some other alternative? Do we want to consider RL of some kind (We literally have a decoder for the variable-naming cipher).
2. Launch and babysit the GPT-20B-OSS finetuning run. We might need to iterate on variations of (1).
3. Implement the decoder for the pretraining setup with Qwen-3-0.6B-Base. One key mistake from the project is that I did not do this FIRST. I still do not know (nor do I have a way of knowing) whether my intermediate checkpoints are getting better at encoding on the core task. I also don't know if my scheme is working. Ideally, I should implement the MLE-based decoder FIRST and then evaluate Qwen-3-0.6B-Base WITH SCAFFOLDING around sampling (i.e. delta-boosted-sampling) INSTEAD of training. If I cannot get good decoding when doing manual delta-boosted sampling then the entire setup is fucked. Also, this is a cheap way to find the right delta.
4. Figure out what bugs stopped my sweep for the pretraining-based approach in Qwen-3-0.6B-Base. I could not save the logs so you'll need to reproduce this (but generally it looks like a data-loading bug; i.e. some data causes errors upon tokenization). Once this is fixed we need to launch a hero-run. Ideally, you'll need around 4 GPUs for this.

The following PRs/issues are also outstanding, but mainly QOL
- See data for debugging: [#104](https://github.com/4gatepylon/StegoICMLMechInterp2026/pull/104). This is for pretraining-based setup.
- Compare training run lengths based on token counts, not steps: [#81](https://github.com/4gatepylon/StegoICMLMechInterp2026/pull/81). Also, clean up file structure [#110](https://github.com/4gatepylon/StegoICMLMechInterp2026/pull/110). This is for pretraining-based setup.
- Have an easy way to compare different runs before needing to do the sampling evaluation OR supporting the sampling evaluation efficiently over the course of training (i.e. using a trainer callback). This is for pretraining-based setup.
- Understand why GPT-20B-OSS is getting stuck/taking a long time to sample. It seems to be doing very long chains of thought. How could we fix this? Here is the beginning of troubleshooting: [#103](https://github.com/4gatepylon/StegoICMLMechInterp2026/pull/103).
- Almost all the unit tests are from vibe-coding/Astra and I did not review them in detail. Most human review was on the actual code itself (and usually the more complex/algorithmic parts). I think there might be a lot of dead code, bloat, and redundant testing that does not reduce the chance of error. We don't need extensive unit testing, but we need a way to tell if our algorithm is correct that is not just inspection (i.e. some ground truth case). Please add this somehow (i.e. train on ground-truth data using a tiny model, etc...)