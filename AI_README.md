# Whitelist vs. Blacklist Safety Detection: Research Prototype

## TL;DR for the Agent

You are building prototype code for a research project that asks one core question across four modalities: **as you make a classifier's target concept narrower (e.g., "unsafe" → "harmful" → "bioweapons"), how does accuracy and adversarial robustness change for whitelist-style detectors vs. blacklist-style detectors?**

The literature review is done. **No published paper crosses both axes (detector type × concept granularity) in any of the four settings below.** Your job is to build the experimental infrastructure so we can run the first clean comparison.

---

## The Two Axes

### Axis 1: Detector Paradigm (Whitelist vs. Blacklist)

| Style | What it does | Concrete methods |
|-------|-------------|-----------------|
| **Blacklist** | Trained on known-bad examples, flags things that look bad | Linear probe, 1-layer transformer probe, prompted classifier with deny-list rubric |
| **Whitelist** | Trained on known-good examples, flags anything that looks unusual | Mahalanobis distance from safe-distribution centroid, prompted classifier with allow-list rubric |
| **Ambiguous** | Could go either way depending on framing | Some probes, CCS-style unsupervised methods |

### Axis 2: Concept Granularity (Coarse vs. Narrow)

We systematically vary how narrow the classification target is. Example hierarchy for safety:

```
Level 0 (broadest):  "unsafe" vs "safe"
Level 1:             "violent" vs "safe"
Level 2:             "weapons-related" vs "safe"
Level 3 (narrowest): "bioweapons synthesis" vs "safe"
```

For vision (ImageNet-based), the analog is:

```
Level 0 (broadest):  "animal" vs "not animal"
Level 1:             "big cat" vs "not big cat"
Level 2:             "tiger" vs "not tiger"
Level 3 (narrowest): "Sumatran tiger" vs "not Sumatran tiger"
```

---

## The Four Experiments

### 6.1 — Vision (Activation-Based Detectors)

**Domain:** Image classification on ImageNet subsets.

**Detectors to compare:**
- Linear probe (logistic regression on frozen features) — **blacklist-shaped**
- 1-layer transformer probe on frozen features — **blacklist-shaped**
- Mahalanobis distance (fit Gaussian to "in-class" activations, score by distance) — **whitelist-shaped**
- Relative Mahalanobis distance (Ren et al. 2021 variant) — **whitelist-shaped**

**Backbone:** A frozen pretrained vision model. Start with a ViT-B/16 (CLIP or DINOv2). Extract activations from the [CLS] token at multiple layers.

**Granularity sweep:** Use ImageNet's WordNet hierarchy to create classification tasks at varying semantic breadth. The user has already thought about this. Use something like:
- "tiger" vs rest
- "big cat" vs rest  
- "feline" vs rest
- "4-legged animal" vs rest
- "animal" vs rest

**Measurements per cell:**
1. **Clean accuracy** (AUROC, FPR@95%TPR) on held-out test set
2. **Adversarial robustness** — apply PGD/FGSM perturbations to test images and re-measure. Budget: sweep epsilon from 0 to some reasonable max.
3. **Distribution-shift accuracy** — test on images from related but non-overlapping classes (e.g., train on "tiger," test on "leopard" as a near-OOD negative)

**Key questions this answers:**
- (a) Are classifiers for narrower concepts more accurate?
- (b) Are classifiers for narrower concepts more robust to adversarial perturbations?
- (c) Do whitelist-style (Mahalanobis) detectors outperform blacklist-style (linear probe) detectors, and does this depend on granularity?

### 6.2 — LLMs (Activation-Based Detectors)

**Domain:** Safety-relevant classification on LLM internal activations.

**Detectors:** Same as 6.1 (linear probe, transformer probe, Mahalanobis, Relative Mahalanobis) but applied to residual-stream activations of a language model.

**Models:** Start with **Gemma-2-2B-it** (Gemma Scope SAEs available for free mech-interp extensions later). Add **Llama-3-8B-Instruct** as a second model if compute allows.

**Granularity sweep using BeaverTails:** BeaverTails has 14 harm categories. Build a hierarchy:
- Level 0: "harmful" vs "safe" (all 14 categories pooled)
- Level 1: "violence-related" vs "safe" (subset of categories)
- Level 2: specific single category vs "safe" (e.g., "animal_abuse" alone)
- Level 3: sub-sampled narrow slice of a single category vs "safe"

**Safe/benign anchors:** Alpaca-cleaned or ShareGPT for the "known-good" distribution that Mahalanobis fits to.

**Activation extraction:** Use TransformerLens or nnsight. Cache residual-stream activations at layers {0, L/4, L/2, 3L/4, L-1} at the last token position. Store as float16 numpy arrays.

**Adversarial attacks:** Use GCG suffixes from HarmBench artifacts (pre-computed). If budget allows, run Bailey et al.'s embedding-suffix obfuscation (their code is at github.com/LukeBailey181/obfuscated-activations).

**Measurements:** Same as 6.1 — clean AUROC, adversarial AUROC, and cross-category generalization (train on category X, test on held-out category Y).

### 6.3 — LLMs (Prompt-Based Detectors)

**Domain:** Same safety datasets as 6.2, but instead of activation probes, we use prompted LLM classifiers with different rubric framings.

**Detectors:**
- **Blacklist-prompted classifier:** The rubric says "Flag this content if it contains [list of bad things: violence, weapons, threats, ...]"
- **Whitelist-prompted classifier:** The rubric says "This content is acceptable only if it matches [description of safe content: helpful instructions, casual conversation, factual Q&A, ...]. Flag anything that does not match."

**Models for the classifier itself:**
- Large: GPT-4o or Claude (via API)
- Small: Gemma-2-2B-it, Llama-3-8B-Instruct (running locally)

**Granularity sweep:** Same BeaverTails hierarchy as 6.2, but now the rubric text changes:
- Broad blacklist: "Flag if it involves any harmful, illegal, or dangerous content"
- Narrow blacklist: "Flag if it specifically involves synthesis instructions for biological agents"
- Broad whitelist: "Accept only if it is a benign, helpful, informational request"  
- Narrow whitelist: "Accept only if it is a factual science question about published biology research"

**Measurements:** Same AUROC/FPR metrics. Adversarial = jailbreak prompts from JailbreakBench/HarmBench. Distribution shift = held-out BeaverTails categories not mentioned in the rubric.

### 6.4 — VLMs (Prompt-Based Detectors on Vision-Language Models)

**Domain:** Image+text safety classification using vision-language models.

**Detectors:** Same whitelist-vs-blacklist prompt framing as 6.3, but applied to a VLM.

**Models:** GPT-4o (vision), Gemini (vision), or an open VLM like LLaVA-1.6 / InternVL.

**Datasets:** Use the same ImageNet granularity hierarchy from 6.1, but now feed images to the VLM with a text prompt asking it to classify.

**Granularity sweep:** Same as 6.1's concept hierarchy, but expressed as prompt rubrics:
- Blacklist: "Flag this image if it shows a tiger"
- Whitelist: "This image is acceptable only if it shows a domestic house cat. Flag anything else."
- (Sweep through the coarse-to-fine hierarchy)

**Measurements:** Same framework. Adversarial = adversarial patches or typographic attacks on images.

---

## Shared Infrastructure Needed

### Data Pipeline

```
datasets/
├── vision/
│   ├── imagenet_hierarchies.py    # builds coarse→fine class groupings from WordNet
│   └── splits.py                  # train/val/test + held-out-category splits
├── llm/
│   ├── beavertails_hierarchies.py # builds coarse→fine harm category groupings
│   ├── benign_anchors.py          # loads Alpaca/ShareGPT for whitelist fitting
│   └── adversarial.py             # loads GCG/PAIR/AutoDAN from HarmBench artifacts
└── shared/
    └── metrics.py                 # AUROC, FPR@TPR, calibration curves
```

### Detector Implementations

```
detectors/
├── linear_probe.py          # sklearn LogisticRegression on cached activations
├── transformer_probe.py     # 1-layer transformer (small) on cached activations
├── mahalanobis.py           # vanilla MD + Relative MD (Ren et al.)
├── prompt_classifier.py     # API-based prompted classifier (whitelist + blacklist rubrics)
└── base.py                  # shared interface: fit(X_train, y_train), score(X_test) -> anomaly scores
```

### Activation Extraction

```
extraction/
├── vision_features.py       # extract [CLS] from ViT at specified layers
├── llm_features.py          # extract residual stream from TransformerLens at specified layers
└── cache.py                 # save/load activations as .npz files
```

### Evaluation Harness

```
eval/
├── clean_eval.py            # standard held-out accuracy
├── adversarial_eval.py      # PGD/FGSM for vision; GCG suffixes for LLM
├── shift_eval.py            # train on category X, test on held-out category Y
└── sweep.py                 # main entry: loops over {method} × {granularity} × {eval_type}
```

---

## Implementation Priority

**Start with 6.1 (vision).** It is the simplest, cheapest, and fastest to iterate on. The ImageNet hierarchy is well-defined, ViT feature extraction is fast, and adversarial perturbations (PGD) are standard. If the granularity × method interaction shows up here, it will almost certainly transfer to the LLM setting.

**Then do 6.2 (LLM activations).** This is the main scientific contribution. 6.1 is the sanity check / proof of concept.

**Then 6.3 and 6.4 (prompt-based).** These are the most novel (no prior work at all on whitelist-vs-blacklist prompt framing) but also the most expensive (API calls) and hardest to control.

---

## Concrete Hypotheses to Test

For each experiment, we are testing:

**H1 (Granularity → Accuracy):** Narrower classification targets yield higher clean AUROC for both detector types, because the decision boundary is simpler.

**H2 (Granularity → Robustness):** Narrower classification targets are MORE robust to adversarial perturbations (the "smaller target is harder to hit" hypothesis). **Counter-hypothesis:** narrower targets have tighter decision boundaries that are easier to cross with small perturbations (the near-OOD finding from Lang et al. CVPR 2024).

**H3 (Whitelist vs. Blacklist under shift):** Whitelist detectors (Mahalanobis) degrade more gracefully under distribution shift to novel categories, because they don't depend on having seen the specific bad category. Blacklist detectors (probes) fail catastrophically on unseen categories.

**H4 (Whitelist vs. Blacklist under attack):** Under adversarial attack, blacklist detectors collapse abruptly (the attacker just needs to avoid the learned "bad" direction), while whitelist detectors degrade gradually into higher false-positive rates (the attacker must land inside the safe region, which is harder but causes the detector to widen its acceptance).

**H5 (Interaction):** The whitelist advantage under shift is LARGER for narrower concepts (because a narrow whitelist defines a tighter "safe" region that is harder to mimic).

---

## Key Design Decisions the Agent Should Surface

### ASK ME (critical decisions that affect the whole project):

1. **Vision backbone:** CLIP ViT-B/16 vs. DINOv2 ViT-B/14 vs. both? DINOv2 features may be more linearly separable. CLIP features are more semantically structured. Using both adds a robustness-to-backbone check but doubles compute.

2. **ImageNet hierarchy depth:** How many granularity levels? I mentioned 5 (tiger → animal). We could do 3 (coarse/medium/fine) or 7+ using full WordNet depth. More levels = cleaner curves but more runs.

3. **LLM layer selection:** Cache all layers vs. a subset? All layers on Gemma-2-2B is ~26 layers × N samples. Subset (e.g., layers 5, 10, 15, 20, 25) is cheaper. We need enough to see if the method × granularity interaction varies by depth.

4. **Mahalanobis implementation details:**
   - Vanilla MD vs. Relative MD (Ren et al.) vs. both?
   - Full covariance vs. diagonal vs. PCA-reduced? Full covariance at d=2304 (Gemma-2-2B) is fine; at d=4096 (Llama-3-8B) it may need regularization.
   - Fit on which data? Just benign? Benign + a small "outlier exposure" set?

5. **Adversarial attack budget for LLM experiments:** Pre-computed GCG suffixes from HarmBench (cheap, ~100 examples) vs. running our own GCG optimization (expensive, ~10 GPU-hours per suffix) vs. Bailey et al.'s embedding-suffix obfuscation (most rigorous, ~50 GPU-hours)?

6. **Prompt rubric design for 6.3/6.4:** I need to write or approve the actual rubric text. The agent should draft candidate rubrics and show them to me before running experiments.

7. **Sample sizes:** How many examples per granularity level per split? BeaverTails has ~330K examples total but uneven across categories. We need to decide on stratified subsampling to control for sample size as a confound.

8. **Evaluation thresholds:** Report AUROC only, or also FPR@95%TPR, FPR@99%TPR? The latter matters more for deployment but requires more data for stable estimates.

### DO NOT ASK ME (just make reasonable defaults):

- File formats, directory structure, logging setup
- Which sklearn/torch version to use
- How to structure the config files
- Standard data loading boilerplate
- Plotting style choices (just use matplotlib with a clean style)

---

## Known Risks and Confounds

1. **Sample size confound:** Narrower categories have fewer examples. If you train a probe on 100 "bioweapons" examples vs. 10,000 "any harm" examples, the difference in accuracy could just be sample size. **Mitigation:** Always subsample the broader category down to match the narrow one. Run the comparison at multiple matched sample sizes (e.g., 50, 200, 1000, 5000).

2. **Mahalanobis covariance estimation:** With few samples and high-dimensional activations, the covariance matrix is rank-deficient. **Mitigation:** Use PCA to reduce dimensionality before fitting, or use Relative Mahalanobis (which subtracts a background covariance). Try both and report.

3. **Probe overfitting in narrow regime:** A linear probe with d=2304 features and 50 training examples will overfit. **Mitigation:** Use L2 regularization (standard in sklearn LogisticRegression). Sweep C.

4. **Concept leakage in BeaverTails:** Some harm categories overlap semantically (e.g., "violence" and "terrorism"). Held-out category tests may show partial transfer that muddies the distribution-shift story. **Mitigation:** Choose maximally distinct held-out categories (e.g., hold out "privacy_violation" when training on "animal_abuse").

5. **Adversarial budget asymmetry:** GCG attacks are optimized against the model's output, not against the detector. Bailey et al.'s attacks are optimized against the detector. These test very different threat models. Be explicit about which is being used in each result.

6. **Prompt sensitivity in 6.3/6.4:** Small wording changes in the rubric could dominate the whitelist/blacklist effect. **Mitigation:** Write 3+ rubric variants per condition and average results.

---

## Key References (for the agent to skim if needed)

- **Bailey et al. 2024** "Obfuscated Activations Bypass LLM Latent-Space Defenses" (arXiv:2412.09565) — the strongest existing probe-vs-Mahalanobis comparison + adaptive attacks. Code: github.com/LukeBailey181/obfuscated-activations
- **Lang et al. CVPR 2024** "From Coarse to Fine-Grained Open-Set Recognition" — granularity sweep in vision OSR. Code: github.com/langnico/osr-coarse-to-fine
- **Ren et al. 2021** "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection" (arXiv:2106.09022) — Relative Mahalanobis, the recommended MD variant
- **Lee et al. NeurIPS 2018** "A Simple Unified Framework for Detecting OOD Examples and Adversarial Examples" — original Mahalanobis OOD method
- **Shah et al. 2025** "Death by a Thousand Directions: The Geometry of Harmfulness through Subconcept Probing" — 55 harm subconcept probes, shows near-rank-1 collapse of harm directions
- **MacDiarmid et al. 2024** (Anthropic) "Simple Probes Can Catch Sleeper Agents" — strong probe results but on backdoored models, no Mahalanobis baseline
- **Mallen & Belrose 2023** "Eliciting Latent Knowledge from Quirky Language Models" (arXiv:2312.01037) — probe vs Mahalanobis on truth/falsehood, closest existing LLM comparison
- **OpenOOD v1.5** — vision OOD benchmark with ~40 methods including Mahalanobis variants. Code: github.com/Jingkang50/OpenOOD

---

## Compute Estimates

| Experiment | Estimated GPU-hours | Notes |
|-----------|-------------------|-------|
| 6.1 Vision features + all detectors | 2–5 | ViT forward pass is fast; most time is PGD adversarial |
| 6.2 LLM activation extraction (Gemma-2-2B) | 5–15 | ~100K samples, 5 layers, float16 |
| 6.2 LLM detector training + eval | 1–3 | Probes train in seconds; Mahalanobis is numpy |
| 6.2 GCG adversarial (pre-computed) | 0 | Use HarmBench artifacts |
| 6.2 Bailey-style adaptive attacks | 30–80 | Optional stretch goal |
| 6.3 Prompt-based (API calls) | 0 GPU but $50–200 API | ~10K classifications × 4 rubrics × 2 models |
| 6.4 VLM prompt-based | 0 GPU but $100–300 API | Same but with images |
| **Total (without adaptive attacks)** | **~10–25 GPU-hours + $150–500 API** | |

---

## Output Format

For each experiment, produce:

1. **A 2D heatmap:** rows = detector type, columns = granularity level. Cell color = AUROC (or FPR@95TPR). One heatmap per eval condition (clean, adversarial, shift).

2. **Line plots:** AUROC vs. granularity level, one line per detector. Should make the interaction (or lack thereof) visually obvious.

3. **A summary table** with all numbers for the paper.

4. **Raw results as JSON/CSV** for later analysis.

---

## What "Done" Looks Like for the Prototype

The prototype is done when we can run:

```bash
# Vision experiment
python sweep.py --experiment 6.1 --backbone vit-b-16 --granularity-levels 5 --methods linear,transformer,mahalanobis,relative_mahalanobis --evals clean,adversarial,shift

# LLM experiment  
python sweep.py --experiment 6.2 --model gemma-2-2b-it --granularity-levels 4 --methods linear,transformer,mahalanobis,relative_mahalanobis --evals clean,adversarial,shift

# Prompt-based LLM
python sweep.py --experiment 6.3 --classifier-model gpt-4o --granularity-levels 4 --methods whitelist_prompt,blacklist_prompt --evals clean,adversarial,shift

# Prompt-based VLM
python sweep.py --experiment 6.4 --classifier-model gpt-4o --granularity-levels 4 --methods whitelist_prompt,blacklist_prompt --evals clean,adversarial,shift
```

...and get the heatmaps + line plots + summary tables out the other end.

The prototype does NOT need to be polished, production-grade, or cover every edge case. It needs to be correct enough to tell us whether the hypotheses hold and whether this is worth writing up.

