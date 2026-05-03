# Missing Pieces, Known Bugs, and Gotchas

Session UUID: CA6756DD-1032-4898-91C3-F880AD829C99
Agent: wbdetect (whitelist/blacklist detection) — NOT the stego agent
Thorough audit as of 2026-04-26. Covers library (`wbdetect/`), experiments, and gaps relative to the spec in `AI_README.md`.

---

## Likely Bugs in Current Code

### 1. CIFAR-100 class IDs in `vision_hierarchy.py` are partially wrong
The `insects` set is `{0, 51, 53, 57, 83}` with a comment saying "bee, mushroom(?), ...". CIFAR-100 fine label 51 is actually "mushroom" (under food_containers or vegetables, not insects). The correct insect IDs are `{0: apple... }` — actually all of these need verification against the actual CIFAR-100 label list. The `large_carnivores` set (bear=3, leopard=42, lion=43, tiger=88, wolf=97) is correct, but the broader `all_animals` set likely has wrong class IDs for aquatic_mammals, small_mammals, insects, reptiles, and fish. **Fix: look up the actual CIFAR-100 fine_label → class_name mapping and verify every ID.**

### 2. `ActivationExtractor.extract()` doesn't use `VisionExtractor.extract_features()` correctly
The base class `extract()` calls `self.extract_features(tensor)` inside the hook, where `tensor = out[0] if isinstance(out, tuple) else out`. But `TransformerEncoderLayer` outputs a tensor directly (not a tuple), and its shape is `(batch, seq_len, d_model)`. The `VisionExtractor.extract_features()` checks `ndim == 3` and takes `[:, cls_position, :]`. This works. But for some HuggingFace models, `encoder.layer.N` returns a `BaseModelOutput` (a named tuple), and `out[0]` would be the hidden states tensor. This should work but hasn't been tested with real HF models — might break depending on model architecture.

### 3. `LLMExtractor.extract_features()` with `token_position=-1` and padding
When using `token_position=-1` (last token), padded sequences will have the "last token" be a PAD token, not the actual last content token. **Fix: need to use `attention_mask` to find the true last non-pad position per sample.** This is wrong for any batched LLM extraction with variable-length inputs.

### 4. `TransformerProbe` doesn't re-initialize on repeated `fit()` calls
In exp01, the same detector object is reused across granularity levels. `fit()` creates a new `self.net`, which is fine. But the old optimizer state, etc., are gone. This is actually OK since we create fresh each time. **However**, `d_model` can differ between layers if someone passes activations from different layers — the net is rebuilt each time so this is fine. No bug here, but worth noting.

### 5. `fpr_at_tpr()` uses `searchsorted` on potentially non-monotonic TPR
`sklearn.metrics.roc_curve` returns TPR in non-decreasing order, so `searchsorted` works. But edge case: if all scores are identical (degenerate classifier), ROC curve has only 2 points and `searchsorted` might return an out-of-bounds index. The `if idx >= len(fpr)` guard handles this. Probably fine.

### 6. exp01 has dead code: `_, y_balanced = hierarchy.subsample_balanced(...)` on line 186
The result is never used — the balancing is done manually on lines 187-195. This is just dead code, not a bug.

### 7. Mahalanobis score direction mismatch with blacklist detectors
For blacklist detectors (probes), higher score = more likely to be flagged = more likely harmful. For Mahalanobis (whitelist), higher score = more distant from safe distribution = more likely harmful. Both go the same direction for `compute_metrics`, which expects higher = more positive. **This is correct as implemented.** But if someone inverts the convention, AUROC will read as 1-AUROC. Worth documenting clearly.

### 8. `PromptClassifier.score()` accepts `np.ndarray` but actually iterates as strings
The `Detector` ABC declares `score(X: np.ndarray) -> np.ndarray`. For PromptClassifier, X is expected to be strings, not float arrays. This type mismatch means the interface isn't truly uniform — you can't pass activation arrays to it. The `score_texts()` method is a workaround, but the ABC contract is violated. Not a runtime bug (it works), but it's an interface design smell.

---

## Missing Implementation (relative to spec)

### High Priority (needed for 6.1 to be real)

1. **No pretrained model integration in any experiment.** exp01 uses a random TinyViT. Need to wire in DINOv2-small, CLIP, or any real pretrained ViT and verify the extractor works with it.

2. **No adversarial evaluation** (`eval/adversarial_eval.py` from spec). Need PGD/FGSM for vision, GCG suffixes for LLM. The spec lists this as a primary measurement.

3. **No distribution-shift evaluation** (`eval/shift_eval.py` from spec). Train on category X, test on held-out category Y. Critical for H3 (whitelist vs blacklist under shift).

4. **No sweep runner** (`eval/sweep.py` from spec). The main entry point that loops over `{method} × {granularity} × {eval_type}` and produces heatmaps + line plots.

5. **No plotting code.** The spec wants 2D heatmaps, line plots (AUROC vs granularity), and summary tables for the paper. None exist.

6. **No multi-layer analysis.** Both extractors support multiple layers, and exp01 extracts from all 4 layers, but only the last layer is used for detection. The spec asks about how results vary by depth.

7. **No ImageNet WordNet hierarchy builder** (`datasets/imagenet_hierarchies.py` from spec). Only CIFAR-100 hierarchy exists. For real 6.1 results, need Tiny ImageNet or full ImageNet with NLTK WordNet lookups.

### Medium Priority (needed for 6.2)

8. **LLMExtractor is completely untested end-to-end.** No experiment loads a real LLM, extracts activations, and runs detectors on them. The padding bug (#3 above) will bite here.

9. **No BeaverTails category key verification.** The category strings in `get_beavertails_hierarchy()` are typed by hand. If the actual BeaverTails dataset uses slightly different key strings (e.g., `"animal abuse"` vs `"animal_abuse"`), the hierarchy will silently produce all-zero labels. **Must verify against the actual dataset.**

10. **No StemQA → activation extraction pipeline.** `load_stemqa()` loads text data, but there's no code to tokenize it, run through an LLM, and cache activations. The `LLMExtractor` class exists but hasn't been wired to StemQA.

11. **No in-domain vs OOD-benign vs OOD-malicious framing in the datasets.** The spec and user discussion identified a 3-way split (StemQA bio = in-domain, other StemQA subjects = OOD-benign, BeaverTails = OOD-malicious). The hierarchy classes only handle binary (positive vs negative). Need to extend for 3-way evaluation.

### Lower Priority (6.3, 6.4, polish)

12. **Prompt classifier rubrics are drafts.** The spec says to write 3+ variants per condition and average results to control for prompt sensitivity. Currently only 1 rubric per granularity level exists.

13. **No VLM prompt classifier** (6.4). The `PromptClassifier` is text-only. Need a version that sends images via the OpenAI vision API.

14. **No experiment for 6.3** (LLM prompt-based). exp02 exists but is a smoke test, not the full granularity sweep.

15. **No config system.** The spec mentions a config-driven sweep. Currently all hyperparameters are hardcoded in experiment scripts. Not blocking for prototyping but needed for systematic sweeps.

16. **No result aggregation across experiments.** Need code to load multiple result JSONs and produce the combined heatmaps/plots.

---

## Gotchas and Things That Will Surprise You

### Model-specific hook outputs
- **DINOv2** (`Dinov2Model`): `encoder.layer.N` outputs are `tuple(hidden_states_tensor,)` — so `out[0]` works. But if you use `Dinov2ForImageClassification`, the layer structure is nested differently.
- **CLIP** (`CLIPVisionModel`): layers are at `vision_model.encoder.layers.N`. The `get_vit_layer_names()` function looks for `encoder.layer.N` (no 's') — it will fail on CLIP. Need to add `encoder.layers.N` pattern.
- **GPT-2** via HuggingFace: `transformer.h.N` is a `GPT2Block` that returns `tuple(hidden_states, ...)`. So `out[0]` works.
- **Pythia**: `gpt_neox.layers.N` is a `GPTNeoXLayer` that returns `tuple(hidden_states, ...)`. Same.
- **Gemma**: `model.layers.N` returns `tuple(hidden_states, ...)`. Should work.
- **General issue**: some layers return a single tensor, others return tuples. The `out[0] if isinstance(out, tuple) else out` pattern handles this, but `BaseModelOutputWithPast` (which is not a tuple but is iterable) might break it.

### CIFAR-100 at 32x32 is too small for real ViTs
DINOv2 expects 224x224 input. CIFAR-100 images are 32x32. You'd need to resize/upsample, which destroys the point of using a pretrained model on natural images. **For real experiments, use Tiny ImageNet (64x64, still small) or proper ImageNet subsets (224x224).**

### BeaverTails is multi-label
A single BeaverTails example can trigger multiple harm categories simultaneously. The `get_binary_labels()` uses `any()`, which means if an example is labeled both "violence" and "drug_abuse", it'll be positive at both the violence-cluster level and the all-harmful level. This is correct behavior but means the narrow/broad comparison isn't cleanly nested the way ImageNet classes are. Consider filtering to single-label examples for cleaner experiments.

### Mahalanobis with very few safe samples
At the "tiger" granularity level in exp01, there were only 8 tiger examples. After balancing, there are 8 safe samples. The Mahalanobis covariance is estimated from 8 points in 64-dimensional space — this is rank-deficient even with regularization. PCA to `min(pca_dim, n-1)` handles the crash, but the resulting distance is essentially meaningless. **Need at least ~2x the feature dimension in safe samples for Mahalanobis to be informative.**

### The `subsample_balanced` method returns `(X[idx], binary[idx])` but exp01 doesn't use it
exp01 calls `subsample_balanced` and ignores the result, then re-implements balancing manually. This is because `subsample_balanced` takes feature matrix `X` but the experiment needs the balanced indices to index into both `X_train` and `train_labels`. The method should probably return indices instead of data.

### Rate limiting in PromptClassifier is per-instance
If you create multiple `PromptClassifier` instances and call `score()` in sequence, each has its own rate limiter. So 2 classifiers × 60 RPM = 120 RPM total, which may exceed API limits. No global rate limiter.

### No caching of API calls
Every run of exp02 re-classifies all texts from scratch. At $0.15/1M tokens for gpt-4o-mini this is cheap, but for gpt-4o at $5/1M tokens and 1000+ samples across rubric variants, costs add up. Need a local cache (hash of prompt+text → response).

### `ActivationCache.save()` uses `np.savez_compressed` with `allow_pickle=True` on load
The `metadata` field is serialized via `np.array(v)` which can produce object arrays for non-numeric metadata. Loading with `allow_pickle=True` is a security risk if loading untrusted cache files. Fine for local research but worth noting.

---

## Spec Items Not Yet Addressed

| Spec Item | Status |
|-----------|--------|
| ImageNet WordNet hierarchy | Not started |
| Tiny ImageNet / real ImageNet data pipeline | Not started |
| Pretrained ViT feature extraction | Not started |
| PGD/FGSM adversarial attacks (vision) | Not started |
| GCG suffix loading from HarmBench | Not started |
| TransformerLens / nnsight integration | Decided against (using plain hooks) |
| Gemma-2-2B / Llama-3-8B support | Coded but untested |
| Multiple rubric variants (3+) | Only 1 per granularity |
| VLM prompt classifier (6.4) | Not started |
| Calibration curves | Not started |
| Heatmap + line plot generation | Not started |
| CSV/JSON raw results export | Partially done (JSON only) |
| `sweep.py` main entry point | Not started |
| Sample size confound control (matched sizes at multiple N) | Partially done (balance only, not multi-N sweep) |
| Cross-category generalization eval | Not started |
