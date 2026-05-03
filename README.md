User's experiments:
```
    6.1. Linear vs. Mahalanobis vs. 1-layer transformer probes as you scale the granularity of data in vision. We are primarily curious in: (a) are classifiers for narrower settings more accurate? (b) are classifiers in narrow settings more robust to adversarial perturbations? (c) are classifiers the use "closed" topology or "whitelist" more accurate 
    6.2. (6.1) but with LLMs instead.
    6.3. (6.1) but with LLMs + prompt-based probes specifically. This will share the same dataset(s) as (6.2) but we will create prompt-based probes with rubrics for large models (OpenAI models) as well as for smallish models (i.e. Gemma models). Also, instead of mahalanobis vs. linear vs. transformer, etc... we will do whitelist vs. blacklist (i.e. the rubric will say "classify only things like XYZ" vs. "classify out anything that blah blah blah"
    6.4. (6.1) but with VLMs + prompt-based probes specifically. The two prompt classes are (a) whitelist and (b) blacklist per class split (so it's analogous to (6.3).

each is like 2 axes: black vs. white (mahalanobis = white, linear/transformer = white/unknown, etc..) and coarse vs narrow classification
```

AI's cleaned up description: look into AI_README.md
