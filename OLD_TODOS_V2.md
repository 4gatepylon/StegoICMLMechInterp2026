# 1. Refactor for general model/data
I also realize, this type of distillation training on pretraining data is a more generalizeable feature.

Please refactor the code to make a trainer class that will train using this distillation objective and LoRa. The trainer class would take in a dataset (you can move our fineweb preprocessing in particular OUT of the class so that we can support other preprocessing pipelines seperately) and then just train using the LoRA adaptors as requested.

# 2. Refactor for SFT with instruction tuning supported
Should allow there to be a finetuning mask and any other metadata so we don't have to train on all the tokens.

# 3. Refactor for RL? TBD a few versions (but mainly would modify GRPO Trainer):
- Rollout from the on-policy model itself then penalize with KL penalty on specified tokens from the boosted/suppressed policy.
- Rollout from the base model itself (so forward KL, I think this is best)
- Maybe the other but idk.... Maybe allow a custom variation functional

# 4. How many tokens for the bits?
# 5. How many tokens for the signifiers?
# 7. Be able to properly chunk document? Wtf is going on here => Batch sampler/sampler + larger scale
# 8. Remove gotchas from root paths
# 9. Not enough validation wtf?
# 12 add bear-type

NOW HAPPENING
# 1. Use SFTTrainer instead of whatever fucking shit we are using
# 10. Improve type annotation in model module (wtf why "Any" everywhere?)
# 11. Jaxtyping - DONE
# 6. Be able to set batch size

NOTE: we need around 400-600 steps to roughly "converge" on train loss
