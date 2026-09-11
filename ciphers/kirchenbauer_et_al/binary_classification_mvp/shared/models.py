"""Device handling, Hugging Face model loading, and LoRA teacher passes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.configuration import ModelSpec


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"Unknown dtype: {name}")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch through the Transformers utility."""

    from transformers import set_seed as set_transformers_seed

    set_transformers_seed(seed)


def load_tokenizer(path_or_name: str) -> Any:
    """Load the tokenizer used by every training, dry-run, and evaluation entry point."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model_config(source: dict[str, Any] | Path | None) -> Any | None:
    """Load optional architecture settings used by every base-model loader."""

    if source is None:
        return None

    from transformers import AutoConfig

    if isinstance(source, Path):
        raw = source.read_text()
        values = json.loads(raw) if source.suffix.lower() == ".json" else yaml.safe_load(raw)
    else:
        values = dict(source)
    model_type = values.pop("model_type", None)
    if model_type is None:
        raise ValueError("A model configuration must define model_type")
    return AutoConfig.for_model(model_type, **values)


def _load_base_model(spec: ModelSpec, dtype: torch.dtype) -> Any:
    """Instantiate the base used to construct teacher, student, and inference models."""

    from transformers import AutoModelForCausalLM

    model_config = _load_model_config(spec.config)
    if spec.weights is not None:
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if model_config is not None:
            kwargs["config"] = model_config
        return AutoModelForCausalLM.from_pretrained(spec.weights, **kwargs)
    if model_config is None:
        raise ValueError("Random initialization requires a model configuration")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.initialization_seed)
        model = AutoModelForCausalLM.from_config(model_config)
    return model.to(dtype=dtype)


def describe_model_spec(spec: ModelSpec) -> str:
    if spec.weights is not None:
        return spec.weights
    return f"random:{spec.initialization_seed}:{spec.config!s}"


def _debug_logits(model: Any) -> torch.Tensor:
    """Run a tiny deterministic forward used only by debug assertions."""

    was_training = model.training
    model.eval()
    vocab_size = model.config.vocab_size
    bos_token_id = model.config.bos_token_id
    first_token = bos_token_id if bos_token_id is not None else 0
    input_ids = torch.tensor([[first_token, 0, 1, 2]], dtype=torch.long)
    if max(input_ids.flatten()) >= vocab_size:
        input_ids %= vocab_size
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits.detach().cpu()
    if was_training:
        model.train()
    return logits


def load_trainable_lora_model(
    spec: ModelSpec,
    dtype: torch.dtype,
    *,
    adapter_path: str | None,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[Any, str]:
    """Load the LoRA student trained by both finetuning stages."""

    from peft import LoraConfig, PeftModel, get_peft_model

    base_model = _load_base_model(spec, dtype)
    if __debug__:
        expected_base_logits = _debug_logits(base_model)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    else:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(base_model, config)
    if __debug__:
        with model.disable_adapter():
            disabled_adapter_logits = _debug_logits(model)
        assert torch.equal(expected_base_logits, disabled_adapter_logits), (
            "Disabling LoRA did not reproduce the pristine base-model logits; "
            f"maximum difference={float((expected_base_logits - disabled_adapter_logits).abs().max())}"
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    return model, describe_model_spec(spec)


def load_inference_model(
    spec: ModelSpec,
    adapter_path: str | None,
    dtype: torch.dtype,
) -> Any:
    """Load the stage-2 adapter used by held-out generation evaluation."""

    from peft import PeftModel

    base_model = _load_base_model(spec, dtype)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.eval()
    model.config.use_cache = True
    return model


def reference_logits_with_disabled_adapter(
    model: Any,
    input_ids: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute original-policy teacher logits using the training model with LoRA disabled."""

    assert model.get_model_status().enabled is True, "LoRA adapters must be enabled before the teacher pass"
    was_training = model.training
    model.eval()
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    try:
        with model.disable_adapter(), torch.inference_mode():
            assert model.get_model_status().enabled is False, "LoRA adapters remained active during the teacher pass"
            logits = (
                model(
                    input_ids=ids,
                    attention_mask=attention_mask,
                )
                .logits[:, :-1]
                .to("cpu")
            )
    finally:
        if was_training:
            model.train()
    assert model.get_model_status().enabled is True, "LoRA adapters were not restored after the teacher pass"
    return logits
