"""Device handling, Hugging Face model loading, LoRA, and CPU swapping."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch


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
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adapter_base_model_name(path_or_name: str) -> str:
    adapter_config = Path(path_or_name) / "adapter_config.json"
    if not adapter_config.is_file():
        return path_or_name
    config = json.loads(adapter_config.read_text())
    return config["base_model_name_or_path"]


def load_tokenizer(path_or_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_reference_model(model_name: str, dtype: torch.dtype) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    return model


def load_trainable_lora_model(
    path_or_name: str,
    dtype: torch.dtype,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[Any, str]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    adapter_config = Path(path_or_name) / "adapter_config.json"
    base_name = adapter_base_model_name(path_or_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_config.is_file():
        model = PeftModel.from_pretrained(base_model, path_or_name, is_trainable=True)
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
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    return model, base_name


def load_inference_model(path_or_name: str, dtype: torch.dtype) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_config = Path(path_or_name) / "adapter_config.json"
    if adapter_config.is_file():
        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_base_model_name(path_or_name),
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, path_or_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path_or_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    model.eval()
    model.config.use_cache = True
    return model


def reference_logits_with_swap(
    reference_model: Any,
    student_model: Any,
    input_ids: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute raw-text teacher logits while only one model occupies the device."""

    student_model.to("cpu")
    clear_device_cache(device)
    reference_model.to(device)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    with torch.inference_mode():
        logits = (
            reference_model(
                input_ids=ids,
                attention_mask=attention_mask,
            )
            .logits[:, :-1]
            .to("cpu")
        )
    reference_model.to("cpu")
    del ids, attention_mask
    clear_device_cache(device)
    student_model.to(device)
    return logits
