from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Any
from torch.utils.hooks import RemovableHandle
from contextlib import contextmanager

# Adapted from SAEScoping/sae_scoping/training/utils/hooks/pt_hooks.py


class NamedForwardHooks:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: Dict[str, RemovableHandle] = {}

    def add_hook(self, name: str, hook_fn: Callable, pre: bool = False):
        named_modules = dict(self.model.named_modules())
        if name not in named_modules:
            raise ValueError(
                f"No module named '{name}' found in the model: "
                f"{list(n for n, _ in self.model.named_modules())}."
            )
        module = named_modules[name]
        handle = (
            module.register_forward_hook(
                lambda mod, inp, out: hook_fn(self, name, mod, inp, out)
            )
            if not pre
            else module.register_forward_pre_hook(
                lambda mod, inp: hook_fn(self, name, mod, inp, None)
            )
        )
        self.hooks[name] = handle

    def remove_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()


@contextmanager
def named_forward_hooks(
    model: nn.Module,
    hook_dict: Dict[str, Callable | tuple[Callable, bool]],
):
    hooks = NamedForwardHooks(model)
    for name, hook_fn_obj in hook_dict.items():
        hook_fn = hook_fn_obj if isinstance(hook_fn_obj, Callable) else hook_fn_obj[0]
        pre = hook_fn_obj[1] if isinstance(hook_fn_obj, tuple) else False
        hooks.add_hook(name, hook_fn, pre=pre)
    try:
        yield hooks
    finally:
        hooks.remove_hooks()


def filter_hook_fn(
    filter_fn: Callable[[torch.Tensor, ...], torch.Tensor] | nn.Module,
    hooks: NamedForwardHooks,
    name: str,
    mod: nn.Module,
    inp: Optional[tuple[torch.Tensor, ...] | torch.Tensor],
    out: Optional[tuple[torch.Tensor, ...] | torch.Tensor],
) -> None:
    in_val = inp if out is None else out
    in_pt = in_val[0] if isinstance(in_val, tuple) else in_val
    assert isinstance(in_pt, torch.Tensor), f"Expected a tensor, got {type(in_pt)}"
    out_pt = filter_fn(in_pt)
    out_val = (
        tuple([out_pt] + list(in_val[1:])) if isinstance(in_val, tuple) else out_pt
    )
    return out_val
