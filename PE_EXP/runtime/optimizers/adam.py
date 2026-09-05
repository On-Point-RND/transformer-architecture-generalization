import inspect

import torch


def _fused(cls, device_type):
    available = "fused" in inspect.signature(cls).parameters
    return {"fused": True} if available and device_type == "cuda" else {}


def build_adamw(groups, config, device_type):
    return torch.optim.AdamW(groups, lr=config.learning_rate,
                             betas=(config.beta1, config.beta2),
                             **_fused(torch.optim.AdamW, device_type))


def build_adam(groups, config, device_type):
    return torch.optim.Adam(groups, lr=config.learning_rate,
                            betas=(config.beta1, config.beta2),
                            **_fused(torch.optim.Adam, device_type))
