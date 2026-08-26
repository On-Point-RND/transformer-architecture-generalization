"""Optimizer registry.

The algorithm is picked in the config:

    optimizer:
      name: adamw
      decay: matrices

Contract — each builder here has the signature

    build(groups, config, device_type) -> torch.optim.Optimizer

where ``groups`` are the parameter groups the model already formed (each with
its own ``weight_decay``) and ``config`` is the optimizer section.

There is deliberately no base class of our own: ``torch.optim.Optimizer`` is
already it, and wrapping it would buy nothing. What varies between experiments
is which algorithm runs and how parameters are grouped — the first lives here,
the second in ``Transformer.configure_optimizers``.

Adding one = add a file here (or a function to an existing one) and an entry in
OPTIMIZERS below.
"""

import importlib

# name -> (module path, builder function)
OPTIMIZERS = {
    "adamw": ("optimizers.adam", "build_adamw"),
    "adam": ("optimizers.adam", "build_adam"),
}


def get_optimizer(name):
    if name not in OPTIMIZERS:
        known = ", ".join(sorted(OPTIMIZERS))
        raise ValueError(
            f"Unknown optimizer '{name}'. Known optimizers: {known}. "
            f"Register a new one in optimizers/__init__.py (see the contract there)."
        )
    module_path, builder = OPTIMIZERS[name]
    return getattr(importlib.import_module(module_path), builder)
