import importlib

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
