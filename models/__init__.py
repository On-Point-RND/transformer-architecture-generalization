"""
Model registry for the sandbox.

Every architecture lives in its own file inside this folder (e.g. models/base.py).
You pick one at run time with the --model flag, e.g.:

    python train.py config/small.py --model=base

The value passed to --model is simply the file name (without the .py) inside the
models/ folder. So to experiment with your own architecture, copy models/base.py
to models/my_model.py, edit it, and run with --model=my_model.

Contract — each model file must expose exactly these two names:

    ModelConfig  # a dataclass with the fields train.py fills in:
                 #   block_size, vocab_size, n_layer, n_head, n_embd, dropout, bias
    Model        # an nn.Module whose __init__ takes a ModelConfig and which
                 # implements forward(idx, targets=None), generate(...),
                 # configure_optimizers(...), get_num_params(), estimate_mfu(...)
                 # and crop_block_size(...)

As long as those names and methods exist, train.py works with the model unchanged.
"""

import importlib


def get_model(name):
    """Return the (ModelConfig, Model) pair from models/<name>.py."""
    try:
        module = importlib.import_module(f"models.{name}")
    except ModuleNotFoundError as e:
        # only swallow the error about the model file itself, not an unrelated
        # import that happens to fail inside the model module
        if e.name == f"models.{name}":
            raise ValueError(
                f"Unknown model '{name}'. Create models/{name}.py or pick an "
                f"existing file from the models/ folder (e.g. --model=base)."
            ) from None
        raise
    for attr in ("ModelConfig", "Model"):
        if not hasattr(module, attr):
            raise ValueError(
                f"models/{name}.py must define '{attr}' (see models/__init__.py "
                f"for the model file contract)."
            )
    return module.ModelConfig, module.Model
