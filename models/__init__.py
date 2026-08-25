"""Architecture registry.

Every architecture lives in its own file in this folder and is selected in the
config with ``model.name``:

    model:
      name: positional

To try a new architecture, copy models/vanilla.py, override the hooks you need
(see core/model.py) and run with ``model.name: <file name>``.

Contract — each model file must expose exactly these two names:

    Config   # a dataclass extending core.config.ModelConfig with this
             # architecture's own fields (defaults included), so YAML
             # validation covers them
    Model    # an nn.Module subclassing core.model.Transformer

Nothing else in the codebase has to change to add one.
"""

import importlib


def get_model(name):
    try:
        module = importlib.import_module(f"models.{name}")
    except ModuleNotFoundError as e:
        # only swallow the error about the model file itself, not an unrelated
        # import that happens to fail inside it
        if e.name == f"models.{name}":
            raise ValueError(
                f"Unknown model '{name}'. Create models/{name}.py or pick an "
                f"existing file from the models/ folder (e.g. name: vanilla)."
            ) from None
        raise
    for attr in ("Config", "Model"):
        if not hasattr(module, attr):
            raise ValueError(
                f"models/{name}.py must define '{attr}' (see models/__init__.py "
                f"for the model file contract)."
            )
    return module.Config, module.Model
