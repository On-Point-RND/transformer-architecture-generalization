import importlib


def get_model(name):
    try:
        module = importlib.import_module(f"models.{name}")
    except ModuleNotFoundError as e:
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
