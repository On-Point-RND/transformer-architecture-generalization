"""Optional MLflow sink for training metrics and local run artifacts."""

import json
import math
import os
import uuid
import warnings
from dataclasses import is_dataclass
from pathlib import Path

from core.config import to_dict


REQUIRED_ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_WORKSPACE",
)


def _flatten(values, prefix=""):
    flat = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if is_dataclass(value):
            value = to_dict(value)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value)
        elif value is not None:
            flat[name] = value
    return flat


def _metric_values(event, fields):
    aliases = {
        "lr": "learning_rate",
        "dt": "step_time_seconds",
        "loss": "train_batch_loss",
    }
    metrics = {}

    def add(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        value = float(value)
        if math.isfinite(value):
            metrics[name] = value

    for key, value in fields.items():
        if key == "iter":
            continue
        if event == "train" and key not in {"loss", "lr", "dt", "mfu", "grad_norm"}:
            continue
        if event in {"eval", "summary"} and not (
            key.startswith("train_")
            or key.startswith("val_")
            or key in {"lr", "mfu", "best_val_loss", "tokens"}
        ):
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in _flatten(value, key).items():
                if event == "diag":
                    nested_key = f"diagnostics.{nested_key}"
                add(nested_key, nested_value)
            continue
        name = aliases.get(key, key)
        if event == "diag":
            name = f"diagnostics.{name}"
        add(name, value)
    return metrics


class MLflowSink:
    """A best-effort metric sink; configuration failures remain fail-fast."""

    def __init__(self, mlflow, run, run_id_path):
        self.mlflow = mlflow
        self.run = run
        self.run_id_path = run_id_path
        self.failed = False
        self.closed = False

    @classmethod
    def from_environment(cls, config, logs_dir, metadata):
        configured = {name: os.environ.get(name, "") for name in REQUIRED_ENV}
        if not any(configured.values()):
            return None
        missing = [name for name, value in configured.items() if not value]
        if missing:
            raise RuntimeError(
                "MLflow is partially configured; set all of: " + ", ".join(REQUIRED_ENV)
                + f" (missing: {', '.join(missing)})"
            )
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "MLflow environment variables are set but mlflow is not installed; "
                "install MLflow 3.10 or newer"
            ) from error

        if not hasattr(mlflow, "set_workspace"):
            raise RuntimeError(
                "this MLflow client does not provide set_workspace(); install "
                "MLflow 3.10 or newer"
            )

        mlflow.set_tracking_uri(configured["MLFLOW_TRACKING_URI"])
        mlflow.set_workspace(configured["MLFLOW_WORKSPACE"])
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "transformer-architecture-generalization"
        )
        mlflow.set_experiment(experiment_name)

        run_id_path = Path(logs_dir) / "mlflow-run-id"
        if run_id_path.is_file():
            run = mlflow.start_run(run_id=run_id_path.read_text(encoding="utf-8").strip())
        else:
            generated_name = (
                f"{metadata['model']}-{Path(config.paths.run_dir).name}-{uuid.uuid4().hex[:8]}"
            )
            run_name = os.environ.get("MLFLOW_RUN_NAME") or generated_name
            tags = {
                "model": metadata["model"],
                "task": metadata["task"],
                "run_dir": str(config.paths.run_dir),
            }
            run = mlflow.start_run(run_name=run_name, tags=tags)
            run_id_path.write_text(run.info.run_id + "\n", encoding="utf-8")
            mlflow.log_params(_flatten(to_dict(config)))
        return cls(mlflow, run, run_id_path)

    def log(self, event, fields):
        if self.failed or self.closed:
            return
        metrics = _metric_values(event, fields)
        if not metrics:
            return
        try:
            self.mlflow.log_metrics(metrics, step=int(fields.get("iter", 0)))
        except Exception as error:  # do not sacrifice a long training run to telemetry
            self.failed = True
            warnings.warn(f"MLflow metric logging disabled after an error: {error}")

    def close(self, artifact_paths=(), status="FINISHED"):
        if self.closed:
            return
        self.closed = True
        if not self.failed:
            for path in artifact_paths:
                path = Path(path)
                if path.is_file():
                    try:
                        self.mlflow.log_artifact(str(path), artifact_path="run-output")
                    except Exception as error:
                        warnings.warn(f"could not upload MLflow artifact {path}: {error}")
        try:
            self.mlflow.end_run(status=status)
        except Exception as error:
            warnings.warn(f"could not close MLflow run {self.run.info.run_id}: {error}")
