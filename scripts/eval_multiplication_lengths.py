
import csv
from contextlib import nullcontext
from pathlib import Path

import torch

from core import checkpoint
from core.train import build_model, DTYPES
from tasks import get_task


RUN_DIR = Path("runs/multiplication_baseline")
CHECKPOINT_NAME = "best.pt"

LENGTHS = [5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 35]

MULTIPLIER_DIGITS = 3
N_TEST = 1000
BATCH_SIZE = 64
SEED = 2026


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"

print("Device:", device)


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

print("Loading:", RUN_DIR / CHECKPOINT_NAME)

ckpt = checkpoint.load(
    RUN_DIR,
    CHECKPOINT_NAME,
    device,
)

sections = checkpoint.config_sections(ckpt)

print("Checkpoint iteration:", ckpt["iter_num"])
print("Best validation loss:", ckpt["best_val_loss"])


# ---------------------------------------------------------
# Convert nested dictionaries to attribute-style config
# ---------------------------------------------------------

class ConfigObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigObject(value)
            setattr(self, key, value)


config = ConfigObject(sections)


# ---------------------------------------------------------
# Reconstruct original task
# ---------------------------------------------------------

original_task_params = dict(sections["task"]["params"])

original_task = get_task(
    sections["task"]["name"],
    {
        **original_task_params,
        "seed": SEED,
    },
)


# ---------------------------------------------------------
# Reconstruct trained model
# ---------------------------------------------------------

model = build_model(
    config,
    original_task,
    device,
    ckpt,
    RUN_DIR,
)

model.eval()

print("Model loaded successfully.")
print("Block size:", config.model.block_size)


# ---------------------------------------------------------
# Autocast
# ---------------------------------------------------------

dtype_name = getattr(config.hardware, "dtype", "float32")

if device_type == "cuda":
    ctx = torch.amp.autocast(
        device_type="cuda",
        dtype=DTYPES[dtype_name],
    )
else:
    ctx = nullcontext()


# ---------------------------------------------------------
# Evaluate one length
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_length(n_digits):

    task = get_task(
        "multiplication",
        {
            "n_digits": n_digits,
            "multiplier_digits": MULTIPLIER_DIGITS,
            "reverse": False,
            "seed": SEED + n_digits,
        },
    )

    items = task.generate_val(N_TEST)

    total_correct_digits = 0
    total_answer_digits = 0

    total_exact = 0
    total_examples = 0

    total_loss = 0.0
    total_loss_examples = 0

    for start in range(0, len(items), BATCH_SIZE):

        batch = items[start:start + BATCH_SIZE]

        x_np, y_np = task.collate(
            batch,
            config.model.block_size,
        )

        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        with ctx:
            logits, loss = model(x, y)

        predicted = logits.argmax(dim=-1).cpu().numpy()
        targets = y.cpu().numpy()

        answer_mask = targets != -1

        total_correct_digits += (
            (predicted == targets) & answer_mask
        ).sum()

        total_answer_digits += answer_mask.sum()

        exact = (
            (predicted == targets) | ~answer_mask
        ).all(axis=1)

        total_exact += exact.sum()
        total_examples += len(batch)

        total_loss += loss.item() * len(batch)
        total_loss_examples += len(batch)

    return {
        "n_digits": n_digits,
        "multiplier_digits": MULTIPLIER_DIGITS,
        "split": "ID" if n_digits == 5 else "OOD",
        "n_examples": total_examples,
        "loss": float(total_loss / total_loss_examples),
        "digit_acc": float(total_correct_digits / total_answer_digits),
        "exact_acc": float(total_exact / total_examples),
    }


# ---------------------------------------------------------
# Evaluate all lengths
# ---------------------------------------------------------

results = []

print("\nLength generalization evaluation")
print("--------------------------------")

for d in LENGTHS:

    result = evaluate_length(d)
    results.append(result)

    print(
        f"{d}×{MULTIPLIER_DIGITS} "
        f"[{result['split']}]  "
        f"loss={result['loss']:.4f}  "
        f"digit_acc={result['digit_acc'] * 100:.2f}%  "
        f"exact_acc={result['exact_acc'] * 100:.2f}%"
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_path = RUN_DIR / "length_generalization.csv"

with open(output_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "n_digits",
            "multiplier_digits",
            "split",
            "n_examples",
            "loss",
            "digit_acc",
            "exact_acc",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


print("\nSaved:")
print(output_path)
