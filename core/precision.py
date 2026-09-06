"""Native CUDA mixed precision, including FP16 on Turing GPUs."""
from contextlib import nullcontext

import torch

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_dtype(device, dtype="auto"):
    if dtype not in (*DTYPES, "auto"):
        raise ValueError(f"Unknown precision: {dtype}")
    if not str(device).startswith("cuda"):
        return "float32"
    with torch.cuda.device(device):
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    if dtype == "auto":
        return "bfloat16" if native_bf16 else "float16"
    if dtype == "bfloat16" and not native_bf16:
        raise ValueError("This GPU has no native BF16; use dtype=auto or float16")
    return dtype


def autocast_context(device, dtype="auto"):
    dtype = resolve_dtype(device, dtype)
    return (nullcontext() if dtype == "float32" else
            torch.amp.autocast("cuda", dtype=DTYPES[dtype]))
