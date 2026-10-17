#!/usr/bin/env python3
"""Execute fused GDN/GDN2 training kernels and compare with PyTorch on CUDA.

Run after installing requirements-cuda.txt, before launching a long sweep:
    python scripts/check_cuda_kernels.py --gpu 0
This does not create experiment runs or write checkpoints.
"""

import argparse
import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import expand_configs
from models.gated_deltanet import GatedDeltaNet
from models.gated_deltanet2 import GatedDeltaNet2


def check(config_path, mixer_class, device, dtype):
    config, = expand_configs(PROJECT_ROOT / config_path)
    # Exercise each model's actual comparison dimensions, plus both gate modes.
    for negative in (False, True):
        config.model.allow_neg_eigval = negative
        config.model.scan_kernel = "fla"
        fused = mixer_class(config.model).to(device)
        reference = copy.deepcopy(fused)
        reference.scan_kernel = "torch"
        for length in (17, 65):
            fused.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            x = torch.randn(2, length, config.model.n_embd, device=device, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                actual, expected = fused(x), reference(xr)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.05)
            weight = torch.randn_like(actual)
            (actual.float() * weight.float()).mean().backward()
            (expected.float() * weight.float()).mean().backward()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(x.grad, xr.grad, atol=3e-4, rtol=0.15)
            for name, parameter in fused.named_parameters():
                other = dict(reference.named_parameters())[name]
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"missing or nonfinite fused gradient: {name}")
                torch.testing.assert_close(parameter.grad, other.grad, atol=0.003, rtol=0.15, msg=name)
            print(f"PASS {config.model.name}: fused forward/backward vs torch, "
                  f"negative_eigenvalues={negative}, length={length}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", choices=("gdn", "gdn2"), action="append", dest="models")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.exit(1, "CUDA unavailable: run this check on the experiment GPU host.\n")
    if not 0 <= args.gpu < torch.cuda.device_count():
        parser.error(f"--gpu must be in [0, {torch.cuda.device_count()})")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; {torch.cuda.get_device_name(device)}")
    torch.manual_seed(1337)
    for model in args.models or ("gdn", "gdn2"):
        path, mixer = {
            "gdn": ("configs/architecture_comparison/gdn_negative_eigenvalues.yaml", GatedDeltaNet),
            "gdn2": ("configs/architecture_comparison/gdn2.yaml", GatedDeltaNet2),
        }[model]
        check(path, mixer, device, getattr(torch, args.dtype))
    print("All selected fused CUDA training kernels passed.")


if __name__ == "__main__":
    main()
