#!/usr/bin/env python3
"""Very small end-to-end smoke test for the SMILES experiment pipeline.

Runs fast on CPU (tiny models, tiny data, a handful of steps) and checks that
every moving part still launches and produces output after the src/ refactor:

  1. every module under src/ imports cleanly;
  2. every sweep config in configs/ expands via run.build_run_specs, and every
     probe config parses;
  3. the Task contract holds for every registered task (vocab / seq_len /
     answer_span / make_splits / difficulty / sample_balanced);
  4. EVERY real sweep config runs END-TO-END (one spec, tiny override) — this is
     the actual "will my experiment launch" guarantee, covering every sweep KIND
     (grid / fixed_width / fixed_budget / width_sweep / cells / head_ablation /
     loop / length_gen) with its real fields;
  5. both probes (probe_linear, probe_layers) run end-to-end for every task;
  6. the real probe configs (add_probe, perm_cross_probe) launch via --config.

It writes only into a throwaway temp dir (cwd is switched there) so the repo's
results/ and checkpoints/ are never touched.

Usage:
    python tests/smoke_test.py            # (from anywhere)
Exit code 0 = all green, 1 = something broke.
"""

from __future__ import annotations

import copy
import glob
import importlib
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
CONFIGS = os.path.join(REPO, "configs")
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import yaml         # noqa: E402

# a tiny training recipe + tiny eval, forced onto every config so runs are fast
TINY = dict(max_steps=30, n_train=800, n_test=120, batch_size=128, eval_every=30,
            early_stop_acc=1.01, min_steps=0, eval_subset=120, train_eval_n=120)
TINY_EVAL = {"chain_per_bin": 5, "inv_per_bin": 5}

# task -> (variant, length) used for the probe checks
PROBE_TASKS = [("addition", "SEQ", 4), ("permutation", "S5", 6), ("sorting", None, 8)]


# ---------------------------------------------------------------------------
class Results:
    def __init__(self):
        self.failures = []

    def check(self, name, fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001 - smoke test wants everything
            self.failures.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()


def section(title):
    print(f"\n=== {title} ===")


def _is_sweep_cfg(cfg) -> bool:
    return isinstance(cfg, dict) and "sweep" in cfg and "train" in cfg and "ckpts" not in cfg


# --- 1. imports -------------------------------------------------------------
def check_imports():
    mods = [os.path.basename(p)[:-3] for p in glob.glob(os.path.join(SRC, "*.py"))]
    mods += ["datasets." + os.path.basename(p)[:-3]
             for p in glob.glob(os.path.join(SRC, "datasets", "*.py"))
             if not p.endswith("__init__.py")]
    mods += ["datasets"]
    for m in sorted(set(mods)):
        if m == "__init__":
            continue
        importlib.import_module(m)


# --- 2. config expansion / parsing -----------------------------------------
def check_configs():
    import run
    for path in sorted(glob.glob(os.path.join(CONFIGS, "*.yaml"))):
        cfg = yaml.safe_load(open(path))
        name = os.path.basename(path)
        if not isinstance(cfg, dict):
            continue
        if "ckpts" in cfg:                      # probe config
            assert cfg.get("out"), f"{name}: probe config missing 'out'"
            continue
        if _is_sweep_cfg(cfg):
            specs = run.build_run_specs(cfg)
            assert specs, f"{name}: expanded to 0 specs"
            for s in specs:                      # tag builder must not throw
                run.spec_tag(s)


# --- 3. Task contract -------------------------------------------------------
def check_task_contract():
    from datasets import TASKS, get_task
    L = 5
    seen = set()
    for reg_name in TASKS:
        task = get_task(reg_name)
        if task.name in seen:
            continue
        seen.add(task.name)
        assert task.vocab_size > 0, f"{task.name}: vocab_size<=0"
        a_start, a_len = task.answer_span(L)
        assert a_start > 0 and a_len > 0, f"{task.name}: bad answer_span"
        assert task.seq_len(L) >= a_start + a_len, f"{task.name}: seq_len too short"
        for variant in (task.variants or [None]):
            tr, te = task.make_splits(L, n_train=30, n_test=15, seed=0, variant=variant)
            assert len(tr) == 30 and len(te) == 15, f"{task.name}/{variant}: split sizes"
            x0, y0 = tr[0]
            assert len(x0) == task.seq_len(L), f"{task.name}/{variant}: seq len mismatch"
            assert len(y0) == len(x0), f"{task.name}/{variant}: target len mismatch"
            d = task.difficulty(x0, L)
            assert isinstance(d, (int, np.integer)) and d >= 0, f"{task.name}/{variant}: difficulty"
            bal = task.sample_balanced(L, per_bin=3, seed=0, variant=variant)
            assert len(bal) > 0, f"{task.name}/{variant}: empty balanced sample"


# --- 4. every real sweep config end-to-end (tiny override, 1 spec) ---------
def _run_real_config(path):
    import run
    cfg = copy.deepcopy(yaml.safe_load(open(path)))
    cfg["train"] = dict(TINY)
    cfg["eval"] = dict(TINY_EVAL)
    tmp_cfg = os.path.join(os.getcwd(), f"real_{cfg['sweep']}.yaml")
    yaml.safe_dump(cfg, open(tmp_cfg, "w"))
    run.run_sweep(tmp_cfg, limit=1, verbose=False)   # trains exactly one spec
    summ = os.path.join("results", f"{cfg['sweep']}_summary.csv")
    assert os.path.exists(summ), f"{os.path.basename(path)}: no summary CSV"


# --- 5. probes end-to-end ---------------------------------------------------
def _tiny_ckpt(task, variant, length, tag):
    from train import TrainConfig, train_run
    train_run(TrainConfig(task=task, variant=variant, length=length, n_layers=2,
                          d_model=32, max_steps=30, n_train=800, n_test=120,
                          batch_size=128, eval_every=30, early_stop_acc=1.01,
                          min_steps=0, seed=0, tag=tag, device="cpu"), verbose=False)
    return os.path.join("checkpoints", f"{tag}.pt")


def _run_script(module_name, argv):
    mod = importlib.import_module(module_name)
    old = sys.argv
    sys.argv = [module_name] + argv
    try:
        mod.main()
    finally:
        sys.argv = old


def check_probes():
    for task, variant, length in PROBE_TASKS:
        tag = f"probe_{task}"
        ck = _tiny_ckpt(task, variant, length, tag)
        _run_script("probe_linear", ["--ckpts", ck, "--out", f"lin_{task}",
                                     "--per-bin", "5", "--n-train", "500", "--steps", "20"])
        assert os.path.exists(f"results/lin_{task}_probe.csv"), f"probe_linear {task}: no CSV"
        _run_script("probe_layers", ["--ckpts", ck, "--out", f"lay_{task}", "--per-bin", "5"])
        assert os.path.exists(f"results/lay_{task}_probe.csv"), f"probe_layers {task}: no CSV"


# --- 6. real probe configs via --config ------------------------------------
def check_probe_configs():
    _tiny_ckpt("addition", "SEQ", 4, "add_grid_L2_d32_s0")             # add_grid_probe glob
    _tiny_ckpt("permutation", "S5", 6, "perm_grid_grid_S5_N6_d32_s0")  # perm cross glob
    for cfg_name in ("add_grid_probe.yaml", "perm_grid_cross_probe.yaml"):
        _run_script("probe_layers", ["--config", os.path.join(CONFIGS, cfg_name)])
    assert os.path.exists("results/add_grid_probe.csv")
    assert os.path.exists("results/perm_cross_S5_on_C5_probe.csv")


def main():
    r = Results()
    section("1. module imports")
    r.check("all src/ modules import", check_imports)

    section("2. config expansion / parsing")
    r.check("all configs expand/parse", check_configs)

    section("3. Task contract (all tasks)")
    r.check("vocab/seq_len/answer_span/splits/difficulty/balanced", check_task_contract)

    tmp = tempfile.mkdtemp(prefix="smiles_smoke_")
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        section("4. every real sweep config end-to-end (1 spec, tiny)")
        sweep_cfgs = [p for p in sorted(glob.glob(os.path.join(CONFIGS, "*.yaml")))
                      if _is_sweep_cfg(yaml.safe_load(open(p)))]
        for path in sweep_cfgs:
            r.check(os.path.basename(path), lambda p=path: _run_real_config(p))

        section("5. probes end-to-end (all tasks)")
        r.check("probe_linear + probe_layers", check_probes)

        section("6. probe configs via --config")
        r.check("add_probe + perm_cross_probe", check_probe_configs)
    finally:
        os.chdir(old_cwd)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    section("SUMMARY")
    if r.failures:
        print(f"  {len(r.failures)} FAILED: {', '.join(r.failures)}")
        return 1
    print("  ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
