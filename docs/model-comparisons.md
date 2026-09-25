# Comparing models and tasks

Create one experiment directory under `configs/`. Put settings shared by every
run in `base.yaml`: task, data seed, training budget, optimizer, hardware, and
common model dimensions. Add one small overlay per model that includes the base,
sets architecture-specific fields, and uses a unique `paths.run_dir`.

Train each overlay separately. For example:

```bash
.venv/bin/python main.py --config configs/kv-comparison/mamba2.yaml
.venv/bin/python main.py --config configs/kv-comparison/transformer.yaml
```

Evaluate the resulting runs together:

```bash
.venv/bin/python evaluate.py \
  runs/kv-comparison/mamba2 runs/kv-comparison/transformer \
  --checkpoint best.pt --device mps -n 5000 \
  -o runs/kv-comparison/evaluations.csv
```

Use `--by FIELD` to break results down by task metadata such as length or
difficulty. For a new task, copy an experiment directory and change
`task.name`, `task.params`, `model.block_size`, and the run directories. For a
new model, copy an overlay and change `model.name` plus its architecture fields.

Keep the seed, task distribution, batch size, training budget, optimizer,
width, and depth shared where possible. Also compare logged parameter counts:
equal width and depth do not guarantee equal model size. Use `--rerun` only to
overwrite an existing run, and `--set hardware.device=cpu` when MPS is absent.
