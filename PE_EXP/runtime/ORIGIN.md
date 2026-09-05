Runtime snapshot from local commit 64282e84e73d6808bdcfb16e9a843c73caa181a0.
The copied Python modules are unchanged. core/__init__.py is added to isolate
imports from a checkout with a different project layout.

Only the modules needed for positional lab, KV and sorting are included.
The registry files retain the original names of other tasks/models, whose
implementations are intentionally not bundled because this runner never uses them.

The runner fingerprints this snapshot in each results manifest.
