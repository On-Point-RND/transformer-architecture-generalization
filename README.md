
## Overview
This repository contains the experimental setup and results for systematically evaluating how specific Transformer architectural choices impact in-distribution (ID) learning and out-of-distribution (OOD) length generalization on synthetic algorithmic tasks. 

## Hypotheses and Results

### Hypothesis 9: Tied Key-Value Projections (K=V)
**Description:** 
This experiment investigates the functional role of attention projection parameterization by comparing standard separate key and value projections ($K \neq V$) against a parameter-tied variant where the key and value projections share a single weight matrix ($K=V$). The expectation is that forcing keys (addresses) and values (content) into a shared low-rank subspace will severely degrade performance on associative retrieval tasks (KV) due to structural contradiction, but will not negatively affect sequential computation tasks (FUNC) where intermediate representations serve dual roles.

**Exact-Match Accuracy Results:**

| Task | Split (Regime) | Baseline Model ($K \neq V$) | Tied Model ($K=V$) | Difference (pp) |
| :--- | :--- | :--- | :--- | :--- |
| KV | test (ID) | 100.00% | 18.20% | -81.80 |
| KV | test_ood | 7.40% | 0.70% | -6.70 |
| FUNC | test (ID) | 98.60% | 98.70% | +0.10 |
| FUNC | test_ood | 0.60% | 0.30% | -0.30 |


### Hypothesis 10: FFN Gating on Conditional Branching (SwiGLU vs. ReLU)
**Description:** 
This experiment evaluates whether replacing a standard ReLU Feed-Forward Network (FFN) with a SwiGLU FFN improves the model's ability to perform multi-step conditional branching. The comparison is conducted under a strict iso-parameter constraint (matching the total parameter count). The expectation is that the multiplicative gating mechanism in SwiGLU will mathematically better approximate conditional branching operations, leading to higher accuracy on deep function composition (FUNC-OOD).

**Exact-Match Accuracy Results:**

| Task | Split (Regime) | ReLU Baseline | SwiGLU FFN | Difference (pp) |
| :--- | :--- | :--- | :--- | :--- |
| FUNC | test (ID) | 98.60% | 98.90% | +0.30 |
| FUNC | test_ood | 0.60% | 0.90% | +0.30 |


### Hypothesis 11: FFN Design on Associative Retrieval (KV-OOD)
**Description:** 
This experiment tests the assumption that FFN design has no measurable impact on long-context associative retrieval. The initial expectation is that replacing ReLU with SwiGLU will yield a minimal difference (< 3 pp) on KV-OOD, assuming the primary bottleneck for retrieval on long contexts is attention matrix dilution rather than FFN computational capacity. To trace the internal transformations, layer-wise activation sparsity (fraction of elements $|x| < 0.01$) and feature variance are measured across decoder layers during KV-OOD evaluation.

**Exact-Match Accuracy Results:**

| Task | Split (Regime) | ReLU Baseline | SwiGLU FFN | Difference (pp) |
| :--- | :--- | :--- | :--- | :--- |
| KV | test (ID) | 100.00% | 100.00% | 0.00 |
| KV | test_ood | 7.40% | 31.10% | +23.70 |
