# Task primitives vs. sequence-mixing design choices

Maps the computational primitive each synthetic task probes to the architectural
design choice it bears on, grounded in the literature and in the task's own
mechanics rather than in which config currently happens to wire them together.

This is the trimmed, most-important-tasks version: `Experiments` lists only the
tasks worth prioritizing per primitive (redundant `positional_lab` variants and
lower-priority tasks cut — see Scope note below), and `Important tasks we need
to add` lists only the highest-payoff addition per primitive, not the full
brainstormed list.

| Primitive | Experiments | Important tasks we need to add | Design change tested | Linear | Local (SWA) | Global | References |
|---|---|---|---|---|---|---|---|
| **Content-based retrieval** | `kv_retrieval`, `nested_kv_retrieval` (key2value mode), `content_addressed_retrieval` | Multi-query per sequence | **attention pattern** (full context needed to reach any key), **state size** (bounded state forces overwriting old key–value pairs) | degrades with #KV / state size | fails beyond window | strong | Zoology (arXiv:2312.04927); Mamba (arXiv:2312.00752); Olmo Hybrid §3.5 "Recall," Table 16 |
| **Positional addressing / copy** | `indexing`, `absolute_position_parity`, `relative_offset_copy` | Selective Copy (randomized spacing) | **positional encoding** (offset must resolve at unseen lengths), **state size** (bounded state can't hold an arbitrary-length copy buffer) | weak | window-bounded | strong | Jelassi et al. 2024; Arora et al. 2024b; S4/Mamba selective-copy |
| **Aggregation / counting** | `selective_count`, `sorting` | - | **attention range** (full needed for all-pairs), **depth** (log n for parallel aggregation) | strong (running sum) | local only | strong | Chomsky Hierarchy (arXiv:2207.02098); CLRS (arXiv:2205.15659); AKS sorting network (log-depth) |
| **Sequential state tracking** | `permutation` (s5/c5), `dyck`, `addition` (carry chain — see log-depth caveat) | - |  **depth** (fixed-depth attention can't compose unboundedly many updates) | untested | depth-bounded | depth-bounded | Olmo Hybrid §3.2/3.5, Table 15; Grazzi et al. 2025; Merrill et al. 2024; Barrington 1986 |
| **Composition** | `function_composition`, `nested_kv_retrieval` (value2keys mode, genuine 2-hop) |  `State-Based Recall` from Olmo — **top priority** | needs **a state-tracking-capable recurrent layer** *and* **an attention layer** together (neither alone composes state-tracking with retrieval) | fails (state tracking alone has no retrieval mechanism) | fails (same gap, plus window limits retrieval reach) | fails (retrieval alone hits the TC⁰ ceiling on composed state) | Olmo Hybrid §3.3–3.5, Theorem 1, Table 17 |
| **Planning / graph reachability** *(candidate new row)* | `maze` | - | **attention range** (spatially-adjacent cells can be sequence-distant), **depth** (one plan step per layer/loop) | untested | untested | works (full attention reaches spatially-adjacent-but-sequence-distant cells) | Ivanitskiy et al. arXiv:2312.02566; Reingold 2008 (undirected reachability in L) |
