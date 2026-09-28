"""RULER Variable Tracking: recover an alias chain from its final value.

    x1 = 4 ;  y1 = 7 ;  x2 = x1 ;  y2 = y1 ;  x3 = x2 ;  ? 4  ->  x1 x2 x3

Each chain starts at a value and continues through aliases. Chains may be
interleaved but preserve their internal order; filler tokens form the haystack.
The answer is the queried chain in assignment order. This is the only RULER
task kept in this repository.
"""

import numpy as np

from .base import DatasetItem, Task, max_int, sample_int, validate_int_spec


class RulerTask(Task):
    PAD_ID = 0
    EQUALS_ID = 1
    SEPARATOR_ID = 2
    QUERY_ID = 3
    N_SPECIAL = 4
    N_FILLER_TOKENS = 16

    def __init__(
        self,
        n_vars: int = 64,
        v_card: int = 32,
        num_chains: int | tuple[int, int] = 1,
        num_hops: int | tuple[int, int] = (2, 9),
        n_filler: int | tuple[int, int] = (0, 33),
        seed: int | None = 42,
    ):
        super().__init__(seed)
        self.n_vars = validate_int_spec(n_vars, "n_vars", 1)
        self.v_card = validate_int_spec(v_card, "v_card", 1)
        self.num_chains = validate_int_spec(num_chains, "num_chains", 1)
        self.num_hops = validate_int_spec(num_hops, "num_hops", 0)
        self.n_filler = validate_int_spec(n_filler, "n_filler", 0)

        variables_needed = max_int(self.num_chains) * (max_int(self.num_hops) + 1)
        if variables_needed > self.n_vars:
            raise ValueError(
                f"the largest example needs {variables_needed} distinct variables, "
                f"but n_vars={self.n_vars}"
            )
        if max_int(self.num_chains) > self.v_card:
            raise ValueError("each chain needs a distinct value; raise v_card")

        self.filler_ids = np.arange(self.N_FILLER_TOKENS) + self.N_SPECIAL
        self.variable_ids = np.arange(self.n_vars) + self.N_SPECIAL + self.N_FILLER_TOKENS
        self.value_ids = (
            np.arange(self.v_card)
            + self.N_SPECIAL
            + self.N_FILLER_TOKENS
            + self.n_vars
        )

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + self.N_FILLER_TOKENS + self.n_vars + self.v_card

    def _sample_one(self) -> DatasetItem:
        n_chains = sample_int(self.rng, self.num_chains)
        n_hops = sample_int(self.rng, self.num_hops)
        n_filler = sample_int(self.rng, self.n_filler)
        names = self.rng.choice(
            self.variable_ids, size=(n_chains, n_hops + 1), replace=False
        )
        values = self.rng.choice(self.value_ids, size=n_chains, replace=False)

        speakers = np.repeat(np.arange(n_chains), n_hops + 1)
        self.rng.shuffle(speakers)
        slots = np.zeros(len(speakers) + n_filler, dtype=bool)
        slots[self.rng.choice(len(slots), size=len(speakers), replace=False)] = True
        filler = iter(self.rng.choice(self.filler_ids, size=n_filler).tolist())
        speakers = iter(speakers.tolist())

        prompt = []
        cursor = [0] * n_chains
        root_position = None
        for is_statement in slots:
            if not is_statement:
                prompt.append(next(filler))
                continue
            chain = next(speakers)
            step = cursor[chain]
            cursor[chain] += 1
            rhs = values[chain] if step == 0 else names[chain, step - 1]
            if chain == 0 and step == 0:
                root_position = len(prompt)
            prompt += [
                int(names[chain, step]),
                self.EQUALS_ID,
                int(rhs),
                self.SEPARATOR_ID,
            ]

        prompt += [self.QUERY_ID, int(values[0])]
        return DatasetItem(
            np.asarray(prompt, dtype=np.int64),
            names[0].astype(np.int64),
            metadata={
                "num_chains": n_chains,
                "num_hops": n_hops,
                "n_filler": n_filler,
                "sequence_length": len(prompt),
                "absolute_target_position": root_position,
                "relative_distance": len(prompt) - 1 - root_position,
            },
        )
