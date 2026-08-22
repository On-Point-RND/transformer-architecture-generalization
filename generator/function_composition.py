from typing import Tuple

import numpy as np

from .base import DatasetItem, Generator


class FunctionCompositionGenerator(Generator):
    PAD_ID = 0
    QUERY_MARKER_ID = 1
    N_SPECIAL = 2

    def __init__(
        self,
        d_card: int,
        n_funcs: int,
        n_steps: int | Tuple[int, int],
        bijective: bool = True,
        seed: int | None = 42,
    ):
        """
        Generator for multi-step function composition.

        Prompt layout:  FN_0 f_0(0) .. f_0(d-1)
                        ...
                        FN_{m-1} f_{m-1}(0) .. f_{m-1}(d-1)
                        QUERY  x  g_1 .. g_k
        Answer layout:  g_k(...g_1(x)...)

        Each function is written as its output row in canonical domain order, so
        the table is unambiguous without emitting the inputs. Every function is
        resampled per example, which forces the composition to be computed from
        the context rather than memorised in the weights.

        :param d_card: Size of the domain (and codomain) of every function.
        :type d_card: int
        :param n_funcs: Number of functions defined in the prompt.
        :type n_funcs: int
        :param n_steps: Length of the composition chain. `n_steps' if the parameter
            is an integer, and a value in [n_steps[0], n_steps[1]) if it is a
            tuple. This is the depth-OOD knob: train at (1, 4), evaluate at 6, 8.
        :type n_steps: int | Tuple[int, int]
        :param bijective: If True every function is a permutation of the domain,
            so information is never destroyed and each step is invertible. If
            False the functions are arbitrary maps, which collapse the domain and
            make later steps recoverable from fewer distinct states.
        :type bijective: bool
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None
        """
        self.d_card = d_card
        self.n_funcs = n_funcs
        self.n_steps = n_steps
        self.bijective = bijective

        self.d_token_ids = np.arange(d_card) + self.N_SPECIAL
        self.f_token_ids = np.arange(n_funcs) + self.N_SPECIAL + d_card

        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + self.d_card + self.n_funcs

    @property
    def max_steps(self) -> int:
        if isinstance(self.n_steps, int):
            return self.n_steps
        return self.n_steps[1] - 1

    @property
    def min_block_size(self) -> int:
        """len(prompt) + len(answer)
        = [m * (d + 1) + 2 + k] + 1 <= block_size + 1
        """
        return self.n_funcs * (self.d_card + 1) + self.max_steps + 2

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.n_steps, int):
            k = self.n_steps
        else:
            k = int(self.rng.integers(self.n_steps[0], self.n_steps[1]))

        if self.bijective:
            tables = [self.rng.permutation(self.d_card) for _ in range(self.n_funcs)]
        else:
            tables = [
                self.rng.integers(0, self.d_card, size=self.d_card)
                for _ in range(self.n_funcs)
            ]

        parts = []
        for j in range(self.n_funcs):
            parts.append(np.array([self.f_token_ids[j]]))
            parts.append(self.d_token_ids[tables[j]])

        x = int(self.rng.integers(self.d_card))
        chain = self.rng.integers(0, self.n_funcs, size=k)

        parts.append(np.array([self.QUERY_MARKER_ID]))
        parts.append(np.array([self.d_token_ids[x]]))
        parts.append(self.f_token_ids[chain])

        prompt = np.concatenate(parts).astype(np.int64)

        y = x
        for j in chain:
            y = int(tables[j][y])
        answer = np.array([self.d_token_ids[y]], dtype=np.int64)

        return DatasetItem(prompt=prompt, answer=answer)
