"""
Nested-dictionary retrieval — the KV-retrieval task with dicts inside dicts.

A random subset of the top-level pairs has its *value* replaced by a whole
nested dictionary, so the prompt looks like

    { k1 v1  k2 { k21 v21  k22 v22 }  k3 v3 }  ? v21   ->   k2 k21

i.e. the query is a **value** and the answer is the **path of keys** that
reaches it (one key for a top-level value, two for a value inside a nested
dict). The number of nested dicts is drawn at random, and each nested dict
gets a random length in [1, n_pairs] — ``n_pairs`` being the same config knob
the flat task uses.

Token layout — identical to generator/kv_retrieval.py
----------------------------------------------------
This class subclasses ``KVRetrievalGenerator`` on purpose: it inherits the
exact same special-token ids (PAD/QUERY_MARKER/SOD/EOD/BOE/EOE), the same
key/value id ranges and therefore the same ``vocab_size`` for a given
(k_card, v_card). Nesting reuses SOD/EOD as the brackets around the inner
dict rather than introducing new token ids. A checkpoint trained on the flat
task can consequently be fed these prompts unchanged — no embedding-table
mismatch, no re-tokenisation.

Using it as an OOD test for flat-dict models
--------------------------------------------
The constructor takes the flat generator's parameters (`k_card`, `v_card`,
`n_pairs`, `duplicate_keys`, `seed`) with the same names, so a checkpoint's
recorded `gen_params` can be reused and overridden for OOD evaluation.

Two knobs make that comparison fair:

* ``query_mode='key2value'`` asks the flat question ("here is a key, give me
  its value") over a nested prompt, where the queried key is always one whose
  value is a plain value. The nested dicts are then pure structural
  distractors — exactly the OOD axis a flat-trained model can be scored on.
  ``'value2keys'`` (the default) is the path-retrieval task described above.
* ``n_nested=0`` degenerates the prompt to a plain flat dict, which in
  ``key2value`` mode is token-for-token the flat task — the in-distribution
  control row for the sweep.

Sizing: nested prompts are longer than flat ones (a nested dict costs ``2 + 4*m``
tokens in place of a single value token) and they burn distinct values, since the
whole prompt is drawn without replacement. Both budgets are checkable up front —
``worst_case(n_pairs, n_nested, max_nested_len)`` returns what the hardest prompt
costs, and ``largest_nested_len(n_pairs, n_nested, v_card, block_size)`` returns
the biggest ``max_nested_len`` that always fits. Pass that value and the generator
cannot raise; leave ``max_nested_len`` at None with a tight ``v_card`` and it will
raise on *some* draws but not others, which is the worst way to find out.
"""

from typing import List, Tuple

import numpy as np

from .base import DatasetItem
from .kv_retrieval import KVRetrievalGenerator


class NestedKVRetrievalGenerator(KVRetrievalGenerator):
    def __init__(
        self,
        k_card: int,
        v_card: int,
        n_pairs: int | Tuple[int, int],
        n_nested: int | Tuple[int, int] | None = None,
        max_nested_len: int | Tuple[int, int] | None = None,
        query_mode: str = "value2keys",
        duplicate_keys: bool = False,
        seed: int | None = 42,
    ):
        """
        Generator for nested KV retrieval.

        :param k_card: Amount of possible keys.
        :type k_card: int
        :param v_card: Amount of possible values.
        :type v_card: int
        :param n_pairs: Amount of top-level pairs in each prompt, and the upper
        bound on the length of every nested dict. `n_pairs' if the parameter is an
        integer, and a value in [n_pairs[0], n_pairs[1]) if it is a tuple.
        :type n_pairs: int | Tuple[int, int]
        :param n_nested: How many top-level pairs hold a nested dict instead of a
        plain value. An integer for a fixed count, a tuple for a value in
        [n_nested[0], n_nested[1]), or None (default) for a random count in
        [1, n_pairs]. 0 gives a plain flat dict.
        :type n_nested: int | Tuple[int, int] | None
        :param max_nested_len: Upper bound on the length of a nested dict: an integer,
        a tuple drawn as [lo, hi), or None (default) for `n_pairs'. Always clamped to
        `n_pairs', so the documented "nested dicts are no longer than n_pairs" holds
        either way. Lower it when v_card or block_size cannot take the worst case —
        `worst_case()' below says what a given setting costs.
        :type max_nested_len: int | Tuple[int, int] | None
        :param query_mode: `'value2keys'' (default) queries a value and answers
        with the key path leading to it; `'key2value'' queries a top-level key
        whose value is a plain value and answers with that value, i.e. the flat
        task's question asked over a nested prompt.
        :type query_mode: str
        :param duplicate_keys: Whether to allow duplicate keys within one dict.
        Keys are drawn independently per dict, so the same key may occur at
        different nesting levels regardless of this flag — except the queried key
        in `key2value' mode, which is always unique across the whole prompt so the
        answer stays well defined.
        :type duplicate_keys: bool
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None
        """
        super().__init__(
            k_card=k_card,
            v_card=v_card,
            n_pairs=n_pairs,
            duplicate_keys=duplicate_keys,
            seed=seed,
        )
        if query_mode not in ("value2keys", "key2value"):
            raise ValueError(
                f"query_mode must be 'value2keys' or 'key2value', got {query_mode!r}"
            )
        self.n_nested = n_nested
        self.max_nested_len = max_nested_len
        self.query_mode = query_mode

    def _draw(self, spec) -> int:
        """Resolve an int-or-[lo, hi) spec (the flat generator's n_pairs convention)."""
        if isinstance(spec, int):
            return spec
        return int(self.rng.integers(spec[0], spec[1]))

    @staticmethod
    def worst_case(n_pairs, n_nested, max_nested_len=None, query_mode="value2keys"):
        """What the *hardest* prompt with these settings costs, as
        ``(prompt_tokens, distinct_values_needed, answer_tokens)``.

        Lets a caller check feasibility up front instead of discovering it as a
        ValueError halfway through a sweep. A setting is usable iff

            prompt_tokens + answer_tokens <= block_size + 1
            distinct_values_needed        <= v_card

        A plain pair costs 4 tokens; a nested dict of m pairs costs 5 + 4m in place
        of one value token. Values are drawn without replacement across the whole
        prompt (that is what makes the key path unique), which is where the value
        budget comes from.
        """
        n_nested = max(0, min(n_nested, n_pairs))
        m = n_pairs if max_nested_len is None else max(1, min(max_nested_len, n_pairs))
        tokens = 4 + 4 * (n_pairs - n_nested) + n_nested * (5 + 4 * m)
        values = (n_pairs - n_nested) + n_nested * m
        answer = 1 if query_mode == "key2value" else 2
        return tokens, values, answer

    @classmethod
    def largest_nested_len(cls, n_pairs, n_nested, v_card, block_size,
                           query_mode="value2keys"):
        """Longest nested dict that still fits both budgets, or 0 if this
        (n_pairs, n_nested) is impossible at all — pass the result as
        ``max_nested_len`` and the generator cannot raise."""
        for m in range(n_pairs, 0, -1):
            tok, val, ans = cls.worst_case(n_pairs, n_nested, m, query_mode)
            if tok + ans <= block_size + 1 and val <= v_card:
                return m
        return 0

    def _sample_one(self) -> DatasetItem:
        n_pairs = self._draw(self.n_pairs)

        if self.n_nested is None:
            n_nested = int(self.rng.integers(1, n_pairs + 1))
        else:
            n_nested = self._draw(self.n_nested)
        n_nested = max(0, min(n_nested, n_pairs))
        if self.query_mode == "key2value":
            # the query needs at least one top-level pair holding a plain value
            n_nested = min(n_nested, n_pairs - 1)

        # which top-level slots get a nested dict, and how long each one is
        nested_slots = set(
            self.rng.choice(n_pairs, size=n_nested, replace=False).tolist()
        )
        # длина вложенного словаря: случайная в [1, cap], cap <= n_pairs всегда
        cap = n_pairs if self.max_nested_len is None else self._draw(self.max_nested_len)
        cap = max(1, min(cap, n_pairs))
        nested_lens = {i: int(self.rng.integers(1, cap + 1)) for i in nested_slots}

        # Values are drawn globally without replacement so that the queried value
        # occurs exactly once in the prompt and its key path is well defined. (The
        # flat generator can afford to draw with replacement: it queries keys.)
        n_values = (n_pairs - n_nested) + sum(nested_lens.values())
        if n_values > self.v_card:
            raise ValueError(
                f"this prompt needs {n_values} distinct values but v_card={self.v_card}; "
                f"values must be unique for the queried value to have one key path. "
                f"Cap the nested dicts instead of hoping: "
                f"max_nested_len={self.largest_nested_len(n_pairs, n_nested, self.v_card, 10**9, self.query_mode)}"
                f" is the largest that always fits this v_card (0 = lower n_pairs/n_nested"
                f" or raise v_card). See worst_case()."
            )
        values = self.rng.choice(self.v_token_ids, size=n_values, replace=False)
        v_next = 0

        # In key2value mode the answer is "the value stored under this key", so the
        # queried key is picked up front and held out of every other draw: it then
        # occurs exactly once in the whole prompt (no shadowing by a later duplicate
        # and no same-named key hiding inside a nested dict), leaving one right
        # answer. In value2keys mode the queried *value* is what has to be unique,
        # which the global value draw above already guarantees, so keys stay free to
        # repeat across nesting levels.
        key_pool = self.k_token_ids
        q_slot, q_key = None, None
        if self.query_mode == "key2value":
            flat_slots = [i for i in range(n_pairs) if i not in nested_slots]
            q_slot = flat_slots[int(self.rng.integers(len(flat_slots)))]
            q_key = int(self.rng.choice(self.k_token_ids))
            key_pool = self.k_token_ids[self.k_token_ids != q_key]

        max_dict_len = max([n_pairs] + list(nested_lens.values()))
        if not self.duplicate_keys and max_dict_len > len(key_pool):
            raise ValueError(
                f"a dict in this prompt holds {max_dict_len} pairs but only "
                f"{len(key_pool)} distinct keys are available (k_card={self.k_card}"
                f"{', minus the reserved query key' if q_key is not None else ''}) — "
                f"raise k_card, lower n_pairs, or set duplicate_keys=True"
            )

        tokens: List[int] = [self.SOD_ID]
        paths = {}  # value token -> key path (1 or 2 keys) reaching it
        answer = None

        top_keys = self.rng.choice(
            key_pool, size=n_pairs, replace=self.duplicate_keys
        )
        if q_slot is not None:
            top_keys[q_slot] = q_key
        for i in range(n_pairs):
            k = int(top_keys[i])
            tokens += [self.BOE_ID, k]
            if i in nested_slots:
                m = nested_lens[i]
                inner_keys = self.rng.choice(
                    key_pool, size=m, replace=self.duplicate_keys
                )
                tokens.append(self.SOD_ID)
                for j in range(m):
                    ik, iv = int(inner_keys[j]), int(values[v_next])
                    v_next += 1
                    tokens += [self.BOE_ID, ik, iv, self.EOE_ID]
                    paths[iv] = [k, ik]
                tokens.append(self.EOD_ID)
            else:
                v = int(values[v_next])
                v_next += 1
                tokens.append(v)
                paths[v] = [k]
                if i == q_slot:
                    answer = [v]
            tokens.append(self.EOE_ID)
        tokens.append(self.EOD_ID)

        tokens.append(self.QUERY_MARKER_ID)
        if self.query_mode == "key2value":
            tokens.append(q_key)
        else:
            queried = list(paths)[int(self.rng.integers(len(paths)))]
            tokens.append(queried)
            answer = paths[queried]

        return DatasetItem(
            prompt=np.array(tokens, dtype=np.int64),
            answer=np.array(answer, dtype=np.int64),
        )
