"""Controlled synthetic tasks that expose one positional property at a time.

All variants use the normal Generator/DatasetItem contract.  Distribution
shifts are constructor parameters, so the same trained checkpoint can be
evaluated ID/OOD by overriding ``length_range``, ``target_position_range``,
``distance_range``, ``filler_range`` or ``correlation``.
"""

import numpy as np

from .base import DatasetItem, Generator


# Every laboratory task is also registered as a standalone dataset name in
# generator/__init__.py.  Keeping the mapping here makes this file the single
# source of truth for task discovery in training and orchestration scripts.
POSITIONAL_TASK_DEFAULTS = {
    "content_addressed_retrieval": {"task": "content_addressed_retrieval"},
    "absolute_position_parity": {"task": "absolute_position_parity"},
    "absolute_position_extrapolation": {"task": "absolute_position_extrapolation"},
    "relative_offset_copy": {"task": "relative_offset_copy"},
    "unseen_long_relative_offset": {"task": "unseen_long_relative_offset"},
    "local_latest_update": {"task": "local_latest_update"},
    "distant_retrieval": {"task": "distant_retrieval"},
    "bucketed_relative_distance": {"task": "bucketed_relative_distance"},
    "exact_distance_beyond_bucket": {"task": "exact_distance_beyond_bucket"},
    "selective_count": {"task": "selective_count"},
    "dense_absolute_position": {"task": "dense_absolute_position"},
    "context_adaptive_retrieval": {"task": "context_adaptive_retrieval"},
    "context_correlation_flip": {"task": "context_correlation_flip"},
    "periodic_distance": {"task": "periodic_distance"},
    "nonperiodic_absolute_threshold": {"task": "nonperiodic_absolute_threshold"},
}


class PositionalLabGenerator(Generator):
    PAD_ID = 0
    BOS, QUERY, MARK_A, MARK_B, UPDATE, RESET, RELEVANT = range(1, 8)
    FILL_LOW, FILL_HIGH = 16, 63
    KEY_LOW, KEY_HIGH = 64, 79
    VALUE_LOW, VALUE_HIGH = 96, 111
    CLASS_LOW = 128

    def __init__(self, task="content_addressed_retrieval", length_range=(16, 32),
                 target_position_range=None, distance_range=(4, 8),
                 filler_range=(4, 16), relevant_count_range=(1, 8), period=4,
                 threshold=16, correlation=1, seed=42):
        self.task = task
        self.length_range = tuple(length_range)
        self.target_position_range = (tuple(target_position_range)
                                      if target_position_range is not None else None)
        self.distance_range = tuple(distance_range)
        self.filler_range = tuple(filler_range)
        self.relevant_count_range = tuple(relevant_count_range)
        self.period = int(period)
        self.threshold = int(threshold)
        self.correlation = int(correlation)
        self.rng = np.random.default_rng(seed)
        if task not in self.TASKS:
            raise ValueError(f"unknown positional lab task {task!r}; known: {sorted(self.TASKS)}")

    @property
    def vocab_size(self):
        return 160

    def _int(self, bounds):
        lo, hi = bounds
        return int(self.rng.integers(lo, hi + 1))

    def _fill(self, n):
        return self.rng.integers(self.FILL_LOW, self.FILL_HIGH + 1, size=n, dtype=np.int64)

    def _item(self, prompt, answer, target_pos=None, distance=None, **extra):
        length = len(prompt)
        metadata = {
            "sequence_length": length,
            "absolute_target_position": target_pos,
            "normalized_target_position": (target_pos / max(length - 1, 1)
                                           if target_pos is not None else None),
            "relative_distance": distance,
            **extra,
        }
        return DatasetItem(np.asarray(prompt, dtype=np.int64),
                           np.atleast_1d(answer).astype(np.int64), metadata)

    def _absolute(self):
        length = self._int(self.length_range)
        bounds = self.target_position_range or (1, length - 2)
        pos = min(self._int(bounds), length - 2)
        prompt = self._fill(length)
        prompt[0], prompt[pos], prompt[-1] = self.BOS, self.MARK_A, self.QUERY
        return self._item(prompt, [self.CLASS_LOW + pos % 2], pos, None,
                          position_parity=pos % 2)

    def _relative_copy(self):
        length = self._int(self.length_range)
        distance = self._int(self.distance_range)
        if distance >= length - 1:
            raise ValueError("relative distance must be smaller than sequence length-1")
        prompt = self._fill(length)
        target_pos = length - 1 - distance
        value = self._int((self.VALUE_LOW, self.VALUE_HIGH))
        prompt[0], prompt[target_pos], prompt[-1] = self.BOS, value, self.QUERY
        return self._item(prompt, [value], target_pos, distance)

    def _relative_class(self, exact=False):
        length = self._int(self.length_range)
        distance = self._int(self.distance_range)
        left = self._int((1, max(1, length - distance - 2)))
        right = left + distance
        if right >= length - 1:
            left, right = 1, 1 + distance
        prompt = self._fill(length)
        prompt[0], prompt[left], prompt[right], prompt[-1] = self.BOS, self.MARK_A, self.MARK_B, self.QUERY
        label = distance % 2 if exact else min(distance, 15) // 4
        return self._item(prompt, [self.CLASS_LOW + label], left, distance,
                          distance_label=label)

    def _content_retrieval(self, distant=False, correlation=None):
        length = self._int(self.length_range)
        key = self._int((self.KEY_LOW, self.KEY_HIGH))
        value = self._int((self.VALUE_LOW, self.VALUE_HIGH))
        prompt = self._fill(length)
        prompt[0], prompt[-2], prompt[-1] = self.BOS, self.QUERY, key
        if correlation is not None:
            near = ((key - self.KEY_LOW) % 2 == 0) == (correlation > 0)
            pos = length - 5 if near else 1
        elif distant:
            pos = 1
        else:
            pos = self._int((1, length - 4))
        prompt[pos:pos + 2] = [key, value]
        # A very near, semantically wrong distractor makes content-vs-distance explicit.
        wrong_key = self.KEY_LOW + ((key - self.KEY_LOW + 1) % (self.KEY_HIGH - self.KEY_LOW + 1))
        prompt[-5:-3] = [wrong_key, self._int((self.VALUE_LOW, self.VALUE_HIGH))]
        return self._item(prompt, [value], pos, length - 2 - pos,
                          correlation=correlation, target_key=key)

    def _latest_update(self):
        length = self._int(self.length_range)
        key = self._int((self.KEY_LOW, self.KEY_HIGH))
        prompt = self._fill(length)
        prompt[0] = self.BOS
        positions = sorted(self.rng.choice(np.arange(1, length - 5), size=2, replace=False).tolist())
        last_value = None
        for pos in positions:
            last_value = self._int((self.VALUE_LOW, self.VALUE_HIGH))
            prompt[pos:pos + 3] = [self.UPDATE, key, last_value]
        prompt[-2:] = [self.QUERY, key]
        return self._item(prompt, [last_value], positions[-1], length - 2 - positions[-1])

    def _selective_count(self):
        length = self._int(self.length_range)
        count = min(self._int(self.relevant_count_range), length - 3)
        prompt = self._fill(length)
        prompt[0], prompt[-1] = self.RESET, self.QUERY
        positions = self.rng.choice(np.arange(1, length - 1), size=count, replace=False)
        prompt[positions] = self.RELEVANT
        return self._item(prompt, [self.CLASS_LOW + count], int(positions[-1]),
                          length - 1 - int(positions[-1]), relevant_count=count,
                          number_of_distractors=length - count - 2)

    def _periodic_distance(self):
        item = self._relative_class(exact=True)
        distance = item.metadata["relative_distance"]
        item.answer[0] = self.CLASS_LOW + distance % self.period
        item.metadata["distance_label"] = distance % self.period
        return item

    def _threshold_position(self):
        item = self._absolute()
        pos = item.metadata["absolute_target_position"]
        item.answer[0] = self.CLASS_LOW + int(pos >= self.threshold)
        item.metadata["threshold_label"] = int(pos >= self.threshold)
        return item

    def _sample_one(self):
        task = self.task
        if task in ("absolute_position_parity", "absolute_position_extrapolation",
                    "dense_absolute_position"):
            return self._absolute()
        if task in ("relative_offset_copy", "unseen_long_relative_offset"):
            return self._relative_copy()
        if task == "bucketed_relative_distance":
            return self._relative_class(exact=False)
        if task == "exact_distance_beyond_bucket":
            return self._relative_class(exact=True)
        if task == "content_addressed_retrieval":
            return self._content_retrieval()
        if task == "distant_retrieval":
            return self._content_retrieval(distant=True)
        if task == "context_adaptive_retrieval":
            return self._content_retrieval(correlation=1)
        if task == "context_correlation_flip":
            return self._content_retrieval(correlation=self.correlation)
        if task == "local_latest_update":
            return self._latest_update()
        if task == "selective_count":
            return self._selective_count()
        if task == "periodic_distance":
            return self._periodic_distance()
        if task == "nonperiodic_absolute_threshold":
            return self._threshold_position()
        raise AssertionError(task)

    TASKS = frozenset(POSITIONAL_TASK_DEFAULTS)
