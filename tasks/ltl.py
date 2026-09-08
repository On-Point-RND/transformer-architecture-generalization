from typing import Tuple

import numpy as np

from .base import DatasetItem, Task

MAX_ATTEMPTS = 1000

OPS = ("NOT", "X", "F", "G", "AND", "OR", "IMPL", "U")
UNARY, BINARY = OPS[:4], OPS[4:]
LOCAL_UNARY, LOCAL_BINARY = ("X", "X", "X", "NOT"), ("AND", "OR", "IMPL")
P_UNARY = 0.7 
UNBOUNDED = -1  


def radius_of(node) -> int:
    op = node[0]
    if op == "atom":
        return 0
    if op in ("F", "G", "U"):
        return UNBOUNDED
    reach = [radius_of(child) for child in node[1:]]
    if UNBOUNDED in reach:
        return UNBOUNDED
    return max(reach) + (op == "X")


def _until(a, b):
    """a U b at every position: v[i] = b[i] or (a[i] and v[i+1]), right to left."""
    v = np.zeros(len(b), dtype=bool)
    carry = False
    for i in range(len(b) - 1, -1, -1):
        carry = bool(b[i]) or (bool(a[i]) and carry)
        v[i] = carry
    return v


def _unary(op, a):
    if op == "NOT":
        return ~a
    if op == "X":
        return np.append(a[1:], False)
    if op == "F":
        return np.logical_or.accumulate(a[::-1])[::-1]
    return np.logical_and.accumulate(a[::-1])[::-1]  # G


def _binary(op, a, b):
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "IMPL":
        return ~a | b
    return _until(a, b)


def evaluate(node, bits):
    """bits: [T, n_atoms] bool -> [T] bool, the subformula's value per position."""
    op = node[0]
    if op == "atom":
        return bits[:, node[1]]
    if op in UNARY:
        return _unary(op, evaluate(node[1], bits))
    return _binary(op, evaluate(node[1], bits), evaluate(node[2], bits))


class LTLTask(Task):
    PAD_ID = 0
    TRACE_MARKER_ID = 1
    QUERY_MARKER_ID = 2
    FALSE_ID = 3
    TRUE_ID = 4
    N_SPECIAL = 5

    def __init__(
        self,
        n_atoms: int = 3,
        trace_len: int | Tuple[int, int] = (16, 32),
        formula_size: int | Tuple[int, int] = (3, 11),
        radius: int | Tuple[int, int] = (0, 5),
        mode: str = "mixed",
        atom_density: Tuple[float, float] = (0.02, 0.98),
        seed: int | None = 42,
    ):
        if mode not in ("local", "global", "mixed"):
            raise ValueError(f"mode must be local/global/mixed, got {mode!r}")

        self.n_atoms = n_atoms
        self.trace_len = trace_len
        self.formula_size = formula_size
        self.radius = radius
        self.mode = mode
        self.atom_density = tuple(atom_density)

        self.unary = LOCAL_UNARY if mode == "local" else UNARY
        self.binary = LOCAL_BINARY if mode == "local" else BINARY
        self.op_ids = {op: self.N_SPECIAL + i for i, op in enumerate(OPS)}
        self.atom_ids = np.arange(n_atoms) + self.N_SPECIAL + len(OPS)
        self.symbol_ids = np.arange(2 ** n_atoms) + self.N_SPECIAL + len(OPS) + n_atoms
        self.powers = 1 << np.arange(n_atoms)

        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + len(OPS) + self.n_atoms + 2 ** self.n_atoms

    @property
    def min_block_size(self) -> int:
        """len(prompt) + len(answer) = (F + T + 2) + 1 <= block_size + 1"""
        return self._high(self.formula_size) + self._high(self.trace_len) + 2

    @staticmethod
    def _high(bounds) -> int:
        return bounds if isinstance(bounds, int) else bounds[1]

    def _int(self, bounds) -> int:
        if isinstance(bounds, int):
            return bounds
        return int(self.rng.integers(bounds[0], bounds[1] + 1))

    def _pick(self, options):
        return options[int(self.rng.integers(len(options)))]

    def _grow(self, size):
        """A random rule tree with exactly ``size`` nodes."""
        if size <= 1:
            return ("atom", int(self.rng.integers(self.n_atoms)))
        if size == 2 or self.rng.random() < P_UNARY:
            return (self._pick(self.unary), self._grow(size - 1))
        left = int(self.rng.integers(1, size - 1))
        return (self._pick(self.binary), self._grow(left), self._grow(size - 1 - left))

    def _trace(self, length):
        density = self.rng.uniform(*self.atom_density, size=self.n_atoms)
        return self.rng.random((length, self.n_atoms)) < density

    def _tokens(self, node):
        if node[0] == "atom":
            return [int(self.atom_ids[node[1]])]
        tokens = [self.op_ids[node[0]]]
        for child in node[1:]:
            tokens.extend(self._tokens(child))
        return tokens

    def _wanted(self, radius, target_radius) -> bool:
        if self.mode == "global":
            return radius == UNBOUNDED
        if self.mode == "local":
            return radius == target_radius
        return True

    def _sample_one(self) -> DatasetItem:
        target = bool(self.rng.integers(2))
        target_radius = self._int(self.radius)
        for _ in range(MAX_ATTEMPTS):
            formula = self._grow(self._int(self.formula_size))
            radius = radius_of(formula)
            if not self._wanted(radius, target_radius):
                continue
            bits = self._trace(self._int(self.trace_len))
            if bool(evaluate(formula, bits)[0]) == target:
                return self._item(formula, bits, radius, target)
        raise RuntimeError(
            f"no rule with radius {target_radius} evaluated to {target} in "
            f"{MAX_ATTEMPTS} draws (mode={self.mode}, radius={self.radius}, "
            f"formula_size={self.formula_size}, atom_density={self.atom_density}); "
            f"a rule of radius r needs at least r+1 nodes, so raise formula_size "
            f"or lower radius"
        )

    def _item(self, formula, bits, radius, value) -> DatasetItem:
        rule = self._tokens(formula)
        symbols = bits.astype(np.int64) @ self.powers
        prompt = np.concatenate([
            rule,
            [self.TRACE_MARKER_ID],
            self.symbol_ids[symbols],
            [self.QUERY_MARKER_ID],
        ]).astype(np.int64)
        answer = np.array([self.TRUE_ID if value else self.FALSE_ID], dtype=np.int64)
        metadata = {
            "radius": radius,
            "kind": "global" if radius == UNBOUNDED else "local",
            "trace_len": len(bits),
            "formula_size": len(rule),
        }
        return DatasetItem(prompt=prompt, answer=answer, metadata=metadata)
