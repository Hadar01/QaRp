"""
qarp_mock.openfermion_mock — Mock openfermion.QubitOperator
==========================================================
On FX700, the real ``openfermion`` package provides QubitOperator.
Locally, this mock enables full integration testing.

QARP v0.4.3 uses openfermion.QubitOperator for Hamiltonians instead
of the old PauliHamiltonian/PauliTerm classes.

Usage:
    op = QubitOperator("Z0 Z1", 0.5)   # 0.5 * Z_0 * Z_1
    op += QubitOperator("Z0", -0.3)     # + (-0.3) * Z_0
    op += QubitOperator("", 1.0)        # + 1.0 * I (identity)
"""

from __future__ import annotations
from typing import Dict, Tuple


class QubitOperator:
    """
    Mock of openfermion.QubitOperator.

    Represents a sum of Pauli strings: H = Σ_k c_k * P_k
    where P_k is a product of Pauli operators on specified qubits.

    Terms are stored as a dict: tuple of (qubit, pauli) → complex coefficient.
    Identity is represented by the empty tuple ().
    """

    def __init__(self, term: str = "", coefficient: complex = 1.0):
        self.terms: Dict[tuple, complex] = {}
        self._n_qubits_hint: int = 0  # Set externally for Ising models
        if term is not None:
            parsed = self._parse_term(term)
            self.terms[parsed] = complex(coefficient)

    @staticmethod
    def _parse_term(term_str: str) -> tuple:
        """Parse 'Z0 Z1' into ((0, 'Z'), (1, 'Z'))."""
        if not term_str.strip():
            return ()  # Identity
        ops = []
        for token in term_str.strip().split():
            pauli = token[0].upper()
            qubit = int(token[1:])
            ops.append((qubit, pauli))
        return tuple(sorted(ops))

    def __iadd__(self, other):
        if isinstance(other, QubitOperator):
            for term, coeff in other.terms.items():
                self.terms[term] = self.terms.get(term, 0.0) + coeff
            self._n_qubits_hint = max(self._n_qubits_hint,
                                       other._n_qubits_hint)
        return self

    def __add__(self, other):
        result = QubitOperator.__new__(QubitOperator)
        result.terms = dict(self.terms)
        result._n_qubits_hint = self._n_qubits_hint
        if isinstance(other, QubitOperator):
            for term, coeff in other.terms.items():
                result.terms[term] = result.terms.get(term, 0.0) + coeff
            result._n_qubits_hint = max(result._n_qubits_hint,
                                         other._n_qubits_hint)
        return result

    def __sub__(self, other):
        result = QubitOperator.__new__(QubitOperator)
        result.terms = dict(self.terms)
        result._n_qubits_hint = self._n_qubits_hint
        if isinstance(other, QubitOperator):
            for term, coeff in other.terms.items():
                result.terms[term] = result.terms.get(term, 0.0) - coeff
            result._n_qubits_hint = max(result._n_qubits_hint,
                                         other._n_qubits_hint)
        return result

    def __mul__(self, scalar):
        result = QubitOperator.__new__(QubitOperator)
        result.terms = {k: v * scalar for k, v in self.terms.items()}
        result._n_qubits_hint = self._n_qubits_hint
        return result

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return self.__mul__(-1)

    @property
    def n_qubits(self) -> int:
        """Infer number of qubits from operator terms (or hint)."""
        max_q = -1
        for term in self.terms:
            for q, _ in term:
                if q > max_q:
                    max_q = q
        inferred = max_q + 1 if max_q >= 0 else 0
        return max(inferred, self._n_qubits_hint)

    def get_constant(self) -> float:
        """Get the identity (constant) term coefficient."""
        return float(self.terms.get((), 0.0).real
                     if isinstance(self.terms.get((), 0.0), complex)
                     else self.terms.get((), 0.0))

    def to_qulacs_observable(self, n_qubits: int):
        """Convert non-identity terms to a Qulacs Observable."""
        from backends.qarp_mock.qulacs_compat import Observable
        obs = Observable(n_qubits)
        for term, coeff in self.terms.items():
            if abs(coeff) < 1e-15 or not term:
                continue  # Skip identity and near-zero
            pauli_str = " ".join(f"{p} {q}" for q, p in term)
            obs.add_operator(float(coeff.real if isinstance(coeff, complex) else coeff),
                             pauli_str)
        return obs

    def __repr__(self) -> str:
        parts = []
        for term, coeff in self.terms.items():
            c = float(coeff.real if isinstance(coeff, complex) else coeff)
            if not term:
                parts.append(f"{c:+.4f} []")
            else:
                ops = " ".join(f"{p}{q}" for q, p in term)
                parts.append(f"{c:+.4f} [{ops}]")
        return " ".join(parts) if parts else "0"
