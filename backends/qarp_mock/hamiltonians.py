"""
qarp_mock.hamiltonians — PauliHamiltonian and PauliTerm
=======================================================
Mirrors ``qarp.hamiltonians`` API.  Stores Pauli operator terms
and can convert to a Qulacs Observable for expectation value evaluation.
"""

from __future__ import annotations
from typing import List, Tuple


class PauliTerm:
    """
    A single term in a Pauli Hamiltonian: coefficient × Π (Pauli_op on qubit).

    Example:
        PauliTerm(0.5, [(0, 'Z'), (1, 'Z')])  →  0.5 · Z₀ Z₁
        PauliTerm(-0.3, [(2, 'Z')])            →  -0.3 · Z₂
    """

    def __init__(self, coefficient: float,
                 qubit_ops: List[Tuple[int, str]]) -> None:
        self.coefficient = float(coefficient)
        self.qubit_ops = list(qubit_ops)  # [(qubit_idx, 'X'|'Y'|'Z'), ...]

    def __repr__(self) -> str:
        ops = " ".join(f"{op}{q}" for q, op in self.qubit_ops)
        return f"PauliTerm({self.coefficient:+.6f}, {ops})"


class PauliHamiltonian:
    """
    Sum of PauliTerms:  H = Σ_k c_k · P_k

    Parameters
    ----------
    terms    : list of PauliTerm objects
    n_qubits : total number of qubits in the system
    """

    def __init__(self, terms: List[PauliTerm], n_qubits: int) -> None:
        self.terms = list(terms)
        self.n_qubits = int(n_qubits)

    def to_qulacs_observable(self):
        """Convert to a Qulacs Observable for state-vector simulation."""
        from qulacs import Observable
        obs = Observable(self.n_qubits)
        for term in self.terms:
            # Qulacs format: "Z 0 Z 1" for a ZZ term on qubits 0 and 1
            pauli_str = " ".join(f"{op} {q}" for q, op in term.qubit_ops)
            obs.add_operator(term.coefficient, pauli_str)
        return obs

    def n_terms(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        return f"PauliHamiltonian({self.n_terms()} terms, {self.n_qubits} qubits)"
