"""
numpy_simulator.py
==================
Pure-numpy statevector quantum simulator — drop-in replacement for Qulacs.

Provides QuantumState, QuantumCircuit, and Observable with identical API
to qulacs, implemented entirely in numpy.  No compiled C extensions.

Performance: fine for ≤20 qubits (2^20 = 1M complex amplitudes).
"""

import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Single-qubit gate matrices
# ---------------------------------------------------------------------------
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


# ---------------------------------------------------------------------------
# QuantumState
# ---------------------------------------------------------------------------

class QuantumState:
    """Pure-numpy statevector |ψ⟩ with 2^n amplitudes."""

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.dim = 1 << n_qubits
        self._vec = np.zeros(self.dim, dtype=complex)

    def set_zero_state(self):
        self._vec[:] = 0.0
        self._vec[0] = 1.0

    def get_vector(self) -> np.ndarray:
        return self._vec.copy()

    def set_vector(self, vec):
        self._vec = np.array(vec, dtype=complex)

    def get_qubit_count(self):
        return self.n

    def sampling(self, n_shots: int) -> list:
        """Sample from |⟨x|ψ⟩|² distribution, returns list of int indices."""
        probs = np.abs(self._vec) ** 2
        total = probs.sum()
        if total < 1e-15 or np.isnan(total):
            # State collapsed to zero — fall back to uniform sampling
            probs = np.ones(self.dim) / self.dim
        else:
            probs /= total  # normalise (numerical safety)
        # Replace any remaining NaN/negative values
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.maximum(probs, 0.0)
        psum = probs.sum()
        if psum < 1e-15:
            probs = np.ones(self.dim) / self.dim
        else:
            probs /= psum
        return list(np.random.choice(self.dim, size=n_shots, p=probs))

    def copy(self):
        s = QuantumState(self.n)
        s._vec = self._vec.copy()
        return s


# ---------------------------------------------------------------------------
# QuantumCircuit
# ---------------------------------------------------------------------------

class QuantumCircuit:
    """
    Gate-list quantum circuit that applies to QuantumState.

    Uses tensor-product approach: for an n-qubit state stored as shape (2,)*n,
    apply the gate by contracting on the target qubit axis.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self._gates: list = []  # (type, args)

    # -- Single-qubit gates ---
    def add_H_gate(self, target: int):
        self._gates.append(('1q', target, _H))

    def add_X_gate(self, target: int):
        self._gates.append(('1q', target, _X))

    def add_Y_gate(self, target: int):
        self._gates.append(('1q', target, _Y))

    def add_Z_gate(self, target: int):
        self._gates.append(('1q', target, _Z))

    def add_RX_gate(self, target: int, angle: float):
        self._gates.append(('1q', target, _rx(angle)))

    def add_RY_gate(self, target: int, angle: float):
        self._gates.append(('1q', target, _ry(angle)))

    def add_RZ_gate(self, target: int, angle: float):
        self._gates.append(('1q', target, _rz(angle)))

    # -- Two-qubit gates ---
    def add_CNOT_gate(self, control: int, target: int):
        self._gates.append(('cnot', control, target))

    def add_CZ_gate(self, control: int, target: int):
        self._gates.append(('cz', control, target))

    def merge_circuit(self, other: 'QuantumCircuit'):
        """Append all gates from another circuit into this one (matches qulacs API)."""
        self._gates.extend(other._gates)

    # -- Apply to state ---
    def update_quantum_state(self, state: QuantumState):
        """Apply all gates to the state in order."""
        # Work with the state vector reshaped as (2,2,...,2) tensor
        psi = state._vec.reshape((2,) * state.n)
        for gate in self._gates:
            if gate[0] == '1q':
                _, target, mat = gate
                psi = self._apply_1q(psi, target, mat, state.n)
            elif gate[0] == 'cnot':
                _, control, target = gate
                psi = self._apply_cnot(psi, control, target, state.n)
            elif gate[0] == 'cz':
                _, control, target = gate
                psi = self._apply_cz(psi, control, target, state.n)
        state._vec = psi.reshape(-1)

    @staticmethod
    def _apply_1q(psi: np.ndarray, target: int, mat: np.ndarray, n: int) -> np.ndarray:
        """Apply a 2×2 gate to qubit `target` of an n-qubit tensor."""
        # np.tensordot contracts axis `target` of psi with axis 1 of mat
        # Then move the result axis back to position `target`
        psi = np.tensordot(mat, psi, axes=([1], [target]))
        # tensordot puts the new axis at position 0; move it to `target`
        psi = np.moveaxis(psi, 0, target)
        return psi

    @staticmethod
    def _apply_cnot(psi: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
        """Apply CNOT: flip target qubit when control=|1⟩."""
        # Build index for control=1 subspace
        idx = [slice(None)] * n
        idx[control] = 1
        # Flip target in that subspace
        psi_sub = psi[tuple(idx)].copy()
        idx_flip = [slice(None)] * (n - 1)
        # target index shifts if target > control (since we sliced control out)
        t_adj = target if target < control else target - 1
        # Swap |0⟩ and |1⟩ on the target axis within the control=1 subspace
        idx0 = [slice(None)] * (n - 1)
        idx0[t_adj] = 0
        idx1 = [slice(None)] * (n - 1)
        idx1[t_adj] = 1
        tmp = psi_sub[tuple(idx0)].copy()
        psi_sub[tuple(idx0)] = psi_sub[tuple(idx1)]
        psi_sub[tuple(idx1)] = tmp
        psi[tuple(idx)] = psi_sub
        return psi

    @staticmethod
    def _apply_cz(psi: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
        """Apply CZ: phase-flip when both control and target are |1⟩."""
        idx = [slice(None)] * n
        idx[control] = 1
        idx[target] = 1
        psi[tuple(idx)] *= -1
        return psi


# ---------------------------------------------------------------------------
# Observable
# ---------------------------------------------------------------------------

class Observable:
    """
    Hermitian observable as sum of Pauli strings.

    Supports the same API as qulacs.Observable:
        obs.add_operator(coeff, "Z 0 Z 1")
        energy = obs.get_expectation_value(state)
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self._terms: list[tuple[float, list[tuple[str, int]]]] = []

    def add_operator(self, coeff: float, pauli_string: str):
        """
        Add a Pauli term.  pauli_string format: "Z 0 Z 1" or "X 2".
        Empty string = identity.
        """
        paulis = []
        if pauli_string.strip():
            tokens = pauli_string.strip().split()
            for i in range(0, len(tokens), 2):
                op = tokens[i]
                qubit = int(tokens[i + 1])
                paulis.append((op, qubit))
        self._terms.append((float(coeff), paulis))

    def get_expectation_value(self, state: QuantumState) -> float:
        """Compute ⟨ψ|O|ψ⟩ by applying each Pauli string to |ψ⟩."""
        total = 0.0
        psi = state._vec
        for coeff, paulis in self._terms:
            if not paulis:
                # Identity term
                total += coeff * float(np.real(np.vdot(psi, psi)))
            else:
                # Apply Pauli operators to |ψ⟩ and compute ⟨ψ|P|ψ⟩
                phi = psi.copy().reshape((2,) * state.n)
                for op, qubit in paulis:
                    mat = {'X': _X, 'Y': _Y, 'Z': _Z}[op]
                    phi = QuantumCircuit._apply_1q(phi, qubit, mat, state.n)
                phi = phi.reshape(-1)
                total += coeff * float(np.real(np.vdot(psi, phi)))
        return total
