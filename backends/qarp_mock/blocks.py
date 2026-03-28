"""
qarp_mock.blocks — Block hierarchy for circuit construction
===========================================================
Mirrors ``qarp.blocks`` API from QARP v0.4.3.

QARP v0.4.3 uses a Block abstraction instead of the old ParametricCircuit.
Blocks are composable circuit building elements with named Symbol parameters.

Key classes:
  - Block           — base class with build(), set_symbols(), to_qulacs()
  - CompositeBlock  — sequences multiple blocks
  - HnBlock         — Hadamard on all qubits (H^⊗n)
  - IsingCostBlock  — Problem unitary exp(-iγ H_P) for Ising Hamiltonian
  - MixerBlock      — Mixer unitary exp(-iβ Σ X_i)
  - LayeredHEABlock — Hardware-efficient ansatz
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np


class Block:
    """Base block class matching QARP v0.4.3 Block interface."""

    def __init__(self, n_qubits: int, name: str = "Block"):
        self.n_qubits = n_qubits
        self.name = name
        self._built = False
        self._symbols: Dict[str, float] = {}
        self._original_qubit_map: Optional[Dict[int, int]] = None

    def build(self):
        self._built = True
        return self

    def set_symbols(self, symbol_map: Dict[str, float]):
        """Bind symbol names to float values."""
        for name, val in symbol_map.items():
            if name in self._symbols:
                self._symbols[name] = float(val)
        return self

    def get_symbols(self) -> List[str]:
        """Return ordered list of symbol names in this block."""
        return list(self._symbols.keys())

    # Alias for backward compatibility with tests
    get_parameter_names = get_symbols

    def to_qulacs(self, noise_model=None):
        """Convert to Qulacs QuantumCircuit."""
        from backends.qarp_mock.qulacs_compat import QuantumCircuit
        return QuantumCircuit(self.n_qubits)

    def dagger(self):
        raise NotImplementedError

    @property
    def is_built(self) -> bool:
        return self._built

    def __repr__(self) -> str:
        return f"{self.name}({self.n_qubits}q, {len(self._symbols)} symbols)"


class CompositeBlock(Block):
    """Sequences multiple blocks — mirrors qarp.blocks.CompositeBlock."""

    def __init__(self, blocks: Sequence[Block], n_qubits: int = None,
                 name: str = "CompositeBlock"):
        n = n_qubits or max(b.n_qubits for b in blocks)
        super().__init__(n, name)
        self.blocks = list(blocks)

    def build(self):
        for b in self.blocks:
            b.build()
        self._built = True
        return self

    def set_symbols(self, symbol_map: Dict[str, float]):
        for b in self.blocks:
            b.set_symbols(symbol_map)
        # Also update our local copy
        for name, val in symbol_map.items():
            if name in self._symbols:
                self._symbols[name] = float(val)
        return self

    def get_symbols(self) -> List[str]:
        syms = []
        seen = set()
        for b in self.blocks:
            for s in b.get_symbols():
                if s not in seen:
                    seen.add(s)
                    syms.append(s)
        return syms

    get_parameter_names = get_symbols

    def to_qulacs(self, noise_model=None):
        from backends.qarp_mock.qulacs_compat import QuantumCircuit
        circuit = QuantumCircuit(self.n_qubits)
        for b in self.blocks:
            sub = b.to_qulacs(noise_model)
            circuit.merge_circuit(sub)
        return circuit

    def __repr__(self) -> str:
        return (f"CompositeBlock({self.n_qubits}q, "
                f"{len(self.blocks)} blocks, "
                f"{len(self.get_symbols())} symbols)")


class HnBlock(Block):
    """Hadamard on all qubits: H^⊗n — mirrors qarp.blocks.primitives.HnBlock."""

    def __init__(self, n_qubits: int, **kwargs):
        super().__init__(n_qubits, "Hn")

    def to_qulacs(self, noise_model=None):
        from backends.qarp_mock.qulacs_compat import QuantumCircuit
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.add_H_gate(i)
        return circuit


class IsingCostBlock(Block):
    """
    Problem unitary exp(-iγ H_P) for Ising Hamiltonian.

    Applies ZZ-interaction gates for J coupling terms and RZ gates for
    single-qubit h field terms, parameterized by a single gamma symbol.

    In real QARP, CostOperatorBlock serves a similar role for Graph problems.
    For arbitrary Ising models, we use this custom block.
    """

    def __init__(self, n_qubits: int, h: dict, J: dict, gamma_symbol: str):
        super().__init__(n_qubits, "IsingCost")
        self.h = dict(h)
        self.J = dict(J)
        self.gamma_symbol = gamma_symbol
        self._symbols[gamma_symbol] = 0.0

    def to_qulacs(self, noise_model=None):
        from backends.qarp_mock.qulacs_compat import QuantumCircuit
        circuit = QuantumCircuit(self.n_qubits)
        gamma = self._symbols.get(self.gamma_symbol, 0.0)

        for qi, h_i in self.h.items():
            if abs(h_i) > 1e-10:
                circuit.add_RZ_gate(qi, 2.0 * gamma * h_i)

        for (i, j), J_ij in self.J.items():
            if abs(J_ij) > 1e-10:
                circuit.add_CNOT_gate(i, j)
                circuit.add_RZ_gate(j, 2.0 * gamma * J_ij)
                circuit.add_CNOT_gate(i, j)

        return circuit


class MixerBlock(Block):
    """Mixer unitary exp(-iβ Σ X_i) — X-rotation on each qubit."""

    def __init__(self, n_qubits: int, beta_symbol: str):
        super().__init__(n_qubits, "Mixer")
        self.beta_symbol = beta_symbol
        self._symbols[beta_symbol] = 0.0

    def to_qulacs(self, noise_model=None):
        from backends.qarp_mock.qulacs_compat import QuantumCircuit
        circuit = QuantumCircuit(self.n_qubits)
        beta = self._symbols.get(self.beta_symbol, 0.0)
        for i in range(self.n_qubits):
            circuit.add_RX_gate(i, 2.0 * beta)
        return circuit


class LayeredHEABlock(Block):
    """
    Hardware-efficient ansatz with alternating rotation + entangling layers.
    Mirrors qarp.blocks.primitives.LayeredHEABlock.
    """

    def __init__(self, n_qubits: int, depth: int = 1,
                 symbol_prefix: str = "hea"):
        super().__init__(n_qubits, "LayeredHEA")
        self.depth = depth
        self.symbol_prefix = symbol_prefix
        for layer in range(depth):
            for q in range(n_qubits):
                for gate in ['ry', 'rz']:
                    name = f"{symbol_prefix}_{gate}_{layer}_{q}"
                    self._symbols[name] = 0.0

    def to_qulacs(self, noise_model=None):
        from backends.qarp_mock.qulacs_compat import QuantumCircuit
        circuit = QuantumCircuit(self.n_qubits)
        for layer in range(self.depth):
            for q in range(self.n_qubits):
                ry = self._symbols.get(
                    f"{self.symbol_prefix}_ry_{layer}_{q}", 0.0)
                rz = self._symbols.get(
                    f"{self.symbol_prefix}_rz_{layer}_{q}", 0.0)
                circuit.add_RY_gate(q, ry)
                circuit.add_RZ_gate(q, rz)
            for q in range(self.n_qubits - 1):
                circuit.add_CNOT_gate(q, q + 1)
        return circuit
