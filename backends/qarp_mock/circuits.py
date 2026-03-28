"""
qarp_mock.circuits — ParametricCircuit and CircuitCutter
========================================================
Mirrors ``qarp.circuits`` API.  ParametricCircuit stores gates with
named parameters; CircuitCutter partitions large circuits into fragments.
Both convert to Qulacs circuits for local execution.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


class ParametricCircuit:
    """
    A parametric quantum circuit with named angle parameters.

    Mirrors QARP's ParametricCircuit API:
      - add_H_gate, add_CNOT_gate              (fixed gates)
      - add_parametric_RZ_gate, add_parametric_RX_gate  (named parameters)
      - set_parameters(angle_dict)              (bind values)

    Internally stores a gate list; to_qulacs_circuit() builds the concrete
    Qulacs QuantumCircuit with current parameter values.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = int(n_qubits)
        self._gates: List[Tuple[str, ...]] = []
        self._param_values: Dict[str, float] = {}
        # Set by CircuitCutter for fragment ↔ original qubit tracking
        self._original_qubit_map: Optional[Dict[int, int]] = None

    # ── Fixed gates ──────────────────────────────────────────────────────

    def add_H_gate(self, qubit: int) -> None:
        self._gates.append(("H", qubit))

    def add_CNOT_gate(self, control: int, target: int) -> None:
        self._gates.append(("CNOT", control, target))

    # ── Parametric gates ─────────────────────────────────────────────────

    def add_parametric_RZ_gate(self, qubit: int, *,
                                parameter_name: str) -> None:
        self._gates.append(("pRZ", qubit, parameter_name))
        self._param_values.setdefault(parameter_name, 0.0)

    def add_parametric_RX_gate(self, qubit: int, *,
                                parameter_name: str) -> None:
        self._gates.append(("pRX", qubit, parameter_name))
        self._param_values.setdefault(parameter_name, 0.0)

    # ── Parameter binding ────────────────────────────────────────────────

    def set_parameters(self, angle_dict: Dict[str, float]) -> None:
        """Bind named parameters to concrete angle values."""
        for name, value in angle_dict.items():
            if name in self._param_values:
                self._param_values[name] = float(value)

    def get_parameter_names(self) -> List[str]:
        """Return all named parameter names in gate order."""
        seen = set()
        names = []
        for gate in self._gates:
            if gate[0] in ("pRZ", "pRX"):
                name = gate[2]
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    # ── Qulacs conversion ────────────────────────────────────────────────

    def to_qulacs_circuit(self):
        """Build a concrete Qulacs QuantumCircuit with current parameter values."""
        from qulacs import QuantumCircuit
        circuit = QuantumCircuit(self.n_qubits)
        for gate in self._gates:
            gtype = gate[0]
            if gtype == "H":
                circuit.add_H_gate(gate[1])
            elif gtype == "CNOT":
                circuit.add_CNOT_gate(gate[1], gate[2])
            elif gtype == "pRZ":
                qubit, param_name = gate[1], gate[2]
                angle = self._param_values.get(param_name, 0.0)
                circuit.add_RZ_gate(qubit, angle)
            elif gtype == "pRX":
                qubit, param_name = gate[1], gate[2]
                angle = self._param_values.get(param_name, 0.0)
                circuit.add_RX_gate(qubit, angle)
        return circuit

    def gate_count(self) -> int:
        return len(self._gates)

    def __repr__(self) -> str:
        return (f"ParametricCircuit({self.n_qubits}q, "
                f"{self.gate_count()} gates, "
                f"{len(self._param_values)} params)")


class CircuitCutter:
    """
    Decomposes a large ParametricCircuit into smaller fragments.

    Mirrors QARP's CircuitCutter API:
      - cut()         → list of fragment ParametricCircuits
      - reconstruct() → combined energy from fragment energies

    The mock uses qubit-index-based partitioning. On FX700, QARP's native
    CircuitCutter performs gate-level cutting with optimal wire selection.
    """

    def __init__(self, circuit: ParametricCircuit,
                 max_fragment_qubits: int = 20) -> None:
        self.circuit = circuit
        self.max_frag = int(max_fragment_qubits)
        self.n_qubits = circuit.n_qubits
        self._fragments: Optional[List[ParametricCircuit]] = None
        self._fragment_qubit_maps: Optional[List[List[int]]] = None

    def cut(self) -> List[ParametricCircuit]:
        """
        Partition the circuit into fragments of ≤ max_fragment_qubits.

        Returns a list of ParametricCircuit fragments, each with its own
        local qubit numbering and an _original_qubit_map for reconstruction.
        """
        n = self.n_qubits
        fragments = []
        qubit_maps = []

        for start in range(0, n, self.max_frag):
            end = min(start + self.max_frag, n)
            chunk_qubits = list(range(start, end))
            frag = self._build_fragment(chunk_qubits)
            fragments.append(frag)
            qubit_maps.append(chunk_qubits)

        self._fragments = fragments
        self._fragment_qubit_maps = qubit_maps
        return fragments

    def _build_fragment(self, qubit_indices: List[int]) -> ParametricCircuit:
        """Build a sub-circuit for the given qubit indices."""
        n_frag = len(qubit_indices)
        frag = ParametricCircuit(n_frag)
        qubit_set = set(qubit_indices)
        local_map = {q: i for i, q in enumerate(qubit_indices)}

        # Store the mapping: local qubit i → original qubit q
        frag._original_qubit_map = {i: q for i, q in enumerate(qubit_indices)}

        for gate in self.circuit._gates:
            gtype = gate[0]
            if gtype == "H":
                q = gate[1]
                if q in qubit_set:
                    frag.add_H_gate(local_map[q])
            elif gtype == "CNOT":
                ctrl, tgt = gate[1], gate[2]
                if ctrl in qubit_set and tgt in qubit_set:
                    frag.add_CNOT_gate(local_map[ctrl], local_map[tgt])
            elif gtype == "pRZ":
                q, param_name = gate[1], gate[2]
                if q in qubit_set:
                    frag.add_parametric_RZ_gate(
                        local_map[q], parameter_name=param_name)
                    frag._param_values[param_name] = \
                        self.circuit._param_values.get(param_name, 0.0)
            elif gtype == "pRX":
                q, param_name = gate[1], gate[2]
                if q in qubit_set:
                    frag.add_parametric_RX_gate(
                        local_map[q], parameter_name=param_name)
                    frag._param_values[param_name] = \
                        self.circuit._param_values.get(param_name, 0.0)

        return frag

    def reconstruct(self, fragment_energies: List[float]) -> float:
        """
        Combine fragment expectation values into a total energy estimate.

        Basic method: sum fragment energies (ignores cross-fragment correlations).
        QARP's native CircuitCutter applies wire-cutting correction terms here.
        """
        return sum(fragment_energies)

    def n_fragments(self) -> int:
        return len(self._fragments) if self._fragments else 0

    def __repr__(self) -> str:
        n_frag = self.n_fragments()
        return (f"CircuitCutter({self.n_qubits}q → "
                f"{n_frag} fragments, max={self.max_frag}q)")
