"""
qarp_mock.engines — QulacsEngine and TketEngine Mock
=====================================================
Mirrors ``qarp.engines`` from QARP v0.4.3.

QARP v0.4.3 Engine API:
  engine.build(measurements=[PrimitiveAlgorithm(...)])
  result = engine.run(parameters={symbol: value})
  samples = engine.sample(circuit=block, n_shots=1000)

On FX700, QulacsEngine uses Qulacs directly; TketEngine dispatches
to any pytket-compatible backend (Qulacs, Tenet, Qiskit Aer).
This mock uses Qulacs state-vector simulation for both.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class QulacsEngine:
    """
    Mock of qarp.engines.QulacsEngine — CPU state-vector simulator.

    Parameters
    ----------
    parallelize : bool — parallelize Qulacs across Pauli terms (no-op in mock)
    """

    def __init__(self, parallelize: bool = False):
        self.parallelize = parallelize
        self._measurements = []
        self._built = False
        logger.info(f"QulacsEngine(mock) initialized (parallelize={parallelize})")

    def build(self, measurements: list) -> None:
        """Register measurements (PrimitiveAlgorithm instances) to evaluate."""
        self._measurements = list(measurements)
        self._built = True

    def run(self, parameters: dict = None) -> "complex | float | list":
        """
        Execute registered measurements with the given parameter bindings.

        Parameters
        ----------
        parameters : dict mapping symbol names to float values

        Returns
        -------
        Single value if one measurement, list if multiple.
        """
        if parameters is None:
            parameters = {}

        results = []
        for m in self._measurements:
            val = m.evaluate(parameters)
            results.append(val)

        return results[0] if len(results) == 1 else results

    def sample(self, circuit, n_shots: int, device=None) -> Dict[tuple, float]:
        """
        Sample the circuit n_shots times.

        Returns
        -------
        dict mapping bitstring tuples to probability estimates.
        Example: {(0, 1, 0): 0.45, (1, 0, 1): 0.55}
        """
        from backends.qarp_mock.qulacs_compat import QuantumState

        qulacs_circuit = circuit.to_qulacs()
        n = circuit.n_qubits
        state = QuantumState(n)
        state.set_zero_state()
        qulacs_circuit.update_quantum_state(state)

        raw = state.sampling(n_shots)
        counts: Dict[tuple, int] = {}
        for val in raw:
            bitstring = tuple(
                int(b) for b in format(int(val), f'0{n}b')[::-1])
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return {k: v / n_shots for k, v in counts.items()}

    def run_gradient(self, parameters: dict = None) -> list:
        """Gradient computation (not implemented in mock)."""
        raise NotImplementedError(
            "Gradient computation not available in mock engine")

    def __repr__(self) -> str:
        return f"QulacsEngine(mock, parallelize={self.parallelize})"


class TketEngine:
    """
    Mock of qarp.engines.TketEngine — pytket backend wrapper.

    Parameters
    ----------
    backend : pytket Backend object (ignored in mock; uses Qulacs)
    """

    def __init__(self, backend=None):
        self.backend = backend
        self._measurements = []
        self._built = False
        self._backend_name = (type(backend).__name__ if backend
                              else "mock_qulacs")
        logger.info(f"TketEngine(mock) initialized: backend={self._backend_name}")

    def build(self, measurements: list) -> None:
        self._measurements = list(measurements)
        self._built = True

    def run(self, parameters: dict = None) -> "complex | float | list":
        if parameters is None:
            parameters = {}
        results = []
        for m in self._measurements:
            val = m.evaluate(parameters)
            results.append(val)
        return results[0] if len(results) == 1 else results

    def sample(self, circuit, n_shots: int, device=None) -> Dict[tuple, float]:
        from backends.qarp_mock.qulacs_compat import QuantumState

        qulacs_circuit = circuit.to_qulacs()
        n = circuit.n_qubits
        state = QuantumState(n)
        state.set_zero_state()
        qulacs_circuit.update_quantum_state(state)

        raw = state.sampling(n_shots)
        counts: Dict[tuple, int] = {}
        for val in raw:
            bitstring = tuple(
                int(b) for b in format(int(val), f'0{n}b')[::-1])
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return {k: v / n_shots for k, v in counts.items()}

    def __repr__(self) -> str:
        return f"TketEngine(mock, backend={self._backend_name})"
