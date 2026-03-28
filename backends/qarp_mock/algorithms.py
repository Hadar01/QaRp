"""
qarp_mock.algorithms — Primitive and Composite Algorithms
==========================================================
Mirrors ``qarp.algorithms.primitives`` and ``qarp.algorithms.composite``
from QARP v0.4.3.

Primitives:
  - StateVector      — exact statevector expectation value
  - PauliAveraging   — shot-based Pauli expectation estimation
  - Sampler          — measurement sampling

Composites:
  - VQE              — Variational Quantum Eigensolver
  - QAOA             — Quantum Approximate Optimization Algorithm
  - AdaptVQE         — Adaptive VQE with operator pool
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import logging

from scipy.optimize import minimize as scipy_minimize

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Primitive Algorithms
# ═══════════════════════════════════════════════════════════════════════════════

class StateVector:
    """
    Exact statevector expectation value ⟨bra|operator|ket⟩.
    Mirrors qarp.algorithms.primitives.StateVector.

    Parameters
    ----------
    bra      : Block (state preparation, usually same as ket)
    operator : QubitOperator (Hamiltonian)
    ket      : Block (state preparation)
    """

    def __init__(self, bra=None, operator=None, ket=None,
                 n_shots=None, device=None):
        self.bra = bra
        self.operator = operator
        self.ket = ket

    def evaluate(self, parameters: dict) -> float:
        """Evaluate ⟨ψ(params)|H|ψ(params)⟩ using Qulacs state vector."""
        from qulacs import QuantumState, Observable

        block = self.ket or self.bra
        if block is None:
            return 0.0

        # Bind parameters to the circuit block
        block.set_symbols(parameters)

        n = block.n_qubits
        qulacs_circuit = block.to_qulacs()
        state = QuantumState(n)
        state.set_zero_state()
        qulacs_circuit.update_quantum_state(state)

        # Handle fragment circuits (from circuit cutting)
        if (hasattr(block, '_original_qubit_map')
                and block._original_qubit_map is not None):
            reverse_map = {orig: local
                           for local, orig in block._original_qubit_map.items()}
            obs = self._build_fragment_observable(n, reverse_map)
        else:
            obs = self.operator.to_qulacs_observable(n)

        constant = self.operator.get_constant()
        return float(obs.get_expectation_value(state)) + float(constant)

    def _build_fragment_observable(self, n_frag: int,
                                   reverse_map: dict):
        """Build Qulacs Observable with remapped qubit indices for fragment."""
        from qulacs import Observable
        obs = Observable(n_frag)
        for term, coeff in self.operator.terms.items():
            if not term:
                continue  # Identity handled separately
            c = float(coeff.real if isinstance(coeff, complex) else coeff)
            if abs(c) < 1e-15:
                continue
            local_ops = []
            all_in = True
            for q, p in term:
                if q in reverse_map:
                    local_ops.append((reverse_map[q], p))
                else:
                    all_in = False
                    break
            if all_in and local_ops:
                pauli_str = " ".join(f"{p} {lq}" for lq, p in local_ops)
                obs.add_operator(c, pauli_str)
        return obs


class PauliAveraging:
    """
    Shot-based Pauli expectation estimation with commuting grouping.
    Mirrors qarp.algorithms.primitives.PauliAveraging.

    In mock mode, delegates to exact StateVector evaluation.
    """

    def __init__(self, bra=None, operator=None, ket=None,
                 fully_optimized=False, n_shots=None, device=None):
        self.bra = bra
        self.operator = operator
        self.ket = ket
        self.n_shots = n_shots or 10000

    def evaluate(self, parameters: dict) -> float:
        sv = StateVector(bra=self.bra, operator=self.operator, ket=self.ket)
        return sv.evaluate(parameters)


class Sampler:
    """
    Measurement sampling of quantum state.
    Mirrors qarp.algorithms.primitives.Sampler.
    """

    def __init__(self, ket=None, device=None, n_shots=None):
        self.ket = ket
        self.n_shots = n_shots or 1000

    def evaluate(self, parameters: dict) -> Dict[tuple, float]:
        from qulacs import QuantumState

        block = self.ket
        if block is None:
            return {}

        block.set_symbols(parameters)
        n = block.n_qubits
        qulacs_circuit = block.to_qulacs()
        state = QuantumState(n)
        state.set_zero_state()
        qulacs_circuit.update_quantum_state(state)

        raw = state.sampling(self.n_shots)
        counts: Dict[tuple, int] = {}
        for val in raw:
            bitstring = tuple(
                int(b) for b in format(int(val), f'0{n}b')[::-1])
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return {k: v / self.n_shots for k, v in counts.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Composite Algorithms
# ═══════════════════════════════════════════════════════════════════════════════

class VQE:
    """
    Variational Quantum Eigensolver — mock implementation.
    Mirrors qarp.algorithms.composite.VQE.

    Usage:
        vqe = VQE(ansatz=block, hamiltonian=qubit_op,
                  optimizer=scipy_opt, engine=engine)
        vqe.build()
        energy, params = vqe.run(max_iter=100)
    """

    def __init__(self, ansatz, hamiltonian, optimizer,
                 primitive=None, engine=None):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer
        self.primitive = primitive
        self.engine = engine
        self._built = False

    def build(self):
        self._built = True
        return self

    def run(self, max_iter: int = None) -> Tuple[float, Optional[List[float]]]:
        """
        Run VQE optimization.

        Returns
        -------
        (energy, optimal_parameters) — matching QARP v0.4.3 return signature.
        """
        symbols = self.ansatz.get_symbols()
        n_params = len(symbols)

        if max_iter and hasattr(self.optimizer, 'options'):
            self.optimizer.options['maxiter'] = max_iter

        best_energy = float('inf')
        best_params = None

        for restart in range(3):
            rng = np.random.default_rng(seed=restart * 17 + 3)
            init_params = rng.uniform(0, np.pi, n_params)

            def cost_fn(params, _syms=symbols):
                param_map = dict(zip(_syms, params))
                if self.engine:
                    measurement = StateVector(
                        bra=self.ansatz, operator=self.hamiltonian,
                        ket=self.ansatz)
                    self.engine.build(measurements=[measurement])
                    return float(self.engine.run(parameters=param_map))
                else:
                    sv = StateVector(
                        bra=self.ansatz, operator=self.hamiltonian,
                        ket=self.ansatz)
                    return float(sv.evaluate(param_map))

            result = self.optimizer.minimize(
                objective_function=cost_fn,
                initial_parameters=init_params,
            )

            if result.fun < best_energy:
                best_energy = result.fun
                best_params = list(result.x)

        return float(best_energy), best_params


class QAOA:
    """
    Quantum Approximate Optimization Algorithm — mock implementation.
    Mirrors qarp.algorithms.composite.QAOA.

    Note: Real QARP QAOA expects a Graph problem object.
    For our Ising model use case, we use VQE with a custom QAOA ansatz instead.
    """

    def __init__(self, problem=None, mixer=None, optimizer=None,
                 depth: int = 1, **kwargs):
        self.problem = problem
        self.mixer = mixer
        self.optimizer = optimizer
        self.depth = depth

    def build(self):
        return self

    def run(self) -> Tuple[float, Optional[List[float]]]:
        raise NotImplementedError(
            "Mock QAOA requires a Graph problem. "
            "Use VQE with a custom QAOA ansatz block for Ising models.")


class AdaptVQE:
    """
    Adaptive VQE — mock implementation.
    Mirrors qarp.algorithms.composite.AdaptVQE.

    Converts QubitOperator back to IsingHamiltonian and delegates to
    our local AdaptVQEOptimizer implementation.

    Usage:
        adapt = AdaptVQE(reference_block=hf_state,
                         system_hamiltonian=qubit_op,
                         excitation_pool=pool_ops,
                         optimizer=scipy_opt, engine=engine)
        adapt.build()
        energy, params = adapt.run(max_iter=25)
    """

    def __init__(self, reference_block=None, system_hamiltonian=None,
                 excitation_pool=None, optimizer=None, gradient=None,
                 verbose=False, diminishing=False, exc_per_iter=1,
                 qubit_adapt=False, gradient_thresh=1e-5,
                 convergence_thresh=1e-5, operator_block=None,
                 primitive=None, engine=None):
        self.reference_block = reference_block
        self.hamiltonian = system_hamiltonian
        self.excitation_pool = excitation_pool
        self.optimizer = optimizer
        self.gradient_thresh = gradient_thresh
        self.engine = engine
        self._built = False

    def build(self):
        self._built = True
        return self

    def run(self, max_iter: int = 25) -> Tuple[float, Optional[List[float]]]:
        """
        Run ADAPT-VQE.

        Converts QubitOperator → IsingHamiltonian, then delegates
        to our local AdaptVQEOptimizer.

        Returns
        -------
        (energy, parameters_list) — matching QARP v0.4.3 signature.
        """
        ising = self._qubit_op_to_ising(self.hamiltonian)

        from core.qaoa_circuit import AdaptVQEOptimizer
        adapt = AdaptVQEOptimizer(
            ising, max_p=max_iter,
            gradient_threshold=self.gradient_thresh,
        )
        result = adapt.optimize()
        return (float(result["best_energy"]),
                list(result.get("best_params", [])))

    @staticmethod
    def _qubit_op_to_ising(op):
        """Convert QubitOperator back to IsingHamiltonian."""
        from core.problem_encoder import IsingHamiltonian

        h: Dict[int, float] = {}
        J: Dict[tuple, float] = {}
        offset = 0.0
        n_qubits = 0

        for term, coeff in op.terms.items():
            c = float(coeff.real if isinstance(coeff, complex) else coeff)
            if not term:
                offset += c
            elif len(term) == 1:
                q, p = term[0]
                if p == 'Z':
                    h[q] = h.get(q, 0.0) + c
                    n_qubits = max(n_qubits, q + 1)
            elif len(term) == 2:
                (q1, p1), (q2, p2) = term
                if p1 == 'Z' and p2 == 'Z':
                    key = (min(q1, q2), max(q1, q2))
                    J[key] = J.get(key, 0.0) + c
                    n_qubits = max(n_qubits, q1 + 1, q2 + 1)

        return IsingHamiltonian(
            n_qubits=n_qubits, h=h, J=J, offset=offset,
            qubit_map={}, route_ids={},
        )
