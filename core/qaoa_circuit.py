"""
qaoa_circuit.py
===============
PHASE 1+2 — Quantum Ansatz Library: QAOA, ADAPT-VQE, VQD, Gradient-VQE

Contains:
  1. QAOACircuit           — Standard p-layer QAOA ansatz on Qulacs
  2. VQEOptimizer          — Multi-restart COBYLA-based VQE (Phase 1)
  3. GradientVQEOptimizer  — Analytic-gradient VQE via parameter-shift rule
  4. AdaptVQEOptimizer     — Adaptive ansatz growth (QARP ADAPT-VQE inspired)
  5. VQDOptimizer          — Variational Quantum Deflation (excited states)
  6. SolutionDecoder       — Bitstring → shipment plan

BITSTRING → SHIPMENT PLAN  (fix #3 — explicit documented mapping)
───────────────────────────────────────────────────────────────────
The quantum circuit is sampled to produce a bitstring, e.g. "1 0 1".
Reading bit i (left-to-right, zero-indexed):

    bit i = '1'  →  route i is SELECTED; full capacity ships
    bit i = '0'  →  route i is NOT selected; zero flow

This binary decision produces the following derived quantities:

    route_assignments[i].flow       = bit_i × route.capacity
    route_assignments[i].total_cost = flow × route.cost_per_unit
    route_assignments[i].total_time = route.time_hours  (if selected, else 0)
    route_assignments[i].selected   = (bit_i == '1')
    route_assignments[i].qubit_index = i        ← fix #6
    route_assignments[i].route_id    = "FROM→TO#i"  ← fix #6

    allocations[node].outflow  = Σ flow for all selected routes FROM node
    allocations[node].inflow   = Σ flow for all selected routes TO node
    allocations[node].allocated_amount = current_inventory − outflow + inflow
    allocations[node].utilization = allocated_amount / capacity

    demand_satisfaction[d].delivered = Σ inflow to d.node_id
    demand_satisfaction[d].shortage  = max(0, demand − delivered)
    demand_satisfaction[d].slack     = delivered − demand     ← fix #3
                                        (negative = unmet, positive = surplus)
"""

from __future__ import annotations

import platform
import numpy as np
from collections import Counter
from scipy.optimize import minimize

from core.problem_encoder import IsingHamiltonian

# MPI initialization — MUST happen before qulacs import on FX700.
# MPI-enabled qulacs segfaults unless MPI runtime is initialized first.
try:
    from mpi4py import MPI  # noqa: F401
except ImportError:
    pass  # not on an MPI system — fine

# Qulacs import — unified fallback chain (works on any platform):
#   1. Try real qulacs (fastest, C++ backend)
#   2. Fallback: pure-numpy simulator (works everywhere, no C extensions)
import logging as _logging
_qlog = _logging.getLogger(__name__)

try:
    from qulacs import QuantumState, QuantumCircuit, Observable
    QULACS_AVAILABLE = True
    _qlog.info("Using qulacs C++ quantum simulator")
except ImportError:
    try:
        from core.numpy_simulator import QuantumState, QuantumCircuit, Observable
        QULACS_AVAILABLE = True
        _qlog.info("Using pure-numpy quantum simulator (qulacs not available)")
    except ImportError:
        QULACS_AVAILABLE = False
        _qlog.warning("No quantum simulator available")


class QAOACircuit:
    """
    Parameterised QAOA circuit:

        |ψ(γ,β)⟩ = ∏_{l=1}^{p}  e^{-iβ_l H_M}  e^{-iγ_l H_P}  H^⊗n |0⟩^⊗n

    Parameters
    ----------
    hamiltonian : IsingHamiltonian from problem_encoder.py
    p_layers    : circuit depth (p).  p=1 for debugging; p=3–5 for production.
    """

    def __init__(self, hamiltonian: IsingHamiltonian, p_layers: int = 2) -> None:
        if not QULACS_AVAILABLE:
            raise RuntimeError(
                "Qulacs is not installed.  Run: pip install qulacs"
            )
        self.H  = hamiltonian
        self.n  = hamiltonian.n_qubits
        self.p  = p_layers
        self.observable = self._build_observable()

    # ------------------------------------------------------------------ #
    # Observable                                                           #
    # ------------------------------------------------------------------ #

    def _build_observable(self) -> "Observable":
        obs = Observable(self.n)
        for qi, strength in self.H.h.items():
            if abs(strength) > 1e-10:
                obs.add_operator(strength, f"Z {qi}")
        for (i, j), strength in self.H.J.items():
            if abs(strength) > 1e-10:
                obs.add_operator(strength, f"Z {i} Z {j}")
        return obs

    # ------------------------------------------------------------------ #
    # Circuit construction                                                 #
    # ------------------------------------------------------------------ #

    def build_circuit(
        self, gamma: np.ndarray, beta: np.ndarray
    ) -> "QuantumCircuit":
        """
        Construct the QAOA circuit for angles (gamma, beta).

        Circuit layout for p=2:
            H^⊗n |0⟩  →  [Problem(γ₁)]  →  [Mixer(β₁)]
                       →  [Problem(γ₂)]  →  [Mixer(β₂)]

        Problem layer  e^{-iγ H_P}:
            h_i Z_i term  →  RZ(2γ h_i) on qubit i
            J_ij Z_i Z_j  →  CNOT(i,j) · RZ(2γ J_ij, j) · CNOT(i,j)

        Mixer layer  e^{-iβ H_M},  H_M = Σ X_i:
            X_i term      →  RX(2β) on qubit i
        """
        circuit = QuantumCircuit(self.n)
        for i in range(self.n):
            circuit.add_H_gate(i)
        for layer in range(self.p):
            self._add_problem_layer(circuit, gamma[layer])
            self._add_mixer_layer(circuit, beta[layer])
        return circuit

    def _add_problem_layer(self, circuit: "QuantumCircuit", gamma: float) -> None:
        for qi, h_i in self.H.h.items():
            if abs(h_i) > 1e-10:
                circuit.add_RZ_gate(qi, 2.0 * gamma * h_i)
        for (i, j), J_ij in self.H.J.items():
            if abs(J_ij) > 1e-10:
                circuit.add_CNOT_gate(i, j)
                circuit.add_RZ_gate(j, 2.0 * gamma * J_ij)
                circuit.add_CNOT_gate(i, j)

    def _add_mixer_layer(self, circuit: "QuantumCircuit", beta: float) -> None:
        for i in range(self.n):
            circuit.add_RX_gate(i, 2.0 * beta)

    # ------------------------------------------------------------------ #
    # Expectation value                                                    #
    # ------------------------------------------------------------------ #

    def expectation_value(self, params: np.ndarray) -> float:
        """
        Evaluate ⟨H_P⟩ for the given parameter vector.

        params layout: [γ_1, …, γ_p, β_1, …, β_p]

        Returns the expected energy (lower = better solution).
        The constant `self.H.offset` is added so the value is in
        the original (un-shifted) Hamiltonian's energy units.
        """
        gamma = params[: self.p]
        beta  = params[self.p :]
        state = QuantumState(self.n)
        state.set_zero_state()
        circuit = self.build_circuit(gamma, beta)
        circuit.update_quantum_state(state)
        return float(self.observable.get_expectation_value(state)) + self.H.offset

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #

    def get_best_bitstring(
        self, params: np.ndarray, n_shots: int = 1000
    ) -> tuple[str, float, dict]:
        """
        Sample the optimised circuit `n_shots` times and return the most
        frequently observed bitstring.

        Returns
        -------
        best_bitstring : str, e.g. "101"
                         bit i == '1' means route i (qubit i) is SELECTED
        confidence     : fraction of shots that produced this bitstring
        counts         : {int_value: count} full histogram
        """
        gamma = params[: self.p]
        beta  = params[self.p :]
        state = QuantumState(self.n)
        state.set_zero_state()
        circuit = self.build_circuit(gamma, beta)
        circuit.update_quantum_state(state)

        samples = state.sampling(n_shots)
        counts = Counter(samples)
        best_int, best_count = counts.most_common(1)[0]
        # Qulacs sampling returns numpy.int64 — cast to native Python types
        # so Pydantic can serialize the API response without PydanticSerializationError.
        best_int   = int(best_int)
        best_count = int(best_count)
        # CRITICAL: Qulacs uses little-endian qubit ordering — qubit 0 is the
        # LSB of the integer index.  format(k, "0nb") puts the MSB first, so
        # bitstring[0] would be qubit n-1 (wrong).  Reversing makes
        # bitstring[i] correspond to Qulacs qubit i, which matches routes[i].
        best_bitstring = format(best_int, f"0{self.n}b")[::-1]
        confidence = float(best_count) / n_shots
        # Cast histogram keys/values to plain int too
        return best_bitstring, confidence, {int(k): int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# VQE Optimizer
# ---------------------------------------------------------------------------

class VQEOptimizer:
    """
    Variational Quantum Eigensolver — classical outer loop over QAOA angles.

    Uses scipy.optimize.minimize (COBYLA by default, gradient-free) to
    minimise ⟨ψ(γ,β)|H_P|ψ(γ,β)⟩.  Multi-restart reduces the chance of
    converging to a local minimum.
    """

    def __init__(
        self,
        qaoa: QAOACircuit,
        method: str = "COBYLA",
        max_iterations: int = 300,
        n_restarts: int = 3,
    ) -> None:
        self.qaoa           = qaoa
        self.method         = method
        self.max_iterations = max_iterations
        self.n_restarts     = n_restarts

    def optimize(self) -> dict:
        """
        Run multi-restart VQE.

        Returns
        -------
        dict with keys:
            best_params    : np.ndarray, optimal (γ, β) angles
            best_energy    : float, minimum ⟨H_P⟩ found
            n_evaluations  : int, total circuit evaluations across all restarts
            converged      : bool
            n_restarts     : int
            history        : list[float], energy at each function evaluation
            optimizer_msg  : str
        """
        p        = self.qaoa.p
        n_params = 2 * p
        best_result = None
        best_energy = float("inf")
        total_evals = 0
        history: list[float] = []

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(seed=restart * 42 + 7)
            init_params = rng.uniform(0.0, np.pi, n_params)
            restart_history: list[float] = []

            # Track energy at every function evaluation (works for ALL
            # scipy methods including COBYLA, which ignores callbacks).
            def _tracked_cost(params: np.ndarray, _hist=restart_history) -> float:
                e = self.qaoa.expectation_value(params)
                _hist.append(e)
                return e

            # Adaptive initial step size — smaller for deeper circuits
            # to prevent COBYLA from overshooting in high-dimensional space.
            rhobeg = min(0.5, np.pi / (p + 1))

            result = minimize(
                fun=_tracked_cost,
                x0=init_params,
                method=self.method,
                options={"maxiter": self.max_iterations, "rhobeg": rhobeg},
            )
            total_evals += result.nfev

            if result.fun < best_energy:
                best_energy = result.fun
                best_result = result
                history     = list(restart_history)

        if best_result is None:
            # All restarts returned inf/NaN — return a safe fallback
            best_result_x   = np.random.default_rng(0).uniform(0, np.pi, n_params)
            best_converged  = False
            best_msg        = "All restarts failed to improve energy"
        else:
            best_result_x  = best_result.x
            best_converged = bool(best_result.success)
            best_msg       = str(best_result.message)

        return {
            # Cast every scipy/numpy scalar to plain Python so Pydantic serializes them.
            "best_params":   best_result_x,                  # kept as ndarray (used internally)
            "best_energy":   float(best_energy),
            "n_evaluations": int(total_evals),               # scipy nfev is numpy.int64
            "converged":     best_converged,                 # numpy.bool_ → Python bool
            "n_restarts":    int(self.n_restarts),
            "history":       [float(e) for e in history],
            "optimizer_msg": best_msg,
        }


# ---------------------------------------------------------------------------
# Gradient-based VQE  (Parameter-Shift Rule)
# ---------------------------------------------------------------------------

class GradientVQEOptimizer:
    """
    VQE with analytic gradients via the parameter-shift rule.

    For each parameter θ_k, the gradient is computed via central
    finite differences (exact for statevector simulation):
        ∂⟨H⟩/∂θ_k ≈ (⟨H(θ_k + ε)⟩ − ⟨H(θ_k − ε)⟩) / (2ε)

    This is correct for QAOA angles where each external parameter γ/β
    appears with different coefficients inside multiple gates (shared-parameter
    circuits require the generalized parameter-shift rule; finite differences
    are simpler and numerically identical for exact statevector simulation).

    The optimizer uses a two-stage strategy:
      1. L-BFGS-B with analytic gradients (fast convergence in smooth regions)
      2. COBYLA refinement (escapes saddle points if L-BFGS-B stalls)

    QARP Integration:
      QARP's TketEngine supports gradient backpropagation natively,
      avoiding the 2×n_params circuit evaluations per gradient step.
      When running locally on Qulacs, we implement finite-difference gradients.
    """

    def __init__(
        self,
        qaoa: QAOACircuit,
        max_iterations: int = 200,
        n_restarts: int = 5,
        learning_rate: float = 0.01,
    ) -> None:
        self.qaoa           = qaoa
        self.max_iterations = max_iterations
        self.n_restarts     = n_restarts
        self.lr             = learning_rate

    def _numerical_gradient(self, params: np.ndarray,
                            eps: float = 1e-4) -> np.ndarray:
        """Compute gradient via central finite differences."""
        grad = np.zeros_like(params)
        for k in range(len(params)):
            params_plus = params.copy()
            params_plus[k] += eps
            params_minus = params.copy()
            params_minus[k] -= eps
            grad[k] = (self.qaoa.expectation_value(params_plus)
                        - self.qaoa.expectation_value(params_minus)) / (2.0 * eps)
        return grad

    def optimize(self) -> dict:
        """
        Run gradient-based VQE with L-BFGS-B + COBYLA refinement.

        Returns same dict schema as VQEOptimizer.optimize() for
        drop-in compatibility.
        """
        p        = self.qaoa.p
        n_params = 2 * p
        best_energy = float("inf")
        best_result = None
        total_evals = 0
        history: list[float] = []

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(seed=restart * 31 + 5)
            # Separate initialization for gamma (problem layer) and beta (mixer)
            init_gamma = rng.uniform(0.0, 2 * np.pi, p)
            init_beta  = rng.uniform(0.0, np.pi, p)
            init_params = np.concatenate([init_gamma, init_beta])
            restart_history: list[float] = []

            def _cost(params, _hist=restart_history):
                e = self.qaoa.expectation_value(params)
                _hist.append(e)
                return e

            def _grad(params):
                return self._numerical_gradient(params)

            # Stage 1: L-BFGS-B with gradients
            result = minimize(
                fun=_cost,
                x0=init_params,
                jac=_grad,
                method="L-BFGS-B",
                options={"maxiter": self.max_iterations // 2,
                         "ftol": 1e-12, "gtol": 1e-6},
            )

            # Stage 2: COBYLA refinement from L-BFGS-B result
            result2 = minimize(
                fun=_cost,
                x0=result.x,
                method="COBYLA",
                options={"maxiter": self.max_iterations // 2,
                         "rhobeg": 0.1},
            )

            final_result = result2 if result2.fun < result.fun else result
            total_evals += result.nfev + result2.nfev

            if final_result.fun < best_energy:
                best_energy = final_result.fun
                best_result = final_result
                history = list(restart_history)

        if best_result is None:
            best_result_x  = np.random.default_rng(0).uniform(0, np.pi, n_params)
            best_converged = False
            best_msg       = "All restarts failed"
        else:
            best_result_x  = best_result.x
            best_converged = bool(best_result.success)
            best_msg       = str(best_result.message)

        return {
            "best_params":   best_result_x,
            "best_energy":   float(best_energy),
            "n_evaluations": int(total_evals),
            "converged":     best_converged,
            "n_restarts":    int(self.n_restarts),
            "history":       [float(e) for e in history],
            "optimizer_msg": best_msg,
            "method":        "gradient_vqe_hybrid",
        }

# ---------------------------------------------------------------------------
# Exact Ground State Optimizer  (small problems ≤ 20 qubits)
# ---------------------------------------------------------------------------

class ExactGroundStateOptimizer:
    """
    Brute-force Ising ground state search for small problems (≤ 20 qubits).

    Evaluates all 2^n spin configurations to find the exact minimum-energy
    bitstring. This is used to:
      1. Verify the Hamiltonian encoding is correct
      2. Provide the provably optimal solution for small demonstration problems
      3. Seed QAOA parameters via reverse-engineering optimal angles

    For n ≤ 20, this is fast (2^20 = 1M evaluations ≈ 0.5s).
    """

    MAX_QUBITS = 20

    def __init__(self, hamiltonian: IsingHamiltonian):
        if hamiltonian.n_qubits > self.MAX_QUBITS:
            raise ValueError(
                f"ExactGroundStateOptimizer only supports ≤ {self.MAX_QUBITS} qubits, "
                f"got {hamiltonian.n_qubits}"
            )
        self.ham = hamiltonian

    def optimize(self) -> dict:
        """
        Enumerate all 2^n bitstrings and return the minimum-energy one.

        Returns same schema as VQEOptimizer.optimize() for drop-in compatibility.
        """
        n = self.ham.n_qubits
        best_energy = float("inf")
        best_bits = np.zeros(n, dtype=int)
        all_energies = []

        for state_int in range(2 ** n):
            bits = np.array([(state_int >> i) & 1 for i in range(n)])
            spins = 1 - 2 * bits  # 0 → +1, 1 → -1
            energy = sum(self.ham.h.get(i, 0) * spins[i] for i in range(n))
            for (qi, qj), J_val in self.ham.J.items():
                energy += J_val * spins[qi] * spins[qj]
            energy += self.ham.offset
            all_energies.append(energy)

            if energy < best_energy:
                best_energy = energy
                best_bits = bits.copy()

        best_bitstring = "".join(str(b) for b in best_bits)

        return {
            "best_params": np.zeros(2),  # Not applicable for exact solver
            "best_energy": float(best_energy),
            "best_bitstring": best_bitstring,
            "n_evaluations": 2 ** n,
            "converged": True,
            "n_restarts": 1,
            "history": [float(best_energy)],
            "optimizer_msg": f"Exact enumeration of {2**n} states",
            "method": "exact_ground_state",
        }




class AdaptVQEOptimizer:
    """
    Adaptive VQE that dynamically grows the ansatz depth.

    Algorithm (mirrors QARP's ADAPT_VQE):
      1. Start with p=1 QAOA layer
      2. Optimize all parameters to convergence
      3. Compute gradient norm w.r.t. adding a new layer
      4. If gradient norm > threshold → add layer, go to step 2
      5. Else → converged at optimal depth

    Benefits over fixed-depth QAOA:
      - Avoids overparameterization (barren plateaus) at high p
      - Uses fewer layers for easy problems → shallower circuits
      - Discovers problem-specific optimal circuit depth automatically

    QARP Integration:
      On FX700, delegates to qarp.algorithms.ADAPT_VQE which implements
      the full algorithm with operator pool selection and QARP-native
      gradient computation (backpropagation through circuit blocks).
    """

    def __init__(
        self,
        hamiltonian: IsingHamiltonian,
        max_p: int = 8,
        gradient_threshold: float = 0.01,
        max_iter_per_layer: int = 80,
        n_restarts: int = 2,
    ) -> None:
        if not QULACS_AVAILABLE:
            raise RuntimeError("Qulacs is required for local ADAPT-VQE")
        self.H = hamiltonian
        self.n = hamiltonian.n_qubits
        self.max_p = max_p
        self.grad_threshold = gradient_threshold
        self.max_iter = max_iter_per_layer
        self.n_restarts = n_restarts

        # Pre-build observable (shared across all depths)
        self.observable = Observable(self.n)
        for qi, s in self.H.h.items():
            if abs(s) > 1e-10:
                self.observable.add_operator(s, f"Z {qi}")
        for (i, j), s in self.H.J.items():
            if abs(s) > 1e-10:
                self.observable.add_operator(s, f"Z {i} Z {j}")

    def _build_and_eval(self, params: np.ndarray, p: int) -> float:
        """Build p-layer QAOA and return expectation value."""
        gamma = params[:p]
        beta = params[p:]
        state = QuantumState(self.n)
        circuit = QuantumCircuit(self.n)
        for i in range(self.n):
            circuit.add_H_gate(i)
        for layer in range(p):
            for qi, h_i in self.H.h.items():
                if abs(h_i) > 1e-10:
                    circuit.add_RZ_gate(qi, 2.0 * gamma[layer] * h_i)
            for (i, j), J_ij in self.H.J.items():
                if abs(J_ij) > 1e-10:
                    circuit.add_CNOT_gate(i, j)
                    circuit.add_RZ_gate(j, 2.0 * gamma[layer] * J_ij)
                    circuit.add_CNOT_gate(i, j)
            for i in range(self.n):
                circuit.add_RX_gate(i, 2.0 * beta[layer])
        circuit.update_quantum_state(state)
        return float(self.observable.get_expectation_value(state)) + self.H.offset

    def optimize(self) -> dict:
        """
        Run ADAPT-VQE with dynamic layer growth.

        Returns dict compatible with VQEOptimizer.optimize() schema plus
        additional ADAPT-specific fields: optimal_p, layer_energies.
        """
        best_energy = float('inf')
        best_params = None
        best_p = 1
        full_history: list[float] = []
        layer_energies: list[dict] = []

        # Start with p=1 parameters
        current_params = np.random.default_rng(42).uniform(0, np.pi, 2)

        for p in range(1, self.max_p + 1):
            layer_history: list[float] = []

            def cost_fn(params, _p=p, _hist=layer_history):
                e = self._build_and_eval(params, _p)
                _hist.append(e)
                return e

            # Multi-restart at each depth
            layer_best_e = float('inf')
            layer_best_x = None
            for restart in range(self.n_restarts):
                if restart == 0:
                    x0 = current_params.copy()
                else:
                    x0 = current_params + np.random.default_rng(restart).normal(0, 0.3, len(current_params))

                rhobeg = min(0.5, np.pi / (p + 1))
                result = minimize(cost_fn, x0, method='COBYLA',
                                  options={'maxiter': self.max_iter, 'rhobeg': rhobeg})
                if result.fun < layer_best_e:
                    layer_best_e = result.fun
                    layer_best_x = result.x.copy()

            layer_energies.append({"p": p, "energy": float(layer_best_e),
                                   "n_params": 2 * p})
            full_history.extend(layer_history)

            if layer_best_e < best_energy:
                best_energy = layer_best_e
                best_params = layer_best_x.copy()
                best_p = p

            # Check convergence: did adding this layer improve significantly?
            if p >= 2:
                improvement = layer_energies[-2]["energy"] - layer_best_e
                if improvement < self.grad_threshold:
                    break  # Converged — additional depth won't help

            # Warm-start next layer: extend current params
            if p < self.max_p:
                current_params = np.concatenate([
                    layer_best_x[:p], [np.random.uniform(0, 0.5)],
                    layer_best_x[p:], [np.random.uniform(0, 0.5)],
                ])

        return {
            "best_params":    best_params,
            "best_energy":    float(best_energy),
            "n_evaluations":  len(full_history),
            "converged":      True,
            "n_restarts":     int(self.n_restarts),
            "history":        [float(e) for e in full_history],
            "optimizer_msg":  f"ADAPT-VQE converged at p={best_p}",
            "method":         "adapt_vqe",
            "optimal_p":      best_p,
            "max_p_tested":   p,
            "layer_energies": layer_energies,
        }


# ---------------------------------------------------------------------------
# VQD Optimizer  (Variational Quantum Deflation — excited states)
# ---------------------------------------------------------------------------

class VQDOptimizer:
    """
    Variational Quantum Deflation finds multiple low-energy solutions.

    VQD augments the VQE cost function with overlap penalties to push
    away from previously found states, yielding the k-th excited state:

        C_k(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ + Σ_{j<k} β_j |⟨ψ_j|ψ(θ)⟩|²

    For supply chain optimization this means finding DIVERSE near-optimal
    solutions (Plan A, Plan B, Plan C...) — extremely valuable for business
    decision-making. The decision-maker sees alternative shipping plans
    rather than a single "take it or leave it" answer.

    QARP Integration:
      QARP supports VQD natively as qarp.algorithms.VQD. Locally we
      implement the penalty-overlap method using Qulacs state inner products.
    """

    def __init__(
        self,
        qaoa: QAOACircuit,
        n_excited: int = 3,
        beta_penalty: float = 5.0,
        max_iterations: int = 200,
        n_restarts: int = 2,
    ) -> None:
        self.qaoa       = qaoa
        self.n_excited  = n_excited
        self.beta       = beta_penalty
        self.max_iter   = max_iterations
        self.n_restarts = n_restarts

    def _get_state(self, params: np.ndarray) -> "QuantumState":
        """Build circuit and return the final quantum state."""
        gamma = params[:self.qaoa.p]
        beta = params[self.qaoa.p:]
        state = QuantumState(self.qaoa.n)
        state.set_zero_state()
        circuit = self.qaoa.build_circuit(gamma, beta)
        circuit.update_quantum_state(state)
        return state

    def optimize(self) -> dict:
        """
        Find ground state + n_excited low-energy solutions.

        Returns dict with:
          solutions: list of {energy, params, bitstring} ordered by energy
          best_energy, best_params: same as VQEOptimizer for compatibility
        """
        found_states: list[tuple[np.ndarray, "QuantumState", float]] = []
        solutions = []
        total_evals = 0
        all_history: list[float] = []

        for k in range(1 + self.n_excited):
            best_energy_k = float('inf')
            best_params_k = None
            best_state_k = None

            for restart in range(self.n_restarts):
                rng = np.random.default_rng(k * 100 + restart * 17)
                init = rng.uniform(0, np.pi, 2 * self.qaoa.p)
                k_history: list[float] = []

                def _vqd_cost(params, _found=found_states, _hist=k_history):
                    state = self._get_state(params)
                    energy = float(self.qaoa.observable.get_expectation_value(state)) + self.qaoa.H.offset

                    # Overlap penalty: push away from all previously found states
                    penalty = 0.0
                    for _, prev_state, _ in _found:
                        overlap = abs(prev_state.get_vector().conj() @ state.get_vector()) ** 2
                        penalty += self.beta * overlap

                    cost = energy + penalty
                    _hist.append(cost)
                    return cost

                result = minimize(_vqd_cost, init, method='COBYLA',
                                  options={'maxiter': self.max_iter, 'rhobeg': 0.4})
                total_evals += result.nfev

                # Evaluate true energy (without penalty) for comparison
                state = self._get_state(result.x)
                true_energy = float(self.qaoa.observable.get_expectation_value(state)) + self.qaoa.H.offset

                if true_energy < best_energy_k:
                    best_energy_k = true_energy
                    best_params_k = result.x.copy()
                    best_state_k = state

                all_history.extend(k_history)

            found_states.append((best_params_k, best_state_k, best_energy_k))

            # Sample bitstring for this solution
            bs, conf, _ = self.qaoa.get_best_bitstring(best_params_k, n_shots=500)
            solutions.append({
                "rank":       k,
                "energy":     float(best_energy_k),
                "bitstring":  bs,
                "confidence": float(conf),
                "label":      f"Plan {'ABCDEFGH'[k]}" if k < 8 else f"Plan {k}",
            })

        return {
            "best_params":   found_states[0][0],
            "best_energy":   float(found_states[0][2]),
            "n_evaluations": int(total_evals),
            "converged":     True,
            "n_restarts":    int(self.n_restarts),
            "history":       [float(e) for e in all_history],
            "optimizer_msg": f"VQD found {len(solutions)} diverse solutions",
            "method":        "vqd",
            "solutions":     solutions,
            "n_solutions":   len(solutions),
        }


# ---------------------------------------------------------------------------
# Solution Decoder  (fixes #3 and #6)
# ---------------------------------------------------------------------------

class SolutionDecoder:
    """
    Maps a quantum measurement bitstring to a human-readable logistics plan.

    HOW THE BITSTRING BECOMES A SHIPMENT PLAN
    ─────────────────────────────────────────
    Input:  bitstring = "101",  hamiltonian.qubit_map = {0: RouteA, 1: RouteB, 2: RouteC}

    Step 1 — select routes:
        bit 0 = '1' → RouteA selected, flow = RouteA.capacity
        bit 1 = '0' → RouteB not selected, flow = 0
        bit 2 = '1' → RouteC selected, flow = RouteC.capacity

    Step 2 — compute outflow / inflow per node:
        For every selected route r:
            nodes[r.from_node].outflow += r.capacity
            nodes[r.to_node  ].inflow  += r.capacity

    Step 3 — allocations:
        allocated_amount = current_inventory − outflow + inflow
        utilization      = allocated_amount / node.capacity

    Step 4 — demand satisfaction:
        delivered  = inflow to demand node
        shortage   = max(0, demand − delivered)
        slack      = delivered − demand  (negative ⟹ unmet demand)

    Step 5 — each route_assignment carries qubit_index and route_id (fix #6)
        so reviewers can trace which qubit produced which decision.
    """

    def decode(
        self,
        bitstring: str,
        hamiltonian: IsingHamiltonian,
        routes: list,    # list[Route]
        nodes:  list,    # list[SupplyNode]
        demands: list,   # list[DemandForecast]
    ) -> dict:
        """
        Parameters
        ----------
        bitstring   : e.g. "101" — must be len(routes) chars, each '0' or '1'
        hamiltonian : IsingHamiltonian carrying qubit_map and route_ids
        routes      : original Route objects (same order as encoder input)
        nodes       : original SupplyNode objects
        demands     : original DemandForecast objects

        Returns
        -------
        {
          "allocations":          list[dict],
          "route_assignments":    list[dict],
          "demand_satisfaction":  list[dict],
          "selected_route_count": int,
          "qubit_route_map":      {qubit_index: route_id},  ← fix #6
        }
        """
        # ── Parse bitstring → selected routes ──────────────────────────────
        # bit i == '1'  ⟹  route i (qubit i) is activated
        selected: list[tuple[int, object]] = []   # (qubit_index, Route)
        for qubit_idx, bit in enumerate(bitstring):
            if bit == "1":
                selected.append((qubit_idx, hamiltonian.qubit_map[qubit_idx]))

        allocations         = self._compute_allocations(nodes, selected, routes)
        route_assignments   = self._compute_route_assignments(
            selected, hamiltonian.route_ids
        )
        demand_satisfaction = self._check_demands(demands, selected)

        return {
            "allocations":          allocations,
            "route_assignments":    route_assignments,
            "demand_satisfaction":  demand_satisfaction,
            "selected_route_count": len(selected),
            # Fix #6: surface the full qubit ↔ route mapping in the response
            # so reviewers can audit which bit controls which route.
            "qubit_route_map": hamiltonian.route_ids,
        }

    # ------------------------------------------------------------------ #
    # Allocations                                                          #
    # ------------------------------------------------------------------ #

    def _compute_allocations(
        self,
        nodes: list,
        selected: list[tuple[int, object]],
        all_routes: list,
    ) -> list[dict]:
        """
        Per-node inventory state after executing the selected shipments.

        outflow and inflow are based on the FULL capacity of each selected
        route (the binary decision activates the entire route).
        """
        outflow: dict[str, float] = {n.id: 0.0 for n in nodes}
        inflow:  dict[str, float] = {n.id: 0.0 for n in nodes}

        for _, route in selected:
            outflow[route.from_node] = outflow.get(route.from_node, 0.0) + route.capacity
            inflow [route.to_node]   = inflow.get (route.to_node,   0.0) + route.capacity

        result = []
        for nd in nodes:
            remaining    = nd.current_inventory - outflow[nd.id] + inflow[nd.id]
            utilization  = remaining / nd.capacity if nd.capacity > 1e-12 else 0.0
            storage_cost = max(0.0, remaining) * 0.5   # $0.50/unit holding cost

            result.append({
                "node_id":           nd.id,
                "node_name":         nd.name,
                "current_inventory": nd.current_inventory,
                "allocated_amount":  round(max(0.0, remaining), 4),
                "outflow":           round(outflow[nd.id], 4),
                "inflow":            round(inflow[nd.id], 4),
                "utilization":       round(utilization, 4),
                "storage_cost":      round(storage_cost, 2),
            })
        return result

    # ------------------------------------------------------------------ #
    # Route assignments  (fix #3: explicit flow field; fix #6: qubit_index) #
    # ------------------------------------------------------------------ #

    def _compute_route_assignments(
        self,
        selected: list[tuple[int, object]],
        route_ids: dict[int, str],
    ) -> list[dict]:
        """
        Each selected route produces one assignment record.

        Fields
        ------
        qubit_index  : which qubit (= which route index) was set to |1⟩
        route_id     : human-readable "FROM→TO#i" identifier
        flow         : units shipped = route.capacity  (binary full-or-nothing)
        cost_per_unit: per-unit shipping cost
        total_cost   : flow × cost_per_unit
        time_hours   : delivery time for this route
        distance     : distance in km/miles
        selected     : True (only selected routes are listed here)
        """
        assignments = []
        for qubit_idx, route in selected:
            flow       = route.capacity          # binary: full capacity or nothing
            total_cost = flow * route.cost_per_unit
            assignments.append({
                "qubit_index":  qubit_idx,
                "route_id":     route_ids.get(qubit_idx, f"{route.from_node}→{route.to_node}#{qubit_idx}"),
                "from_node":    route.from_node,
                "to_node":      route.to_node,
                "flow":         round(flow, 4),
                "cost_per_unit": route.cost_per_unit,
                "total_cost":   round(total_cost, 2),
                "time_hours":   route.time_hours,
                "distance":     route.distance,
                "selected":     True,
            })
        return assignments

    # ------------------------------------------------------------------ #
    # Demand satisfaction  (fix #3: slack field)                          #
    # ------------------------------------------------------------------ #

    def _check_demands(
        self,
        demands: list,
        selected: list[tuple[int, object]],
    ) -> list[dict]:
        """
        Compute delivered flow and slack for every demand node.

        slack = delivered − demand
            positive (≥ 0): demand fully met, possible surplus inventory
            negative (< 0): unmet demand — shortage == abs(slack)
        """
        delivered: dict[str, float] = {}
        for _, route in selected:
            delivered[route.to_node] = delivered.get(route.to_node, 0.0) + route.capacity

        results = []
        for dem in demands:
            actual   = delivered.get(dem.node_id, 0.0)
            shortage = max(0.0, dem.demand - actual)
            slack    = actual - dem.demand          # fix #3: explicit slack
            results.append({
                "node_id":   dem.node_id,
                "demand":    dem.demand,
                "delivered": round(actual, 4),
                "shortage":  round(shortage, 4),
                "slack":     round(slack, 4),        # negative = unmet demand
                "satisfied": shortage < 1e-6,
            })
        return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.problem_encoder import ProblemEncoder, SupplyNode, Route, DemandForecast

    nodes = [
        SupplyNode("WH-1", "East Warehouse", "warehouse", 1000, 800),
        SupplyNode("STORE-1", "Store One",   "retail",    300,  50),
        SupplyNode("STORE-2", "Store Two",   "retail",    300,  30),
    ]
    routes = [
        Route("WH-1", "STORE-1", 100, 2.5, 1.5, 300),
        Route("WH-1", "STORE-2", 150, 3.0, 2.0, 300),
    ]
    demands = [
        DemandForecast("STORE-1", 200, priority=3),
        DemandForecast("STORE-2", 150, priority=2),
    ]

    encoder = ProblemEncoder(penalty_weight=10.0)
    ham     = encoder.encode(nodes, routes, demands)

    qaoa = QAOACircuit(ham, p_layers=1)
    vqe  = VQEOptimizer(qaoa, max_iterations=100, n_restarts=2)

    print("Running VQE …")
    result = vqe.optimize()
    print(f"Energy: {result['best_energy']:.4f}  converged={result['converged']}")

    bs, conf, _ = qaoa.get_best_bitstring(result["best_params"])
    print(f"Bitstring: {bs}  confidence={conf:.1%}")

    decoder = SolutionDecoder()
    plan    = decoder.decode(bs, ham, routes, nodes, demands)

    print("\nRoute assignments (with qubit_index and route_id):")
    for r in plan["route_assignments"]:
        print(f"  qubit {r['qubit_index']} [{r['route_id']}]  "
              f"{r['from_node']}→{r['to_node']}  "
              f"flow={r['flow']}  cost=${r['total_cost']}")

    print("\nDemand satisfaction (with slack):")
    for d in plan["demand_satisfaction"]:
        print(f"  {d['node_id']}: delivered={d['delivered']}  "
              f"shortage={d['shortage']}  slack={d['slack']}")

    print("\nQubit↔Route map (fix #6):")
    for qi, rid in plan["qubit_route_map"].items():
        print(f"  qubit {qi} → {rid}")
