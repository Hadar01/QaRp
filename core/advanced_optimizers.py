"""
advanced_optimizers.py
======================
State-of-the-art QAOA optimizer variants for demonstrating genuine
quantum advantage on supply chain optimization problems.

Contains:
  1. CVaRQAOAOptimizer      — Conditional Value at Risk (tail-risk) optimization
  2. LayerByLayerOptimizer  — Incremental circuit depth with cyclic refinement
  3. RecursiveQAOA (RQAOA)  — Correlation-based recursive variable elimination

References:
  [1] Barkoutsos et al., "Improving Variational Quantum Optimization using
      CVaR", Quantum 4, 256 (2020)
  [2] Skolik et al., "Layerwise learning for quantum neural networks",
      Quantum Machine Intelligence 3, 5 (2021)
  [3] Bravyi et al., "Obstacles to Variational Quantum Optimization from
      Symmetry Protection", Physical Review Letters 125, 260505 (2020)
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import minimize

from core.problem_encoder import IsingHamiltonian
from core.qaoa_circuit import (
    QAOACircuit, VQEOptimizer, ExactGroundStateOptimizer,
)

# MPI initialization — MUST happen before qulacs import on FX700
try:
    from mpi4py import MPI  # noqa: F401
except ImportError:
    pass

# Use the same Qulacs / numpy-simulator fallback chain
try:
    from qulacs import QuantumState, QuantumCircuit, Observable
except ImportError:
    from core.numpy_simulator import QuantumState, QuantumCircuit, Observable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Evaluate Ising energy from a bit array
# ═══════════════════════════════════════════════════════════════════════════

def ising_energy(ham: IsingHamiltonian, bits) -> float:
    """
    Compute the Ising energy for a given bit assignment.

    Parameters
    ----------
    ham  : IsingHamiltonian with h, J, offset
    bits : array-like of 0/1 values, length n_qubits
           bit[i] = 1 means qubit i is in state |1⟩ (spin -1)

    Returns
    -------
    float : Ising energy  H = Σ h_i s_i + Σ J_ij s_i s_j + offset
    """
    spins = np.array([1 - 2 * int(b) for b in bits])
    energy = sum(h_val * spins[i] for i, h_val in ham.h.items())
    for (qi, qj), J_val in ham.J.items():
        energy += J_val * spins[qi] * spins[qj]
    return float(energy + ham.offset)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  CVaR-QAOA Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class CVaRQAOAOptimizer:
    """
    CVaR-QAOA: replace the standard ⟨H⟩ cost with Conditional Value at Risk.

    Instead of minimizing the mean energy over all measurement outcomes,
    CVaR only averages the **α-best (lowest-energy) fraction** of samples.
    This steers the optimizer toward the global minimum rather than wasting
    gradient information on poor bitstrings.

    Uses **ascending-CVaR**: α starts broad (0.5) for landscape exploration,
    then tightens (→ 0.1) for exploitation — inspired by simulated annealing.

    Reference: Barkoutsos et al., Quantum 4, 256 (2020)
    """

    def __init__(
        self,
        qaoa: QAOACircuit,
        alpha_start: float = 0.5,
        alpha_end: float = 0.1,
        n_shots: int = 2048,
        max_iterations: int = 400,
        n_restarts: int = 8,
    ):
        self.qaoa = qaoa
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.n_shots = n_shots
        self.max_iterations = max_iterations
        self.n_restarts = n_restarts

    def _cvar_cost(self, params: np.ndarray, eval_count: list) -> float:
        """
        Build QAOA state, sample bitstrings, compute CVaR of Ising energies.

        The α parameter is linearly annealed from alpha_start → alpha_end
        as eval_count grows, implementing the ascending-CVaR schedule.
        """
        qaoa = self.qaoa
        gamma = params[: qaoa.p]
        beta = params[qaoa.p :]

        # Build and run QAOA circuit
        state = QuantumState(qaoa.n)
        state.set_zero_state()
        circuit = qaoa.build_circuit(gamma, beta)
        circuit.update_quantum_state(state)

        # Sample bitstrings
        samples = state.sampling(self.n_shots)

        # Evaluate Ising energy for each sample
        energies = np.empty(len(samples))
        ham = qaoa.H
        for idx, sample_int in enumerate(samples):
            bits = [(sample_int >> i) & 1 for i in range(qaoa.n)]
            spins = np.array([1 - 2 * b for b in bits])
            e = sum(ham.h.get(i, 0) * spins[i] for i in range(qaoa.n))
            for (qi, qj), J_val in ham.J.items():
                e += J_val * spins[qi] * spins[qj]
            energies[idx] = e + ham.offset

        # Ascending CVaR schedule: α shrinks as optimisation proceeds
        eval_count[0] += 1
        progress = min(1.0, eval_count[0] / (self.max_iterations * 0.7))
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress

        # CVaR = mean of the α-lowest energies
        energies.sort()
        k = max(1, int(alpha * len(energies)))
        return float(np.mean(energies[:k]))

    def optimize(self) -> dict:
        """
        Multi-restart CVaR-QAOA optimization.

        Returns dict compatible with VQEOptimizer.optimize() output schema.
        """
        p = self.qaoa.p
        n_params = 2 * p
        best_energy = float("inf")
        best_params = None
        total_evals = 0
        best_history = []

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(seed=restart * 37 + 7)
            init_gamma = rng.uniform(0.0, 2 * np.pi, p)
            init_beta = rng.uniform(0.0, np.pi, p)
            init_params = np.concatenate([init_gamma, init_beta])

            eval_count = [0]   # mutable counter for the annealing schedule
            history = []

            def _cost(params, _hist=history, _ec=eval_count):
                e = self._cvar_cost(params, _ec)
                _hist.append(e)
                return e

            rhobeg = min(0.5, np.pi / (p + 1))
            result = minimize(
                fun=_cost,
                x0=init_params,
                method="COBYLA",
                options={"maxiter": self.max_iterations, "rhobeg": rhobeg},
            )
            total_evals += result.nfev

            # Evaluate final energy using the **exact** expectation value
            # (not CVaR) so we get a fair comparison metric
            final_energy = self.qaoa.expectation_value(result.x)

            if final_energy < best_energy:
                best_energy = final_energy
                best_params = result.x.copy()
                best_history = list(history)

        return {
            "best_params":   best_params if best_params is not None else np.zeros(n_params),
            "best_energy":   float(best_energy),
            "n_evaluations": int(total_evals),
            "converged":     best_energy < float("inf"),
            "n_restarts":    self.n_restarts,
            "history":       [float(e) for e in best_history],
            "optimizer_msg": f"CVaR-QAOA α={self.alpha_start}→{self.alpha_end}, "
                             f"{self.n_restarts} restarts",
            "method":        "cvar_qaoa",
        }


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Layer-by-Layer Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class LayerByLayerOptimizer:
    """
    Layer-by-layer QAOA with cyclic refinement.

    Algorithm:
      1. Start with p=1: optimise γ₁, β₁  (2 parameters → trivial landscape)
      2. Extend to p=2: initialise [γ₁_opt, ε, β₁_opt, ε] and optimise all 4
      3. Continue building depth up to target_p
      4. Final refinement: COBYLA over all 2p params with tight tolerance

    New layers are initialised near zero so they start as an identity
    perturbation to the already-optimised shallower circuit.

    Reference: Skolik et al., Quantum Machine Intelligence 3, 5 (2021)
    """

    def __init__(
        self,
        hamiltonian: IsingHamiltonian,
        target_p: int = 6,
        max_iter_per_layer: int = 150,
        max_iter_refinement: int = 300,
        n_restarts_first: int = 5,
    ):
        self.ham = hamiltonian
        self.target_p = target_p
        self.max_iter_per_layer = max_iter_per_layer
        self.max_iter_refinement = max_iter_refinement
        self.n_restarts_first = n_restarts_first

    def optimize(self) -> dict:
        """
        Build depth incrementally, then refine.

        Returns dict compatible with VQEOptimizer.optimize() output schema.
        """
        total_evals = 0
        layer_energies = []

        # ── Phase 1: Build depth layer by layer ──────────────────────────
        best_gamma = np.array([], dtype=float)
        best_beta = np.array([], dtype=float)

        for p in range(1, self.target_p + 1):
            qaoa = QAOACircuit(self.ham, p_layers=p)

            if p == 1:
                # First layer: multi-restart search over 2 parameters
                best_e = float("inf")
                best_x = None
                for restart in range(self.n_restarts_first):
                    rng = np.random.default_rng(seed=restart * 53 + 11)
                    x0 = np.array([
                        rng.uniform(0, 2 * np.pi),
                        rng.uniform(0, np.pi),
                    ])
                    res = minimize(
                        fun=qaoa.expectation_value,
                        x0=x0,
                        method="COBYLA",
                        options={"maxiter": self.max_iter_per_layer,
                                 "rhobeg": 0.5},
                    )
                    total_evals += res.nfev
                    if res.fun < best_e:
                        best_e = res.fun
                        best_x = res.x.copy()
                best_gamma = best_x[:1]
                best_beta = best_x[1:]
            else:
                # Deeper layers: initialise new params near zero (identity)
                init_gamma = np.concatenate([best_gamma, [0.01]])
                init_beta = np.concatenate([best_beta, [0.01]])
                init_params = np.concatenate([init_gamma, init_beta])

                rhobeg = min(0.4, np.pi / (p + 1))
                res = minimize(
                    fun=qaoa.expectation_value,
                    x0=init_params,
                    method="COBYLA",
                    options={"maxiter": self.max_iter_per_layer,
                             "rhobeg": rhobeg},
                )
                total_evals += res.nfev

                best_gamma = res.x[:p]
                best_beta = res.x[p:]

            current_energy = qaoa.expectation_value(
                np.concatenate([best_gamma, best_beta])
            )
            layer_energies.append(float(current_energy))
            logger.debug(f"Layer p={p}: energy={current_energy:.4f}")

        # ── Phase 2: Full-parameter refinement with tight tolerance ──────
        qaoa_final = QAOACircuit(self.ham, p_layers=self.target_p)
        final_init = np.concatenate([best_gamma, best_beta])

        res_refine = minimize(
            fun=qaoa_final.expectation_value,
            x0=final_init,
            method="COBYLA",
            options={"maxiter": self.max_iter_refinement,
                     "rhobeg": 0.15, "catol": 1e-10},
        )
        total_evals += res_refine.nfev

        # Keep whichever is better: layered result or refined result
        if res_refine.fun < layer_energies[-1]:
            final_params = res_refine.x
            final_energy = res_refine.fun
        else:
            final_params = final_init
            final_energy = layer_energies[-1]

        return {
            "best_params":    final_params,
            "best_energy":    float(final_energy),
            "n_evaluations":  int(total_evals),
            "converged":      True,
            "n_restarts":     1,
            "history":        [float(e) for e in layer_energies],
            "optimizer_msg":  f"Layer-by-layer p=1→{self.target_p}, then refinement",
            "method":         "layer_by_layer",
            "layer_energies": [float(e) for e in layer_energies],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Recursive QAOA (RQAOA)
# ═══════════════════════════════════════════════════════════════════════════

def _reduce_hamiltonian_pair(
    h: dict, J: dict, offset: float,
    active: list[int],
    eliminate: int, reference: int, sign: int,
) -> tuple[dict, dict, float, list[int]]:
    """
    Reduce Hamiltonian by one variable: fix z_{eliminate} = sign · z_{reference}.

    Substitutes z_j = σ · z_i into H and removes qubit j, producing a
    Hamiltonian over (n - 1) active variables.

    Parameters
    ----------
    h, J, offset : current Ising coefficients (using original qubit indices)
    active       : list of currently active qubit indices
    eliminate    : qubit index to remove (j)
    reference    : qubit index to keep  (i)
    sign         : +1 (correlated) or -1 (anti-correlated)
    """
    j, i, sigma = eliminate, reference, sign
    new_h = dict(h)
    new_J = dict(J)
    new_offset = offset

    # 1. Field term: h[j] · z_j = h[j] · σ · z_i  →  merge into h[i]
    new_h[i] = new_h.get(i, 0.0) + sigma * new_h.get(j, 0.0)

    # 2. Process every J term that involves the eliminated qubit j
    to_remove = []
    to_add = {}

    for (a, b), J_val in new_J.items():
        if a != j and b != j:
            continue  # doesn't involve j — skip
        to_remove.append((a, b))

        if {a, b} == {i, j}:
            # J_ij · z_i · z_j = J_ij · z_i · σ · z_i = J_ij · σ  (constant)
            new_offset += sigma * J_val
        else:
            # J_{j,k} · z_j · z_k = J · σ · z_i · z_k  →  coupling (i, k)
            k = b if a == j else a
            pair = (min(i, k), max(i, k))
            to_add[pair] = to_add.get(pair, 0.0) + sigma * J_val

    for key in to_remove:
        del new_J[key]
    for key, val in to_add.items():
        new_J[key] = new_J.get(key, 0.0) + val

    # 3. Clean up: remove j from h and the active set
    new_h.pop(j, None)
    new_active = [q for q in active if q != j]
    return new_h, new_J, new_offset, new_active


def _reduce_hamiltonian_single(
    h: dict, J: dict, offset: float,
    active: list[int],
    fix_qubit: int, fix_spin: int,
) -> tuple[dict, dict, float, list[int]]:
    """
    Reduce Hamiltonian by fixing z_{fix_qubit} = fix_spin (±1 constant).

    All terms involving fix_qubit become constants (added to offset) or
    field terms on neighbouring qubits.
    """
    j, s = fix_qubit, fix_spin
    new_h = dict(h)
    new_J = dict(J)
    new_offset = offset

    # h[j] · z_j = h[j] · s  (constant)
    new_offset += new_h.get(j, 0.0) * s

    # J terms involving j  →  field terms on the other qubit
    to_remove = []
    for (a, b), J_val in new_J.items():
        if a != j and b != j:
            continue
        to_remove.append((a, b))
        k = b if a == j else a
        # J · z_j · z_k = J · s · z_k  →  acts as field h[k] += J · s
        new_h[k] = new_h.get(k, 0.0) + J_val * s

    for key in to_remove:
        del new_J[key]

    new_h.pop(j, None)
    new_active = [q for q in active if q != j]
    return new_h, new_J, new_offset, new_active


def _build_compact_hamiltonian(
    h: dict, J: dict, offset: float, active: list[int],
) -> IsingHamiltonian:
    """
    Re-index active qubits to contiguous 0..n-1 and build IsingHamiltonian.
    """
    idx_map = {orig: compact for compact, orig in enumerate(active)}
    n = len(active)

    compact_h = {}
    for orig_q, val in h.items():
        if orig_q in idx_map and abs(val) > 1e-15:
            compact_h[idx_map[orig_q]] = val

    compact_J = {}
    for (orig_i, orig_j), val in J.items():
        if orig_i in idx_map and orig_j in idx_map and abs(val) > 1e-15:
            ci, cj = idx_map[orig_i], idx_map[orig_j]
            key = (min(ci, cj), max(ci, cj))
            compact_J[key] = compact_J.get(key, 0.0) + val

    return IsingHamiltonian(
        n_qubits=n, h=compact_h, J=compact_J, offset=offset,
        qubit_map={}, route_ids={},
    )


def _compute_correlations(qaoa: QAOACircuit, params: np.ndarray):
    """
    Compute ⟨Zᵢ⟩ and ⟨ZᵢZⱼ⟩ from the QAOA state (exact statevector).

    Returns
    -------
    single : dict[int, float]  — single-qubit expectations ⟨Zᵢ⟩
    pair   : dict[tuple, float] — pair correlations ⟨ZᵢZⱼ⟩
    """
    gamma = params[: qaoa.p]
    beta = params[qaoa.p :]

    state = QuantumState(qaoa.n)
    state.set_zero_state()
    circuit = qaoa.build_circuit(gamma, beta)
    circuit.update_quantum_state(state)

    vec = state.get_vector()
    probs = np.abs(vec) ** 2
    n = qaoa.n

    # Efficient single-pass: precompute spins for each basis state
    single = np.zeros(n)
    pair_matrix = np.zeros((n, n))

    for x in range(2 ** n):
        p_x = probs[x]
        if p_x < 1e-18:
            continue
        spins = np.array([1 - 2 * ((x >> i) & 1) for i in range(n)],
                         dtype=float)
        single += p_x * spins
        pair_matrix += p_x * np.outer(spins, spins)

    single_dict = {i: float(single[i]) for i in range(n)}
    pair_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_dict[(i, j)] = float(pair_matrix[i, j])

    return single_dict, pair_dict


class RecursiveQAOA:
    """
    RQAOA — Recursive Quantum Approximate Optimization Algorithm.

    Uses quantum correlations to iteratively reduce the problem size:
      1. Run shallow QAOA (p=1–2) on the current n-qubit problem
      2. Measure ⟨Zᵢ⟩ and ⟨ZᵢZⱼ⟩ from the optimised statevector
      3. Find the strongest correlation → fix one variable
      4. Substitute into the Hamiltonian → new (n-1)-qubit problem
      5. Repeat until only `threshold` qubits remain
      6. Solve the tiny remainder exactly (brute-force)
      7. Back-substitute all fixed variables → full n-qubit solution

    This is genuinely quantum-enhanced: the correlations ⟨ZᵢZⱼ⟩ encode
    global problem structure that classical heuristics (greedy, local search)
    cannot efficiently access.

    Reference: Bravyi et al., PRL 125, 260505 (2020)
    """

    def __init__(
        self,
        hamiltonian: IsingHamiltonian,
        qaoa_p: int = 2,
        threshold: int = 3,
        qaoa_restarts: int = 3,
        qaoa_max_iter: int = 150,
    ):
        self.original_ham = hamiltonian
        self.qaoa_p = qaoa_p
        self.threshold = min(threshold, hamiltonian.n_qubits)
        self.qaoa_restarts = qaoa_restarts
        self.qaoa_max_iter = qaoa_max_iter

    def optimize(self) -> dict:
        """
        Run RQAOA and return results in the standard VQE output schema.
        """
        n_original = self.original_ham.n_qubits

        # Working copies of the Hamiltonian coefficients
        h = dict(self.original_ham.h)
        J = dict(self.original_ham.J)
        offset = self.original_ham.offset
        active = list(range(n_original))

        substitutions = []   # track all variable fixings for back-substitution
        reduction_log = []   # human-readable log of each reduction step
        total_evals = 0

        # ── Recursive reduction loop ─────────────────────────────────────
        while len(active) > self.threshold:
            n_active = len(active)
            compact_ham = _build_compact_hamiltonian(h, J, offset, active)

            # Run shallow QAOA on the compact problem
            p_use = min(self.qaoa_p, max(1, n_active - 1))
            qaoa = QAOACircuit(compact_ham, p_layers=p_use)
            vqe = VQEOptimizer(
                qaoa,
                max_iterations=self.qaoa_max_iter,
                n_restarts=self.qaoa_restarts,
            )
            res = vqe.optimize()
            total_evals += res["n_evaluations"]

            # Compute correlations from the optimised state
            single, pair = _compute_correlations(qaoa, res["best_params"])

            # Map compact indices → original indices
            idx_to_orig = {ci: active[ci] for ci in range(n_active)}

            # Find the strongest signal (single or pair)
            best_strength = -1.0
            best_action = None   # ("single", orig_q, spin) or ("pair", elim, ref, sign)

            for ci, val in single.items():
                if abs(val) > best_strength:
                    best_strength = abs(val)
                    orig_q = idx_to_orig[ci]
                    best_action = ("single", orig_q, 1 if val > 0 else -1)

            for (ci, cj), val in pair.items():
                if abs(val) > best_strength:
                    best_strength = abs(val)
                    orig_i = idx_to_orig[ci]
                    orig_j = idx_to_orig[cj]
                    sign = 1 if val > 0 else -1
                    # Eliminate the higher-index qubit
                    best_action = ("pair", orig_j, orig_i, sign)

            # Apply the reduction
            if best_action[0] == "single":
                _, orig_q, fix_spin = best_action
                h, J, offset, active = _reduce_hamiltonian_single(
                    h, J, offset, active, orig_q, fix_spin,
                )
                substitutions.append(("single", orig_q, fix_spin))
                reduction_log.append(
                    f"  Round {len(substitutions)}: Fix z_{orig_q} = "
                    f"{fix_spin:+d}  (|signal| = {best_strength:.4f})"
                )
            else:
                _, elim, ref, sign = best_action
                h, J, offset, active = _reduce_hamiltonian_pair(
                    h, J, offset, active, elim, ref, sign,
                )
                substitutions.append(("pair", elim, ref, sign))
                sign_str = "+" if sign > 0 else "-"
                reduction_log.append(
                    f"  Round {len(substitutions)}: Fix z_{elim} = "
                    f"{sign_str}z_{ref}  (|signal| = {best_strength:.4f})"
                )

            logger.debug(reduction_log[-1])

        # ── Solve the remaining small problem exactly ────────────────────
        compact_ham = _build_compact_hamiltonian(h, J, offset, active)
        exact_solver = ExactGroundStateOptimizer(compact_ham)
        exact_result = exact_solver.optimize()
        total_evals += exact_result["n_evaluations"]

        # Map compact exact solution → spins for active qubits
        compact_bits = [int(b) for b in exact_result["best_bitstring"]]
        spins = {}
        for ci, orig_q in enumerate(active):
            spins[orig_q] = 1 - 2 * compact_bits[ci]  # bit → spin

        # ── Back-substitute to recover the full n-qubit solution ─────────
        for sub in reversed(substitutions):
            if sub[0] == "single":
                _, qubit, fix_spin = sub
                spins[qubit] = fix_spin
            else:
                _, elim, ref, sign = sub
                spins[elim] = sign * spins[ref]

        # Convert spins → bitstring
        bits = []
        for i in range(n_original):
            bits.append(0 if spins.get(i, 1) == 1 else 1)
        bitstring = "".join(str(b) for b in bits)

        # Evaluate the final energy on the ORIGINAL Hamiltonian (verification)
        final_energy = ising_energy(self.original_ham, bits)

        # Build the reduction summary
        n_rounds = len(substitutions)
        n_solved = len(active)

        return {
            "best_params":   np.zeros(2),
            "best_energy":   float(final_energy),
            "best_bitstring": bitstring,
            "n_evaluations": int(total_evals),
            "converged":     True,
            "n_restarts":    1,
            "history":       [float(final_energy)],
            "optimizer_msg": (
                f"RQAOA: {n_rounds} reductions "
                f"({n_original}→{n_solved} qubits), "
                f"exact solve on {n_solved}-qubit core"
            ),
            "method":        "rqaoa",
            "reduction_log": reduction_log,
            "n_reductions":  n_rounds,
            "final_n_qubits": n_solved,
        }
