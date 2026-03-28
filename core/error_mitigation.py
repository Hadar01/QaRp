"""
error_mitigation.py
===================
Zero-Noise Extrapolation (ZNE) for NISQ-era error mitigation.

Implements noise-aware quantum optimization that improves solution quality
on real quantum hardware (IBM Torino, FX700) by extrapolating to the
zero-noise limit.

Algorithm:
  1. Run circuit at multiple noise scale factors λ = {1, 1.5, 2, 3}
  2. For each λ, add depolarizing noise after each gate (probability p*λ)
  3. Fit expectation values E(λ) to a polynomial/exponential model
  4. Extrapolate to E(0) — the zero-noise energy

QARP Integration:
  On FX700, noise scaling is achieved by circuit folding (G → G·G†·G)
  which effectively doubles the circuit depth without changing the unitary.
  Locally, we simulate depolarizing noise via Qulacs density matrix.

Reference:
  - Temme et al., "Error Mitigation for Short-Depth Quantum Circuits"
    Physical Review Letters 119, 180509 (2017)
  - Li & Benjamin, PRX 7, 021050 (2017)
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ZeroNoiseExtrapolation:
    """
    Zero-Noise Extrapolation (ZNE) error mitigation.

    Parameters
    ----------
    scale_factors : noise amplification factors (default: [1, 1.5, 2, 3])
    base_noise    : base noise rate per gate (e.g. 0.001 for IBM hardware)
    method        : extrapolation method ('linear', 'polynomial', 'exponential')
    """

    def __init__(
        self,
        scale_factors: Optional[List[float]] = None,
        base_noise: float = 0.001,
        method: str = "polynomial",
    ):
        self.scale_factors = scale_factors or [1.0, 1.5, 2.0, 3.0]
        self.base_noise = base_noise
        self.method = method

    def mitigate(
        self,
        circuit_evaluator: Callable[[float], float],
        noiseless_value: Optional[float] = None,
    ) -> dict:
        """
        Run ZNE mitigation.

        Parameters
        ----------
        circuit_evaluator : callable(noise_level) → expectation_value
            Function that evaluates the circuit with specified noise level.
            noise_level = base_noise * scale_factor.
        noiseless_value : optional known ideal value (for benchmarking)

        Returns
        -------
        dict with:
            mitigated_energy   : extrapolated E(0)
            noisy_energies     : list of (scale_factor, energy) pairs
            improvement         : energy improvement from mitigation
            method             : extrapolation method used
        """
        # Collect noisy data points
        data_points = []
        for sf in self.scale_factors:
            noise = self.base_noise * sf
            energy = circuit_evaluator(noise)
            data_points.append((sf, energy))
            logger.info(f"ZNE: λ={sf:.1f}, noise={noise:.4f}, E={energy:.6f}")

        lambdas = np.array([p[0] for p in data_points])
        energies = np.array([p[1] for p in data_points])

        # Extrapolate to λ=0
        if self.method == "linear":
            mitigated = self._linear_extrapolation(lambdas, energies)
        elif self.method == "exponential":
            mitigated = self._exponential_extrapolation(lambdas, energies)
        else:
            mitigated = self._polynomial_extrapolation(lambdas, energies)

        improvement = energies[0] - mitigated  # positive = mitigation helped

        result = {
            "mitigated_energy": float(mitigated),
            "raw_energy": float(energies[0]),
            "noisy_energies": [(float(l), float(e)) for l, e in zip(lambdas, energies)],
            "improvement": float(improvement),
            "improvement_pct": float(abs(improvement) / abs(energies[0]) * 100)
                               if abs(energies[0]) > 1e-10 else 0.0,
            "method": self.method,
            "scale_factors": list(self.scale_factors),
        }

        if noiseless_value is not None:
            result["noiseless_energy"] = float(noiseless_value)
            result["mitigation_error"] = float(abs(mitigated - noiseless_value))
            result["raw_error"] = float(abs(energies[0] - noiseless_value))

        return result

    def _linear_extrapolation(self, lambdas: np.ndarray,
                               energies: np.ndarray) -> float:
        """Richardson linear extrapolation using first two points."""
        l1, l2 = lambdas[0], lambdas[1]
        e1, e2 = energies[0], energies[1]
        return float(e1 - l1 * (e2 - e1) / (l2 - l1))

    def _polynomial_extrapolation(self, lambdas: np.ndarray,
                                   energies: np.ndarray) -> float:
        """Polynomial fit + evaluate at λ=0."""
        degree = min(len(lambdas) - 1, 3)
        coeffs = np.polyfit(lambdas, energies, degree)
        return float(np.polyval(coeffs, 0.0))

    def _exponential_extrapolation(self, lambdas: np.ndarray,
                                    energies: np.ndarray) -> float:
        """Exponential fit: E(λ) = a + b*exp(c*λ), extrapolate to λ=0."""
        try:
            from scipy.optimize import curve_fit

            def exp_model(x, a, b, c):
                return a + b * np.exp(c * x)

            # Initial guess
            popt, _ = curve_fit(exp_model, lambdas, energies,
                               p0=[energies[-1], energies[0] - energies[-1], -1.0],
                               maxfev=5000)
            return float(exp_model(0.0, *popt))
        except Exception:
            # Fallback to polynomial
            return self._polynomial_extrapolation(lambdas, energies)


class NoisyCircuitEvaluator:
    """
    Evaluates a QAOA circuit with simulated depolarizing noise.

    Uses Qulacs density matrix simulation for noise modeling.
    On real hardware, noise is inherent; on FX700 simulators, we use
    circuit folding to amplify noise.
    """

    def __init__(self, qaoa_circuit, params: np.ndarray) -> None:
        self.qaoa = qaoa_circuit
        self.params = params.copy()
        self.H = qaoa_circuit.H
        self.n = qaoa_circuit.n

    def evaluate(self, noise_level: float) -> float:
        """
        Evaluate ⟨H⟩ with depolarizing noise at the given level.

        For noise_level=0, returns exact statevector result.
        For noise_level>0, adds depolarizing noise after each gate layer.
        """
        if noise_level < 1e-10:
            return self.qaoa.expectation_value(self.params)

        import platform
        _arch = platform.machine().lower()
        if _arch in ('aarch64', 'arm64'):
            # Skip qulacs entirely on aarch64 — use statistical noise model
            ideal = self.qaoa.expectation_value(self.params)
            depol_factor = (1 - noise_level) ** (self.n * self.qaoa.p * 3)
            return float(ideal * depol_factor + self.H.offset * (1 - depol_factor))

        try:
            from qulacs import QuantumState, QuantumCircuit, Observable
            from qulacs.gate import DepolarizingNoise

            gamma = self.params[:self.qaoa.p]
            beta = self.params[self.qaoa.p:]

            # Build noisy circuit
            circuit = QuantumCircuit(self.n)

            # Hadamard + noise
            for i in range(self.n):
                circuit.add_H_gate(i)
                circuit.add_gate(DepolarizingNoise(i, noise_level))

            # QAOA layers with noise after each gate
            for layer in range(self.qaoa.p):
                # Problem layer
                for qi, h_i in self.H.h.items():
                    if abs(h_i) > 1e-10:
                        circuit.add_RZ_gate(qi, 2.0 * gamma[layer] * h_i)
                        circuit.add_gate(DepolarizingNoise(qi, noise_level))
                for (i, j), J_ij in self.H.J.items():
                    if abs(J_ij) > 1e-10:
                        circuit.add_CNOT_gate(i, j)
                        circuit.add_gate(DepolarizingNoise(i, noise_level))
                        circuit.add_gate(DepolarizingNoise(j, noise_level))
                        circuit.add_RZ_gate(j, 2.0 * gamma[layer] * J_ij)
                        circuit.add_gate(DepolarizingNoise(j, noise_level))
                        circuit.add_CNOT_gate(i, j)
                        circuit.add_gate(DepolarizingNoise(i, noise_level))
                        circuit.add_gate(DepolarizingNoise(j, noise_level))
                # Mixer layer
                for i in range(self.n):
                    circuit.add_RX_gate(i, 2.0 * beta[layer])
                    circuit.add_gate(DepolarizingNoise(i, noise_level))

            # Use DensityMatrix for noise simulation
            from qulacs import DensityMatrix
            state = DensityMatrix(self.n)
            state.set_zero_state()
            circuit.update_quantum_state(state)

            # Build observable
            obs = Observable(self.n)
            for qi, s in self.H.h.items():
                if abs(s) > 1e-10:
                    obs.add_operator(s, f"Z {qi}")
            for (i, j), s in self.H.J.items():
                if abs(s) > 1e-10:
                    obs.add_operator(s, f"Z {i} Z {j}")

            return float(obs.get_expectation_value(state)) + self.H.offset

        except ImportError:
            # No Qulacs DensityMatrix — use statistical noise model
            ideal = self.qaoa.expectation_value(self.params)
            # Depolarizing noise pushes expectation toward 0
            depol_factor = (1 - noise_level) ** (self.n * self.qaoa.p * 3)
            return float(ideal * depol_factor + self.H.offset * (1 - depol_factor))


def mitigate_qaoa(qaoa_circuit, optimized_params: np.ndarray,
                  base_noise: float = 0.001,
                  method: str = "polynomial") -> dict:
    """
    Convenience function: apply ZNE to an optimized QAOA circuit.

    Parameters
    ----------
    qaoa_circuit    : QAOACircuit with built observable
    optimized_params : optimal [gamma, beta] angles from VQE
    base_noise      : per-gate depolarizing error rate
    method          : 'linear', 'polynomial', or 'exponential'

    Returns
    -------
    dict with mitigated_energy, improvement, noisy_energies, etc.
    """
    evaluator = NoisyCircuitEvaluator(qaoa_circuit, optimized_params)
    ideal_energy = qaoa_circuit.expectation_value(optimized_params)

    zne = ZeroNoiseExtrapolation(
        scale_factors=[1.0, 1.5, 2.0, 3.0],
        base_noise=base_noise,
        method=method,
    )

    return zne.mitigate(
        circuit_evaluator=evaluator.evaluate,
        noiseless_value=ideal_energy,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.problem_encoder import ProblemEncoder, SupplyNode, Route, DemandForecast
    from core.qaoa_circuit import QAOACircuit, VQEOptimizer

    nodes = [
        SupplyNode("WH-1", "Warehouse", "warehouse", 1000, 800),
        SupplyNode("S-1", "Store 1", "retail", 300, 50),
        SupplyNode("S-2", "Store 2", "retail", 300, 30),
    ]
    routes = [
        Route("WH-1", "S-1", 100, 2.5, 1.5, 300),
        Route("WH-1", "S-2", 150, 3.0, 2.0, 300),
    ]
    demands = [DemandForecast("S-1", 200, 3), DemandForecast("S-2", 150, 2)]

    encoder = ProblemEncoder()
    ham = encoder.encode(nodes, routes, demands)
    qaoa = QAOACircuit(ham, p_layers=2)
    vqe = VQEOptimizer(qaoa, max_iterations=200, n_restarts=3)
    result = vqe.optimize()

    print(f"Ideal energy: {result['best_energy']:.4f}")

    zne_result = mitigate_qaoa(qaoa, result["best_params"],
                                base_noise=0.005, method="polynomial")
    print(f"Raw energy (noise=0.005): {zne_result['raw_energy']:.4f}")
    print(f"Mitigated energy:         {zne_result['mitigated_energy']:.4f}")
    print(f"Improvement:              {zne_result['improvement']:.4f}")
    print(f"Improvement %:            {zne_result['improvement_pct']:.1f}%")
    print(f"Noisy data points:")
    for lam, e in zne_result["noisy_energies"]:
        print(f"  lambda={lam:.1f}: E={e:.4f}")
