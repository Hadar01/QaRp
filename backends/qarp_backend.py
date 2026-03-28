"""
qarp_backend.py
===============
PHASE 2 — QARP v0.4.3 Platform Integration (Fujitsu FX700)
===========================================================

QARP ARCHITECTURE (v0.4.3):
────────────────────────────
  ┌──────────────────────────────────────────────────────────┐
  │  Your Code  →  qarp.engines.QulacsEngine / TketEngine    │
  │                      ↓                                   │
  │  engine.build(measurements=[StateVector(...)])            │
  │  result = engine.run(parameters={symbol: value})          │
  │                                                          │
  │  Backend choice (runtime-configurable via TketEngine):    │
  │    • QulacsEngine  (default, single-node, CPU)            │
  │    • TketEngine + pytket-tenet (MPS, tensor network)      │
  │    • TketEngine + Qiskit AerBackend (MPS)                 │
  └──────────────────────────────────────────────────────────┘

KEY CHANGES from v1.6.2 to v0.4.3:
  - Hamiltonians: openfermion.QubitOperator replaces PauliHamiltonian
  - Circuits: qarp.blocks.CompositeBlock replaces ParametricCircuit
  - Engines: build(measurements) + run(params) pattern
  - Algorithms: VQE, QAOA, AdaptVQE composites
  - Cutting: EAPartitioning replaces CircuitCutter
"""

import os
import time
import logging
from typing import Optional
from enum import Enum
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

# ── QARP v0.4.3 imports ────────────────────────────────────────────────────
# On FX700, the real qarp package + openfermion are installed in site-packages.
# Locally, qarp_mock provides the same API using Qulacs.
try:
    from qarp.engines import QulacsEngine, TketEngine
    from qarp.algorithms.primitives import StateVector, PauliAveraging, Sampler
    from qarp.algorithms.composite import VQE as QarpVQE, AdaptVQE as QarpAdaptVQE
    from qarp.optimizers import ScipyOptimizer as QarpScipyOptimizer
    from qarp.cutting import EAPartitioning
    from qarp.devices import Device, NoiseModel
    from openfermion import QubitOperator
    QARP_AVAILABLE = True
    QARP_MOCK = False
except ImportError:
    try:
        from backends.qarp_mock import (
            QulacsEngine, TketEngine,
            StateVector, PauliAveraging, Sampler,
            VQE as QarpVQE, AdaptVQE as QarpAdaptVQE,
            ScipyOptimizer as QarpScipyOptimizer,
            EAPartitioning,
            Device, NoiseModel,
            QubitOperator,
        )
        # Block imports for ansatz building (mock-specific since real QARP
        # uses pytket Circuit objects inside blocks)
        from backends.qarp_mock.blocks import (
            CompositeBlock, HnBlock, IsingCostBlock, MixerBlock,
        )
        QARP_AVAILABLE = True
        QARP_MOCK = True
        logging.info(
            "QARP v0.4.3 mock loaded — all code paths execute locally via Qulacs. "
            "On FX700, the real qarp package takes priority."
        )
    except ImportError:
        QARP_AVAILABLE = False
        QARP_MOCK = False
        logging.warning(
            "QARP and qarp_mock both unavailable. Running in LOCAL SIMULATION mode. "
            "On FX700, ensure venv is activated and qarp is in site-packages."
        )

# Also import block classes for real QARP (if not mock)
if QARP_AVAILABLE and not QARP_MOCK:
    try:
        from qarp.blocks import CompositeBlock
        from qarp.blocks.primitives import HnBlock
    except ImportError:
        pass  # Will build circuits differently on FX700

# ── Optional backend imports ─────────────────────────────────────────────────
try:
    from pytket.extensions.tenet import (
        MPSInnerProductBackend, InnerProductBackend, Config as TenetConfig,
    )
    TENET_AVAILABLE = True
except ImportError:
    TENET_AVAILABLE = False

try:
    from pytket.extensions.qiskit import AerBackend
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ── Qulacs fallback (available in local dev env; mpiQulacs on FX700) ─────────
try:
    from qulacs import QuantumState, QuantumCircuit, Observable
    QULACS_AVAILABLE = True
except ImportError:
    QULACS_AVAILABLE = False
    logging.warning(
        "qulacs not found. LOCAL_SIM mode will be unavailable. "
        "Install qulacs for local development, or use mpiQulacs on FX700."
    )

from core.problem_encoder import IsingHamiltonian

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend Configuration
# ---------------------------------------------------------------------------

class BackendType(str, Enum):
    """
    Selects the simulation backend for QARP v0.4.3.

    QARP v0.4.3 uses QulacsEngine (default) or TketEngine(backend=...).
    QulacsEngine is the fastest for small circuits.
    TketEngine wraps any pytket-compatible backend (Tenet MPS, Qiskit Aer).
    """
    QULACS_SINGLE = "qulacs"          # QulacsEngine — fastest for ≤12 qubits
    QULACS_MPI    = "qulacs_mpi"      # QulacsEngine with parallelize=True
    QISKIT_AER    = "qiskit_aer"      # TketEngine + Qiskit AerBackend (MPS)
    PYTKET_TENET  = "pytket_tenet"    # TketEngine + pytket-tenet (MPS)
    LOCAL_SIM     = "local_sim"       # Phase 1 raw Qulacs — for local dev only


@dataclass
class QARPConfig:
    """Runtime configuration for the QARP backend."""
    backend_type:   BackendType = BackendType.LOCAL_SIM
    mps_bond_dim:   int         = 2        # for Tenet/MPS backends; higher = more accurate
    n_mpi_nodes:    int         = 1        # for MPI backend (matched to sbatch -N)
    shots:          int         = 10000    # measurement samples for sampler-based VQE
    max_iterations: int         = 300      # scipy optimizer max_iter
    optimizer_method: str       = "COBYLA" # scipy method


# ---------------------------------------------------------------------------
# QARP Backend Factory
# ---------------------------------------------------------------------------

def build_qarp_engine(config: QARPConfig):
    """
    Constructs a QARP v0.4.3 engine with the selected backend.

    QARP v0.4.3 provides:
      - QulacsEngine(parallelize=False) — direct Qulacs, fastest for small circuits
      - TketEngine(backend=...)         — any pytket-compatible backend

    Returns:
        Engine instance if QARP available, else None (falls back to local).
    """
    if not QARP_AVAILABLE:
        logger.warning("QARP unavailable — falling back to local Qulacs simulation.")
        return None

    if config.backend_type == BackendType.QULACS_SINGLE:
        engine = QulacsEngine(parallelize=False)
        logger.info("Backend: QARP QulacsEngine (single-node)")
        return engine

    elif config.backend_type == BackendType.QULACS_MPI:
        engine = QulacsEngine(parallelize=True)
        logger.info(f"Backend: QARP QulacsEngine (parallelize=True, {config.n_mpi_nodes} nodes)")
        return engine

    elif config.backend_type == BackendType.QISKIT_AER:
        if not QISKIT_AVAILABLE:
            raise RuntimeError("qiskit-aer not installed. Run: pip install qiskit-aer")
        aer_backend = AerBackend()
        engine = TketEngine(backend=aer_backend)
        logger.info("Backend: QARP TketEngine + Qiskit AerBackend (Tensor Network / MPS)")
        return engine

    elif config.backend_type == BackendType.PYTKET_TENET:
        if not TENET_AVAILABLE:
            raise RuntimeError(
                "pytket-tenet not installed. Run `make install` in pytket-tenet/ directory. "
                "See Part 2 of the FX700 setup guide."
            )
        tenet_config = TenetConfig(
            mps_bond_dim=config.mps_bond_dim,
            transpile_to_mps=False,
        )
        tenet_backend = MPSInnerProductBackend(tenet_config)
        engine = TketEngine(backend=tenet_backend)
        logger.info(f"Backend: QARP TketEngine + pytket-tenet (bond_dim={config.mps_bond_dim})")
        return engine

    elif config.backend_type == BackendType.LOCAL_SIM:
        logger.info("Backend: LOCAL_SIM — using Phase 1 Qulacs directly (no QARP wrapper)")
        return None

    raise ValueError(f"Unknown backend: {config.backend_type}")


# ---------------------------------------------------------------------------
# QARP Hamiltonian Builder
# ---------------------------------------------------------------------------

def build_qarp_hamiltonian(ising: IsingHamiltonian):
    """
    Converts our IsingHamiltonian into a QARP v0.4.3 compatible Hamiltonian.

    QARP v0.4.3 uses openfermion.QubitOperator for Hamiltonians:
        H = Σ h_i Z_i + Σ J_ij Z_i Z_j + offset * I

    If QARP is unavailable, returns None (local simulator handles it).
    """
    if not QARP_AVAILABLE:
        return None

    # Start with the offset as an identity term
    op = QubitOperator("", ising.offset)

    # Single-qubit Z terms: h_i * Z_i
    for qubit_idx, strength in ising.h.items():
        if abs(strength) > 1e-10:
            op += QubitOperator(f"Z{qubit_idx}", strength)

    # Two-qubit ZZ terms: J_ij * Z_i * Z_j
    for (i, j), strength in ising.J.items():
        if abs(strength) > 1e-10:
            op += QubitOperator(f"Z{i} Z{j}", strength)

    # Preserve the true qubit count even when some h_i are zero
    op._n_qubits_hint = ising.n_qubits

    return op


# ---------------------------------------------------------------------------
# QARP QAOA Ansatz Builder
# ---------------------------------------------------------------------------

def build_qarp_ansatz(ising: IsingHamiltonian, p_layers: int):
    """
    Builds a QARP v0.4.3 Block implementing the QAOA ansatz.

    QARP v0.4.3 uses the Block abstraction instead of ParametricCircuit:
      - CompositeBlock sequences multiple sub-blocks
      - HnBlock applies H^⊗n (Hadamard initialization)
      - IsingCostBlock applies exp(-iγ H_P) (problem unitary)
      - MixerBlock applies exp(-iβ Σ X_i) (mixer unitary)

    Symbols (named parameters):
      gamma_0, gamma_1, ..., gamma_{p-1}   — cost layer angles
      beta_0,  beta_1,  ..., beta_{p-1}    — mixer layer angles

    Returns:
        CompositeBlock if QARP available, else None.
    """
    if not QARP_AVAILABLE:
        return None

    n = ising.n_qubits
    blocks = [HnBlock(n)]

    for layer in range(p_layers):
        blocks.append(IsingCostBlock(n, ising.h, ising.J, f"gamma_{layer}"))
        blocks.append(MixerBlock(n, f"beta_{layer}"))

    ansatz = CompositeBlock(blocks=blocks, n_qubits=n)
    return ansatz


# ---------------------------------------------------------------------------
# QARP VQE Runner
# ---------------------------------------------------------------------------

class QARPOptimizer:
    """
    Runs VQE using QARP v0.4.3's engine for expectation value evaluation.

    Uses the QARP v0.4.3 build/run pattern:
      measurement = StateVector(bra=ansatz, operator=hamiltonian, ket=ansatz)
      engine.build(measurements=[measurement])
      energy = engine.run(parameters={symbol: value})

    Falls back to Phase 1 Qulacs when QARP is unavailable.
    """

    def __init__(
        self,
        ising: IsingHamiltonian,
        config: QARPConfig,
        p_layers: int = 2,
    ):
        self.ising   = ising
        self.config  = config
        self.p       = p_layers
        self.n       = ising.n_qubits

        # Build backend engine
        self.engine = build_qarp_engine(config)

        # Build QARP objects if available, otherwise use local Qulacs path
        if self.engine is not None:
            self.ham     = build_qarp_hamiltonian(ising)
            self.circuit = build_qarp_ansatz(ising, p_layers)
            self._mode   = "qarp"
        else:
            # Fall back to Phase 1 local simulation
            self._mode   = "local"
            self._setup_local_fallback()

        logger.info(f"QARPOptimizer initialized: mode={self._mode}, n_qubits={self.n}, p={self.p}")

    def _setup_local_fallback(self):
        """Initialize Phase 1 Qulacs objects for local dev/testing."""
        if not QULACS_AVAILABLE:
            raise RuntimeError(
                "LOCAL_SIM mode requires qulacs. "
                "Install it with: pip install qulacs\n"
                "On FX700, use 'qulacs' or 'qulacs_mpi' backend instead."
            )
        from core.qaoa_circuit import QAOACircuit, VQEOptimizer
        self._local_qaoa = QAOACircuit(self.ising, p_layers=self.p)
        self._local_vqe  = VQEOptimizer(
            self._local_qaoa,
            method=self.config.optimizer_method,
            max_iterations=self.config.max_iterations,
            n_restarts=3,
        )

    def optimize(self) -> dict:
        """Run VQE and return optimization result dict."""
        if self._mode == "local":
            logger.info("Running LOCAL simulation (Phase 1 Qulacs fallback)")
            return self._local_vqe.optimize()
        else:
            logger.info(f"Running QARP simulation (backend={self.config.backend_type})")
            return self._run_qarp_vqe()

    def _run_qarp_vqe(self) -> dict:
        """
        QARP v0.4.3 native VQE loop.

        Uses the engine.build(measurements) + engine.run(parameters) pattern.
        """
        # Get the symbol names from the ansatz (gamma_0, beta_0, gamma_1, beta_1, ...)
        symbols = self.circuit.get_symbols()
        n_params = len(symbols)

        best_energy = float('inf')
        best_params = None
        best_result = None
        total_evals = 0
        history = []

        # Set up the measurement: ⟨ψ(params)|H|ψ(params)⟩
        measurement = StateVector(
            bra=self.circuit, operator=self.ham, ket=self.circuit)
        self.engine.build(measurements=[measurement])

        def cost_fn(params: np.ndarray) -> float:
            """Cost function: bind parameters and evaluate via engine."""
            param_map = dict(zip(symbols, params))
            val = float(self.engine.run(parameters=param_map))
            history.append(val)
            return val

        for restart in range(3):
            rng = np.random.default_rng(seed=restart * 17 + 3)
            init_params = rng.uniform(0, np.pi, n_params)

            result = minimize(
                fun=cost_fn,
                x0=init_params,
                method=self.config.optimizer_method,
                options={"maxiter": self.config.max_iterations},
            )
            total_evals += result.nfev

            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
                best_result = result

        if best_result is None:
            best_converged = False
            best_msg       = "All restarts failed to improve energy"
        else:
            best_converged = bool(best_result.success)
            best_msg       = str(best_result.message)

        return {
            "best_params":    best_params if best_params is not None else np.zeros(n_params),
            "best_energy":    float(best_energy),
            "n_evaluations":  int(total_evals),
            "converged":      best_converged,
            "n_restarts":     3,
            "history":        [float(e) for e in history],
            "optimizer_msg":  best_msg,
            "backend":        self.config.backend_type.value,
        }

    def get_best_bitstring(self, params: np.ndarray, n_shots: int = 1000):
        """
        Sample the optimized circuit.
        Falls back to Phase 1 method if using local simulation.
        """
        if self._mode == "local":
            return self._local_qaoa.get_best_bitstring(params, n_shots)

        # QARP v0.4.3 sampler path: set symbols, then engine.sample()
        symbols = self.circuit.get_symbols()
        param_map = dict(zip(symbols, params))
        self.circuit.set_symbols(param_map)

        samples = self.engine.sample(
            circuit=self.circuit,
            n_shots=n_shots,
        )

        # samples is dict[tuple[int,...], float] — bitstring tuples → probabilities
        from collections import Counter
        best_bs_tuple = max(samples, key=samples.get)
        best_prob = samples[best_bs_tuple]

        # Convert tuple to string (qubit 0 = first char)
        best_bitstring = ''.join(str(b) for b in best_bs_tuple)
        confidence = float(best_prob)

        # Convert all counts to {int: count} format for compatibility
        counts = {}
        for bs_tuple, prob in samples.items():
            int_val = int(''.join(str(b) for b in reversed(bs_tuple)), 2)
            counts[int_val] = int(round(prob * n_shots))

        return best_bitstring, confidence, counts


# ---------------------------------------------------------------------------
# FX700 Job Script Generator
# ---------------------------------------------------------------------------

def generate_slurm_script(
    config: QARPConfig,
    n_qubits: int,
    script_name: str = "run_qsc_job.job",
) -> str:
    """
    Auto-generates a Slurm batch script for FX700 submission.

    Based on the job script patterns from the FX700 setup guide.
    Adjust --time based on problem size using the scaling table
    in README.md.

    Args:
        config       : QARPConfig with backend and MPI settings
        n_qubits     : number of qubits (= number of routes)
        script_name  : output filename (reserved for future use)

    Returns:
        The script content as a string.
    """
    # Estimate walltime from problem size + backend
    if n_qubits <= 10:
        walltime = "00:30:00"
    elif n_qubits <= 20:
        walltime = "02:00:00"
    elif n_qubits <= 30:
        walltime = "06:00:00"
    else:
        walltime = "12:00:00"   # coordinate with Fujitsu support for 30+ qubits

    n_nodes = config.n_mpi_nodes

    # MPI run command mirrors the patterns in the FX700 setup guide
    if config.backend_type == BackendType.QULACS_MPI:
        mpi_prefix = f"mpirun -N {n_nodes} -npernode 1 -n {n_nodes} \\\n    job.sh "
    else:
        mpi_prefix = ""

    tenet_warning = ""
    if config.backend_type == BackendType.PYTKET_TENET:
        tenet_warning = """# IMPORTANT: Submit pytket-tenet jobs ONE AT A TIME.
# Simultaneous submission causes Julia environment file-access conflicts.
# First run may take 2+ hours due to JIT precompilation (expected behavior).
"""

    script = f"""#!/bin/bash
#SBATCH --job-name=qsc_supply_chain
#SBATCH -N {n_nodes}
#SBATCH -p Batch
#SBATCH --time={walltime}
#SBATCH --output=qsc-%j.log
#SBATCH --error=qsc-%j.err
# NOTE: The IntrHPC queue is suspended during QSC2025. Use -p Batch (sbatch command).
# Ref: Fujitsu Quantum Simulator2 v1.6.2 QSC2025 release notes (2026/2/24).

{tenet_warning}
# ── Environment setup (from FX700 QARP Setup Guide) ─────────────────────────
export LD_LIBRARY_PATH=/home/share/developer/boost1.90.0/lib:/usr/local/lib
export BOOST_ROOT=/home/share/developer/boost-1.90.0

cd ~/QARPdemo
source venv/bin/activate

# ── Run quantum supply chain optimization ────────────────────────────────────
{mpi_prefix}python -u run_optimization.py \\
    --backend {config.backend_type.value} \\
    --n_qubits {n_qubits} \\
    --shots {config.shots} \\
    --max_iter {config.max_iterations}

echo "Job completed: $(date)"
"""
    return script


# ---------------------------------------------------------------------------
# Environment detector
# ---------------------------------------------------------------------------

class QARPAdaptOptimizer:
    """
    ADAPT-VQE via QARP v0.4.3 — adaptive ansatz growth.

    Uses QARP's AdaptVQE composite algorithm when available.
    Falls back to local ADAPT implementation otherwise.
    """

    def __init__(self, ising: IsingHamiltonian, config: QARPConfig,
                 max_layers: int = 8, gradient_threshold: float = 0.01):
        self.ising = ising
        self.config = config
        self.n = ising.n_qubits
        self.max_layers = max_layers
        self.grad_threshold = gradient_threshold
        self.engine = build_qarp_engine(config)

    def optimize(self) -> dict:
        """Run ADAPT-VQE / ADAPT-QAOA."""
        if self.engine is not None and QARP_AVAILABLE:
            return self._run_qarp_adapt()
        return self._run_local_adapt()

    def _run_qarp_adapt(self) -> dict:
        """QARP v0.4.3 native ADAPT-VQE execution."""
        qarp_ham = build_qarp_hamiltonian(self.ising)

        # Build reference state (Hadamard on all qubits)
        reference = HnBlock(self.n)

        # QARP v0.4.3 AdaptVQE with ScipyOptimizer
        optimizer = QarpScipyOptimizer(
            method=self.config.optimizer_method,
            options={"maxiter": self.config.max_iterations},
        )

        adapt = QarpAdaptVQE(
            reference_block=reference,
            system_hamiltonian=qarp_ham,
            excitation_pool=None,  # Use default pool
            optimizer=optimizer,
            gradient_thresh=self.grad_threshold,
            engine=self.engine,
        )
        adapt.build()
        energy, params = adapt.run(max_iter=self.max_layers)

        return {
            "best_energy": float(energy),
            "optimal_layers": self.max_layers,
            "method": "adapt_vqe_qarp",
            "best_params": params if params else [],
        }

    def _run_local_adapt(self) -> dict:
        """Local ADAPT-QAOA implementation mirroring QARP's behavior."""
        from core.qaoa_circuit import AdaptVQEOptimizer
        adapt = AdaptVQEOptimizer(
            hamiltonian=self.ising,
            max_p=self.max_layers,
            gradient_threshold=self.grad_threshold,
        )
        return adapt.optimize()


class QARPCircuitCuttingOptimizer:
    """
    Circuit cutting via QARP v0.4.3 for large supply chain optimization.

    Uses QARP's EAPartitioning (evolutionary algorithm-based) when available.
    Falls back to graph-based partitioning locally.
    """

    def __init__(self, ising: IsingHamiltonian, routes: list,
                 config: QARPConfig, max_fragment_qubits: int = 20):
        self.ising = ising
        self.routes = routes
        self.config = config
        self.max_frag = max_fragment_qubits
        self.engine = build_qarp_engine(config)

    def optimize(self, p_layers: int = 3) -> dict:
        """Run circuit-cutting optimization."""
        if self.engine is not None and QARP_AVAILABLE:
            return self._run_qarp_cutting(p_layers)
        return self._run_local_cutting(p_layers)

    def _run_qarp_cutting(self, p_layers: int) -> dict:
        """QARP v0.4.3 native circuit cutting using EAPartitioning."""
        qarp_ham = build_qarp_hamiltonian(self.ising)
        circuit = build_qarp_ansatz(self.ising, p_layers)

        # EAPartitioning: evolutionary algorithm-based circuit partitioning
        ea = EAPartitioning(
            circuit, max_size_subcircuits=[self.max_frag])
        cut_result = ea.cut()

        # Evaluate each fragment at default parameters
        fragment_energies = []
        for frag in cut_result.subcircuits:
            measurement = StateVector(
                bra=frag, operator=qarp_ham, ket=frag)
            self.engine.build(measurements=[measurement])
            energy = float(self.engine.run(parameters={}))
            fragment_energies.append(energy)

        total_energy = sum(fragment_energies)

        return {
            "method": "circuit_cutting_qarp",
            "n_fragments": len(cut_result.subcircuits),
            "combined_energy": float(total_energy),
        }
    
    def _run_local_cutting(self, p_layers: int) -> dict:
        """Local circuit cutting using graph partitioning."""
        from core.advanced_algorithms import CircuitCuttingPipeline, parse_supply_chain
        # We need nodes/demands for the pipeline — extract from routes
        # The caller should pass these; for now use a simplified wrapper
        from core.problem_encoder import ProblemEncoder
        encoder = ProblemEncoder(penalty_weight=10.0)
        # Use the Ising directly — partition routes by index ranges
        from core.qaoa_circuit import QAOACircuit, VQEOptimizer, SolutionDecoder
        n = self.ising.n_qubits
        # Partition into fragments
        indices = list(range(n))
        fragments = []
        for ci in range(0, n, self.max_frag):
            chunk = indices[ci:ci + self.max_frag]
            fragments.append(chunk)
        # Optimize each fragment independently
        full_bits = ['0'] * n
        frag_results = []
        for frag_idx, chunk in enumerate(fragments):
            # Build sub-hamiltonian from indices
            sub_h = {i: self.ising.h.get(ci, 0.0) for i, ci in enumerate(chunk)}
            sub_J = {}
            for (qi, qj), val in self.ising.J.items():
                if qi in chunk and qj in chunk:
                    sub_J[(chunk.index(qi), chunk.index(qj))] = val
            from core.problem_encoder import IsingHamiltonian
            sub_ising = IsingHamiltonian(
                h=sub_h, J=sub_J, offset=0.0,
                n_qubits=len(chunk), qubit_map={}, route_ids={}
            )
            qaoa = QAOACircuit(sub_ising, p_layers=p_layers)
            vqe = VQEOptimizer(qaoa, max_iterations=80, n_restarts=2)
            res = vqe.optimize()
            bs, conf, _ = qaoa.get_best_bitstring(res['best_params'], n_shots=500)
            for li, gi in enumerate(chunk):
                if li < len(bs):
                    full_bits[gi] = bs[li]
            frag_results.append({
                'n_qubits': len(chunk), 'energy': res['best_energy'],
                'method': 'qaoa_vqe',
            })
        return {
            'method': 'circuit_cutting_local',
            'n_fragments': len(frag_results),
            'fragments': frag_results,
            'combined_bitstring': ''.join(full_bits),
        }


def detect_environment() -> dict:
    """
    Detects whether we're running on FX700 or locally.
    Used by main.py to auto-select the best backend.
    """
    is_fx700 = os.path.exists("/home/share/developer/QARPdemo")
    has_mpi  = os.path.exists("/usr/local/bin/mpirun") or os.path.exists("/usr/bin/mpirun")

    if is_fx700 and QARP_AVAILABLE and has_mpi:
        suggested_backend = BackendType.QULACS_MPI
    elif is_fx700 and QARP_AVAILABLE:
        suggested_backend = BackendType.QULACS_SINGLE
    else:
        suggested_backend = BackendType.LOCAL_SIM

    return {
        "is_fx700":         is_fx700,
        "qarp_available":   QARP_AVAILABLE,
        "qarp_mock":        QARP_MOCK if QARP_AVAILABLE else False,
        "tenet_available":  TENET_AVAILABLE,
        "qiskit_available": QISKIT_AVAILABLE,
        "mpi_available":    has_mpi,
        "suggested_backend": suggested_backend.value,
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.problem_encoder import (
        ProblemEncoder, SupplyNode, Route, DemandForecast
    )
    from core.qaoa_circuit import SolutionDecoder

    env = detect_environment()
    print("Environment detection:")
    for k, v in env.items():
        print(f"  {k}: {v}")

    nodes = [
        SupplyNode("WH-1", "Warehouse", "warehouse", 1000, 800),
        SupplyNode("S-1",  "Store 1",   "retail",    300,  50),
        SupplyNode("S-2",  "Store 2",   "retail",    300,  30),
    ]
    routes = [
        Route("WH-1", "S-1", 100, 2.5, 1.5, 300),
        Route("WH-1", "S-2", 150, 3.0, 2.0, 300),
    ]
    demands = [
        DemandForecast("S-1", 200, priority=3),
        DemandForecast("S-2", 150, priority=2),
    ]

    encoder = ProblemEncoder()
    ham = encoder.encode(nodes, routes, demands)

    # LOCAL_SIM mode — works without FX700
    cfg = QARPConfig(backend_type=BackendType.LOCAL_SIM, max_iterations=100)
    optimizer = QARPOptimizer(ham, cfg, p_layers=1)

    print(f"\nRunning optimization (mode={optimizer._mode})...")
    result = optimizer.optimize()
    print(f"Best energy:   {result['best_energy']:.4f}")
    print(f"Converged:     {result['converged']}")
    print(f"Evaluations:   {result['n_evaluations']}")

    bs, conf, _ = optimizer.get_best_bitstring(result["best_params"])
    print(f"Bitstring:     {bs}  ({conf:.0%} confidence)")

    # Generate example Slurm script
    script = generate_slurm_script(
        QARPConfig(backend_type=BackendType.QULACS_MPI, n_mpi_nodes=4),
        n_qubits=ham.n_qubits
    )
    print(f"\nSample Slurm script:\n{script}")
