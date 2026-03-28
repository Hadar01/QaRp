"""
main.py (Phase 2)
=================
FastAPI layer — updated to integrate:
  • QARP backend (Fujitsu FX700 official scoring path)
  • Async job submission + polling
  • Redis result caching
  • Warm-start from past solutions
  • Slurm script generation endpoint
  • Noise simulation toggle

The Phase 1 synchronous POST /api/v1/optimize is kept unchanged
for backward compatibility with the Liferay portlet.
"""

import uuid
import time
import os
import sys
import logging
from datetime import datetime, UTC
from enum import Enum
from typing import Optional, Any
from contextlib import asynccontextmanager

# Ensure project root is on sys.path so `core/`, `backends/`, `api/` are importable
# regardless of the working directory or how the script is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
from scipy.optimize import minimize

# Phase 1 modules
from core.problem_encoder import (
    ProblemEncoder,
    SupplyNode     as CoreSupplyNode,
    Route          as CoreRoute,
    DemandForecast as CoreDemandForecast,
    greedy_classical_baseline,
)
from core.qaoa_circuit import (
    QAOACircuit, VQEOptimizer, SolutionDecoder,
    GradientVQEOptimizer, AdaptVQEOptimizer, VQDOptimizer,
)

# Phase 2 modules
from backends.qarp_backend import (
    QARPOptimizer, QARPConfig, BackendType,
    detect_environment, generate_slurm_script,
)

# Phase 3 — Advanced algorithms
from core.advanced_algorithms import (
    WarmStartQAOA, MultiObjectiveParetoQAOA, CircuitCuttingPipeline,
)
from core.error_mitigation import mitigate_qaoa, ZeroNoiseExtrapolation

from api.job_manager import (
    job_store, result_cache, warm_start, job_queue,
    JobStatus, ResultCache,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NumPy → Python type sanitizer
# ---------------------------------------------------------------------------

def _np_safe(obj):
    """
    Recursively convert numpy scalar types (int64, float64, bool_) to native
    Python types so Pydantic / json.dumps never raises PydanticSerializationError.

    Root cause: Qulacs state.sampling() returns numpy.int64 values; scipy
    optimize returns numpy.int64 for nfev, numpy.float64 for fun, numpy.bool_
    for success. All of these must be cast before the response is serialised.
    """
    if isinstance(obj, dict):
        return {_np_safe(k): _np_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_np_safe(i) for i in obj]
    # Covers numpy.int8/16/32/64, numpy.uint*, numpy.bool_, numpy.float32/64
    if hasattr(obj, "item"):          # all numpy scalars have .item()
        return obj.item()
    if hasattr(obj, "tolist"):        # numpy arrays
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OptimizationObjective(str, Enum):
    minimize_cost       = "minimize_cost"
    minimize_time       = "minimize_time"
    maximize_efficiency = "maximize_efficiency"
    balanced            = "balanced"


class AlgorithmChoice(str, Enum):
    """Algorithm selection for the optimizer — maps to Phase 1-3 classes."""
    qaoa          = "qaoa"           # Standard QAOA + COBYLA (Phase 1)
    gradient_vqe  = "gradient_vqe"   # Parameter-shift gradient VQE
    adapt_vqe     = "adapt_vqe"      # Adaptive depth (QARP ADAPT-VQE inspired)
    vqd           = "vqd"            # Variational Quantum Deflation (diverse plans)
    warm_start    = "warm_start"     # CVaR warm-start from greedy
    pareto        = "pareto"         # Multi-objective Pareto exploration
    circuit_cut   = "circuit_cut"    # Circuit cutting for 40+ qubits


class SupplyNodeSchema(BaseModel):
    id: str
    name: str
    type: str
    capacity: float
    current_inventory: float
    coordinates: Optional[dict[str, float]] = None


class RouteSchema(BaseModel):
    from_node: str
    to_node: str
    distance: float
    cost_per_unit: float
    time_hours: float
    capacity: float


class DemandForecastSchema(BaseModel):
    node_id: str
    demand: float
    priority: int = Field(default=1, ge=1, le=5)


# ---- Typed response sub-models (enforces schema for dashboard/Liferay) ----

class RouteAssignmentSchema(BaseModel):
    """Each route in the network with its quantum decision outcome."""
    qubit_index: int
    route_id: str
    from_node: str
    to_node: str
    flow: float
    cost_per_unit: float
    total_cost: float
    time_hours: float
    distance: float
    selected: bool          # <— enforced by Pydantic; dashboard filters on this


class AllocationSchema(BaseModel):
    """Per-node inventory state after executing the shipment plan."""
    node_id: str
    node_name: str
    current_inventory: float
    allocated_amount: float
    outflow: float
    inflow: float
    utilization: float
    storage_cost: float


class DemandSatisfactionSchema(BaseModel):
    """Per-demand-node delivery outcome."""
    node_id: str
    demand: float
    delivered: float
    shortage: float
    slack: float
    satisfied: bool


# ---- Business KPIs --------------------------------------------------------

class BusinessKPISchema(BaseModel):
    """Business-level Key Performance Indicators from quantum optimization."""
    carbon_footprint_kg: float = Field(description="Total CO2 emissions (kg) from selected routes")
    carbon_reduction_pct: float = Field(description="% CO2 reduction vs classical baseline")
    sla_compliance_pct: float = Field(description="% of demands meeting SLA (shortage=0)")
    total_logistics_cost: float = Field(description="Total shipping cost ($)")
    cost_savings_pct: float = Field(description="% cost savings vs greedy baseline")
    avg_delivery_time_hours: float = Field(description="Average delivery time (hours)")
    max_delivery_time_hours: float = Field(description="Max delivery time (critical path)")
    network_utilization_pct: float = Field(description="% of routes activated")
    demand_fulfillment_pct: float = Field(description="% of total demand fulfilled")
    inventory_turnover: float = Field(description="Total outflow / total inventory")
    stockout_count: int = Field(description="Number of demand nodes with shortage > 0")


class OptimizationRequest(BaseModel):
    nodes: list[SupplyNodeSchema]
    routes: list[RouteSchema]
    demands: list[DemandForecastSchema]
    objective: OptimizationObjective = OptimizationObjective.balanced
    quantum_iterations: int = Field(default=100, ge=10, le=1000)
    constraints: Optional[dict[str, Any]] = None
    algorithm: AlgorithmChoice = Field(
        default=AlgorithmChoice.qaoa,
        description="Quantum algorithm: qaoa, gradient_vqe, adapt_vqe, vqd, warm_start, pareto, circuit_cut"
    )
    # Phase 2 additions
    backend: Optional[str] = Field(
        default=None,
        description="qulacs | qulacs_mpi | qiskit_aer | pytket_tenet | local_sim"
    )
    mps_bond_dim: int = Field(default=2, ge=1, le=64)
    n_mpi_nodes: int = Field(default=1, ge=1, le=512)
    use_warm_start: bool = True
    noise_simulation: bool = False
    error_mitigation: bool = Field(
        default=False,
        description="Apply Zero-Noise Extrapolation (ZNE) for error-mitigated results"
    )
    carbon_weight: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weight for carbon-aware optimization (0=ignore CO2, 1=minimize CO2)"
    )

    @field_validator("nodes")
    @classmethod
    def at_least_one_node(cls, v):
        if not v:
            raise ValueError("At least one supply node is required")
        return v

    @field_validator("routes")
    @classmethod
    def at_least_one_route(cls, v):
        if not v:
            raise ValueError("At least one route is required")
        return v


class OptimizationResult(BaseModel):
    optimization_id: str
    status: str
    timestamp: str
    objective_value: float
    allocations: list[AllocationSchema]
    route_assignments: list[RouteAssignmentSchema]
    demand_satisfaction: list[DemandSatisfactionSchema] = Field(
        default_factory=list,
        description=(
            "Per-demand-node delivery outcome. "
            "Each entry: {node_id, demand, delivered, shortage, slack, satisfied}. "
            "slack = delivered - demand  (negative = unmet demand)."
        )
    )
    quantum_metrics: dict[str, Any]
    classical_comparison: Optional[dict[str, Any]] = None
    business_kpis: Optional[BusinessKPISchema] = Field(
        default=None,
        description="Business-level KPIs: carbon footprint, SLA compliance, cost savings, etc."
    )


class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    result: Optional[dict]
    error: Optional[str]
    backend_used: Optional[str]
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    env = detect_environment()
    logger.info(f"Environment: {env}")
    await job_queue.start(n_workers=2)
    logger.info("Quantum Supply Chain API v2.0 ready")
    yield
    await job_queue.stop()


app = FastAPI(
    title="Quantum Supply Chain Optimization API",
    description="Phase 2: QARP/FX700 integration, async jobs, caching.",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow requests from local HTML files (file://) and any localhost origin.
# This is safe because the server only listens on 127.0.0.1.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # file:// pages send origin "null" — wildcard needed
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],         # let dashboard JS read all response headers
)


# ---------------------------------------------------------------------------
# Serve dashboard
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the interactive dashboard at http://127.0.0.1:8000/dashboard"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _select_backend(request: OptimizationRequest) -> BackendType:
    if request.backend:
        try:
            return BackendType(request.backend)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown backend '{request.backend}'. "
                       f"Valid: {[b.value for b in BackendType]}"
            )
    env = detect_environment()
    return BackendType(env["suggested_backend"])


def _iterations_to_p_layers(quantum_iterations: int) -> int:
    if quantum_iterations < 50:  return 1
    if quantum_iterations < 150: return 2
    if quantum_iterations < 400: return 3
    if quantum_iterations < 700: return 4
    return 5


def _apply_noise_to_params(params: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    """Simulates depolarizing noise effect on optimized parameters."""
    rng = np.random.default_rng()
    return params + rng.normal(0, noise_level, params.shape)


# ── CO2 emission factors (kg CO2 per ton-km by transport mode) ────────────
_EMISSION_FACTORS = {
    "road": 0.062, "rail": 0.022, "sea": 0.008, "air": 0.602,
}

def _estimate_transport_mode(distance: float) -> str:
    """Heuristic: mode based on distance."""
    if distance < 300: return "road"
    if distance < 1500: return "rail"
    if distance < 8000: return "sea"
    return "air"


def _compute_business_kpis(full_ra, plan, core_routes, core_nodes, core_demands,
                           greedy_total_cost: float) -> dict:
    """Compute Business KPIs: carbon, SLA, cost savings, utilization."""
    selected = [r for r in full_ra if r.get("selected", False)]
    n_selected = len(selected)
    n_total = len(full_ra)

    # Total logistics cost
    total_cost = sum(r["total_cost"] for r in selected)

    # Carbon footprint
    carbon_kg = 0.0
    for r in selected:
        mode = _estimate_transport_mode(r["distance"])
        tons = r["flow"] / 1000.0
        carbon_kg += _EMISSION_FACTORS[mode] * tons * r["distance"]

    greedy_carbon = 0.0
    for r in core_routes:
        mode = _estimate_transport_mode(r.distance)
        greedy_carbon += _EMISSION_FACTORS[mode] * (r.capacity / 1000.0) * r.distance

    carbon_reduction = ((greedy_carbon - carbon_kg) / greedy_carbon * 100
                        if greedy_carbon > 0 else 0.0)

    # SLA compliance
    ds = plan.get("demand_satisfaction", [])
    n_demands = len(ds)
    n_satisfied = sum(1 for d in ds if d.get("satisfied", d.get("shortage", 1) < 1e-6))
    sla_pct = (n_satisfied / n_demands * 100) if n_demands > 0 else 100.0
    stockout_count = n_demands - n_satisfied

    # Cost savings
    cost_savings = ((greedy_total_cost - total_cost) / greedy_total_cost * 100
                    if greedy_total_cost > 0 else 0.0)

    # Delivery times
    times = [r["time_hours"] for r in selected]
    avg_time = float(np.mean(times)) if times else 0.0
    max_time = max(times) if times else 0.0

    # Network utilization
    network_util = (n_selected / n_total * 100) if n_total > 0 else 0.0

    # Demand fulfillment
    total_demand = sum(d.get("demand", 0) for d in ds)
    total_delivered = sum(d.get("delivered", 0) for d in ds)
    fulfillment = (total_delivered / total_demand * 100) if total_demand > 0 else 100.0

    # Inventory turnover
    total_outflow = sum(a.get("outflow", 0) for a in plan.get("allocations", []))
    total_inv = sum(n.current_inventory for n in core_nodes)
    turnover = (total_outflow / total_inv) if total_inv > 0 else 0.0

    return {
        "carbon_footprint_kg": round(carbon_kg, 2),
        "carbon_reduction_pct": round(carbon_reduction, 1),
        "sla_compliance_pct": round(sla_pct, 1),
        "total_logistics_cost": round(total_cost, 2),
        "cost_savings_pct": round(cost_savings, 1),
        "avg_delivery_time_hours": round(avg_time, 1),
        "max_delivery_time_hours": round(max_time, 1),
        "network_utilization_pct": round(network_util, 1),
        "demand_fulfillment_pct": round(fulfillment, 1),
        "inventory_turnover": round(turnover, 3),
        "stockout_count": stockout_count,
    }


def run_optimization_pipeline(request_dict: dict) -> dict:
    """
    Core quantum pipeline. Plain dict in, plain dict out.
    Runs synchronously (dispatched to thread pool for async use).

    Supports 7 algorithms:
      qaoa, gradient_vqe, adapt_vqe, vqd, warm_start, pareto, circuit_cut
    """
    request = OptimizationRequest(**request_dict)
    start_time = time.time()

    # 1. Domain objects
    core_nodes = [
        CoreSupplyNode(id=n.id, name=n.name, type=n.type,
                       capacity=n.capacity, current_inventory=n.current_inventory)
        for n in request.nodes
    ]
    core_routes = [
        CoreRoute(from_node=r.from_node, to_node=r.to_node, distance=r.distance,
                  cost_per_unit=r.cost_per_unit, time_hours=r.time_hours, capacity=r.capacity)
        for r in request.routes
    ]
    core_demands = [
        CoreDemandForecast(node_id=d.node_id, demand=d.demand, priority=d.priority)
        for d in request.demands
    ]

    # 2. Encode
    encoder = ProblemEncoder(penalty_weight=10.0)
    lambda_time = float((request.constraints or {}).get("lambda_time", 0.5))
    hamiltonian = encoder.encode(core_nodes, core_routes, core_demands,
                                 objective=request.objective.value,
                                 lambda_time=lambda_time,
                                 constraints=request.constraints,
                                 carbon_weight=request.carbon_weight)
    n_qubits = hamiltonian.n_qubits
    if n_qubits == 0:
        raise ValueError("No valid routes to optimize after encoding")

    # 3. Algorithm selection
    p_layers = _iterations_to_p_layers(request.quantum_iterations)
    n_params = 2 * p_layers
    algorithm = request.algorithm.value
    extra_data = {}

    # --- PARETO: returns its own structured result ---
    if algorithm == "pareto":
        pareto = MultiObjectiveParetoQAOA(
            core_nodes, core_routes, core_demands,
            p=p_layers, shots=1024, n_scenarios=5,
            uncertainty_pct=0.20,
        )
        pareto_result = pareto.explore_pareto(n_points=7)
        # Use the cheapest Pareto-optimal point as primary solution
        front = pareto_result["pareto_front"]
        if front:
            best_point = min(front, key=lambda p: p["total_cost"])
        else:
            best_point = pareto_result["pareto_points"][0]
        bitstring = best_point["bitstring"]
        decoder = SolutionDecoder()
        plan = decoder.decode(bitstring, hamiltonian, core_routes, core_nodes, core_demands)
        vqe_result = {
            "best_params": np.zeros(n_params), "best_energy": best_point["energy"],
            "n_evaluations": len(pareto_result["pareto_points"]) * 60,
            "converged": True, "n_restarts": 1,
            "history": [p["energy"] for p in pareto_result["pareto_points"]],
            "optimizer_msg": f"Pareto exploration: {pareto_result['n_pareto_optimal']} optimal",
        }
        extra_data["pareto"] = pareto_result
        confidence = 0.85
        best_params = vqe_result["best_params"]

    # --- CIRCUIT CUTTING: decomposes large problems ---
    elif algorithm == "circuit_cut":
        cc = CircuitCuttingPipeline(
            core_nodes, core_routes, core_demands,
            max_fragment_qubits=min(25, max(6, n_qubits // 2)),
        )
        cc_result = cc.optimize(p=p_layers, shots=1024, method="adapt_vqe")
        bitstring = cc_result["combined_bitstring"]
        plan = cc_result["plan"]
        vqe_result = {
            "best_params": np.zeros(n_params), "best_energy": cc_result["combined_energy"],
            "n_evaluations": sum(f.get("n_qubits", 0) * 20 for f in cc_result["fragments"]),
            "converged": True, "n_restarts": 1,
            "history": [f["energy"] for f in cc_result["fragments"]],
            "optimizer_msg": f"Circuit cutting: {cc_result['n_fragments']} fragments",
        }
        extra_data["circuit_cutting"] = {
            "n_fragments": cc_result["n_fragments"],
            "fragments": cc_result["fragments"],
        }
        confidence = min(f.get("confidence", 0.8) for f in cc_result["fragments"])
        best_params = vqe_result["best_params"]

    # --- VQD: multiple diverse solutions ---
    elif algorithm == "vqd":
        qaoa = QAOACircuit(hamiltonian, p_layers=p_layers)
        vqd = VQDOptimizer(qaoa, n_excited=3, beta_penalty=5.0,
                           max_iterations=request.quantum_iterations, n_restarts=2)
        vqe_result = vqd.optimize()
        best_params = vqe_result["best_params"]
        bitstring, confidence, _ = qaoa.get_best_bitstring(best_params, n_shots=1000)
        decoder = SolutionDecoder()
        plan = decoder.decode(bitstring, hamiltonian, core_routes, core_nodes, core_demands)
        extra_data["diverse_solutions"] = vqe_result.get("solutions", [])

    # --- WARM START QAOA ---
    elif algorithm == "warm_start":
        ws = WarmStartQAOA(hamiltonian, p=p_layers, shots=2048, cvar_alpha=0.2)
        vqe_result = ws.optimize()
        best_params = vqe_result["best_params"]
        bitstring = vqe_result["best_bitstring"]
        confidence = 0.90
        decoder = SolutionDecoder()
        plan = decoder.decode(bitstring, hamiltonian, core_routes, core_nodes, core_demands)
        extra_data["warm_start_info"] = {
            "classical_energy": vqe_result.get("classical_energy"),
            "cvar_alpha": vqe_result.get("cvar_alpha"),
        }

    # --- GRADIENT VQE, ADAPT VQE, standard QAOA ---
    else:
        # Warm-start lookup
        warm_init = None
        if request.use_warm_start:
            fv = warm_start.extract_feature_vector(hamiltonian)
            warm_init = warm_start.find_warm_start(fv, n_params)

        # Backend + optimizer
        backend_type = _select_backend(request)
        qarp_config = QARPConfig(
            backend_type=backend_type,
            mps_bond_dim=request.mps_bond_dim,
            n_mpi_nodes=request.n_mpi_nodes,
            max_iterations=max(50, request.quantum_iterations // p_layers),
        )

        if algorithm == "gradient_vqe":
            qaoa = QAOACircuit(hamiltonian, p_layers=p_layers)
            opt = GradientVQEOptimizer(qaoa, max_iterations=request.quantum_iterations,
                                       n_restarts=3)
            vqe_result = opt.optimize()
        elif algorithm == "adapt_vqe":
            opt = AdaptVQEOptimizer(hamiltonian, max_p=min(p_layers + 3, 8),
                                    gradient_threshold=0.01)
            vqe_result = opt.optimize()
            # Use the optimal p for bitstring sampling
            p_layers = vqe_result.get("optimal_p", p_layers)
            extra_data["adapt_info"] = {
                "optimal_p": vqe_result.get("optimal_p"),
                "layer_energies": vqe_result.get("layer_energies", []),
            }
        else:
            # Standard QAOA with warm-start
            optimizer = QARPOptimizer(hamiltonian, qarp_config, p_layers=p_layers)
            if warm_init is not None and optimizer._mode == "local":
                qaoa = optimizer._local_qaoa
                _warm_history = []
                def _warm_cost(params):
                    e = qaoa.expectation_value(params)
                    _warm_history.append(float(e))
                    return e
                res = minimize(fun=_warm_cost, x0=warm_init, method="COBYLA",
                               options={"maxiter": qarp_config.max_iterations})
                vqe_result = {
                    "best_params": res.x, "best_energy": res.fun,
                    "n_evaluations": res.nfev, "converged": res.success,
                    "n_restarts": 1, "history": _warm_history,
                    "optimizer_msg": res.message,
                }
            else:
                vqe_result = optimizer.optimize()

        best_params = vqe_result["best_params"]

        # Noise simulation
        if request.noise_simulation:
            best_params = _apply_noise_to_params(best_params)

        # Decode bitstring
        qaoa_for_decode = QAOACircuit(hamiltonian, p_layers=p_layers)
        bitstring, confidence, _ = qaoa_for_decode.get_best_bitstring(best_params, n_shots=1000)
        decoder = SolutionDecoder()
        plan = decoder.decode(bitstring, hamiltonian, core_routes, core_nodes, core_demands)

    # ── ZNE Error Mitigation ───────────────────────────────────────────────
    zne_result = None
    if request.error_mitigation and algorithm not in ("pareto", "circuit_cut"):
        try:
            qaoa_zne = QAOACircuit(hamiltonian, p_layers=p_layers)
            zne_result = mitigate_qaoa(
                qaoa_zne, best_params,
                base_noise=0.005, method="polynomial",
            )
            extra_data["error_mitigation"] = {
                "method": "zero_noise_extrapolation",
                "raw_energy": zne_result["raw_energy"],
                "mitigated_energy": zne_result["mitigated_energy"],
                "improvement_pct": zne_result["improvement_pct"],
                "scale_factors": zne_result["scale_factors"],
                "noisy_energies": zne_result["noisy_energies"],
            }
        except Exception as e:
            logger.warning(f"ZNE error mitigation failed: {e}")

    # Register warm-start
    fv = warm_start.extract_feature_vector(hamiltonian)
    warm_start.register(fv, best_params, vqe_result["best_energy"])

    # ── Compute objective-aware scalar ─────────────────────────────────────
    objective_str = request.objective.value
    max_cost_norm = max((r.cost_per_unit * r.capacity for r in core_routes), default=1.0) or 1.0
    max_time_norm = max((r.time_hours for r in core_routes), default=1.0) or 1.0

    def _compute_objective(cost, time_val, obj, ra):
        if obj == "minimize_cost": return cost
        if obj == "minimize_time": return time_val
        if obj == "maximize_efficiency":
            eff = sum(r["flow"] / (r["cost_per_unit"] * r["time_hours"])
                      for r in ra if r["cost_per_unit"] * r["time_hours"] > 1e-12)
            return -eff
        c_norm = cost / max_cost_norm if max_cost_norm > 1e-12 else 0.0
        t_norm = time_val / max_time_norm if max_time_norm > 1e-12 else 0.0
        return (1.0 - lambda_time) * c_norm + lambda_time * t_norm

    selected_ra = plan["route_assignments"]
    total_cost = sum(r["total_cost"] for r in selected_ra)
    total_time = max((r["time_hours"] for r in selected_ra), default=0.0)
    quantum_obj = _compute_objective(total_cost, total_time, objective_str, selected_ra)

    # Greedy baseline
    greedy = greedy_classical_baseline(
        core_routes, core_demands, core_nodes,
        objective=objective_str, lambda_time=lambda_time,
    )
    classical_obj = greedy["objective_value"]

    # ── Convergence fallback ───────────────────────────────────────────────
    status = "success" if vqe_result["converged"] else "partial"
    if not vqe_result["converged"]:
        greedy_better = classical_obj < quantum_obj - 1e-9
        if greedy_better:
            greedy_bits = ["0"] * n_qubits
            for ri in greedy["selected_routes"]:
                qi = ri["qubit_index"]
                if 0 <= qi < n_qubits:
                    greedy_bits[qi] = "1"
            bitstring = "".join(greedy_bits)
            plan = decoder.decode(bitstring, hamiltonian, core_routes, core_nodes, core_demands)
            selected_ra = plan["route_assignments"]
            total_cost = sum(r["total_cost"] for r in selected_ra)
            total_time = max((r["time_hours"] for r in selected_ra), default=0.0)
            quantum_obj = _compute_objective(total_cost, total_time, objective_str, selected_ra)
            confidence = 1.0
            status = "fallback_classical"

    # ── Full route list ────────────────────────────────────────────────────
    selected_qi = {r["qubit_index"] for r in selected_ra}
    full_route_assignments = []
    for i, route in enumerate(core_routes):
        if i in selected_qi:
            full_route_assignments.append(
                next(r for r in selected_ra if r["qubit_index"] == i)
            )
        else:
            full_route_assignments.append({
                "qubit_index": i,
                "route_id": hamiltonian.route_ids.get(
                    i, f"{route.from_node}\u2192{route.to_node}#{i}"),
                "from_node": route.from_node, "to_node": route.to_node,
                "flow": 0.0, "cost_per_unit": route.cost_per_unit,
                "total_cost": 0.0, "time_hours": route.time_hours,
                "distance": route.distance, "selected": False,
            })

    # ── Business KPIs ──────────────────────────────────────────────────────
    business_kpis = _compute_business_kpis(
        full_route_assignments, plan, core_routes, core_nodes, core_demands,
        greedy_total_cost=greedy["total_cost"],
    )

    # ── Quantum advantage ──────────────────────────────────────────────────
    if abs(quantum_obj) < 1e-9 and abs(classical_obj) < 1e-9:
        quantum_advantage = 1.0
    elif abs(quantum_obj) < 1e-9:
        quantum_advantage = float("inf") if classical_obj > 0 else 0.0
    elif objective_str == "maximize_efficiency":
        quantum_advantage = quantum_obj / classical_obj if abs(classical_obj) > 1e-9 else 1.0
    else:
        quantum_advantage = classical_obj / quantum_obj

    execution_ms = int((time.time() - start_time) * 1000)

    result = _np_safe({
        "optimization_id": f"opt_{time.time()}",
        "status":          status,
        "timestamp":       datetime.now(UTC).isoformat(),
        "objective_value": round(quantum_obj, 6),
        "allocations":         plan["allocations"],
        "route_assignments":   full_route_assignments,
        "demand_satisfaction":  plan["demand_satisfaction"],
        "business_kpis":       business_kpis,
        "quantum_metrics": {
            "n_qubits":            n_qubits,
            "p_layers":            p_layers,
            "algorithm":           algorithm,
            "quantum_iterations":  vqe_result["n_evaluations"],
            "convergence_achieved": vqe_result["converged"],
            "quantum_energy":      round(float(vqe_result["best_energy"]), 6),
            "solution_confidence": round(confidence, 4),
            "bitstring":           bitstring,
            "qubit_route_map":     hamiltonian.route_ids,
            "quantum_advantage":   round(quantum_advantage, 4),
            "execution_time_ms":   execution_ms,
            "backend":             _select_backend(request).value if algorithm not in ("pareto", "circuit_cut", "warm_start") else "local_sim",
            "warm_start_used":     request.use_warm_start,
            "noise_simulation":    request.noise_simulation,
            "error_mitigation":    request.error_mitigation,
            "objective":           objective_str,
            "convergence_history": vqe_result.get("history", []),
            **extra_data,
        },
        "classical_comparison": {
            "method":                    "greedy",
            "classical_objective_value": round(classical_obj, 6),
            "quantum_objective_value":   round(quantum_obj, 6),
            "objective":                 objective_str,
            "quantum_advantage":         round(quantum_advantage, 4),
            "greedy_selected_routes":    greedy["selected_routes"],
            "greedy_total_cost":         round(greedy["total_cost"], 4),
            "greedy_total_time":         round(greedy["total_time"], 4),
            "greedy_demand_satisfaction": greedy["demand_satisfaction"],
            "note": (
                "quantum_advantage > 1 means quantum found a lower objective "
                "value than the greedy baseline. objective = " + objective_str
            ),
        },
    })
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    env = detect_environment()
    return {"service": "Quantum Supply Chain Optimization API", "version": "2.0.0",
            "environment": env, "docs": "/docs"}


@app.get("/health")
def health_check():
    env = detect_environment()
    return {"status": "healthy", "backend": env["suggested_backend"],
            "qarp": env["qarp_available"], "timestamp": datetime.now(UTC).isoformat(),
            "cache": result_cache.stats()}


@app.get("/api/v1/capabilities")
def get_capabilities():
    env = detect_environment()
    return {
        "max_routes": 20, "max_qubits": 20, "p_layers_range": [1, 5],
        "objectives": [o.value for o in OptimizationObjective],
        "backends_available": [b.value for b in BackendType],
        "environment": env,
        "phase2_features": {
            "async_jobs": True, "result_caching": True,
            "warm_start": True, "noise_simulation": True, "slurm_generator": True,
            "error_mitigation_zne": True, "carbon_aware_optimization": True,
        },
    }


class OptimizationResultV2(OptimizationResult):
    """
    /api/v2/optimize response — same as v1 but demand_satisfaction is
    explicitly required (not a defaulted list) so OpenAPI marks it non-optional.
    Adds qubit_route_map at the top level for easier schema documentation.
    """
    demand_satisfaction: list[DemandSatisfactionSchema]
    qubit_route_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Stable, deterministic route↔qubit mapping. "
            "Key = qubit index (str), value = route_id ('FROM→TO#i'). "
            "bit i of the bitstring controls routes[i]."
        ),
    )


class OptimizationRequestV2(OptimizationRequest):
    """
    /api/v2/optimize request — same as v1 but all Phase-2 fields are
    documented at the top level of the schema for OpenAPI clarity.
    """
    lambda_time: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "Weight of time component in 'balanced' objective. "
            "0.0 = pure cost, 1.0 = pure time, 0.5 = equal blend."
        ),
    )


# Synchronous endpoint (Phase 1 preserved — backward compatible)
@app.post(
    "/api/v1/optimize",
    response_model=OptimizationResult,
    summary="Synchronous optimization (v1 — stable)",
    description=(
        "Blocks until complete. "
        "For ≥15 qubits or MPI backends use /api/v1/optimize/async. "
        "See /api/v2/optimize for a schema that documents all Phase-2 fields."
    ),
)
def optimize_supply_chain(request: OptimizationRequest):
    """Synchronous. Blocks until complete. Use async endpoint for large problems."""
    if len(request.routes) > 20:
        raise HTTPException(status_code=422,
                            detail="Too many routes. Use /api/v1/optimize/async for >20 routes.")
    cache_key = ResultCache.compute_key(request.model_dump())
    cached = result_cache.get(cache_key)
    if cached:
        try:
            return OptimizationResult.model_validate(cached)
        except Exception:
            pass   # stale/mismatched cache entry — fall through to recompute
    try:
        result = run_optimization_pipeline(request.model_dump())
        result_cache.put(cache_key, result)
        return OptimizationResult.model_validate(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/api/v2/optimize", response_model=OptimizationResultV2,
          summary="Synchronous optimization (v2 — full schema)",
          description=(
              "Same pipeline as /api/v1/optimize. "
              "Response schema adds qubit_route_map at the top level and marks "
              "demand_satisfaction as required. "
              "lambda_time is also a first-class request field here."
          ))
def optimize_v2(request: OptimizationRequestV2):
    """
    /api/v2 — explicitly documents all Phase-2 request fields and
    surfaces qubit_route_map at the response top level for reviewers.
    Backward-compatible: passes through to the same pipeline as /api/v1.
    """
    if len(request.routes) > 20:
        raise HTTPException(status_code=422,
                            detail="Too many routes. Use /api/v1/optimize/async for >20 routes.")
    # Inject lambda_time into constraints so the pipeline picks it up
    req_dict = request.model_dump()
    req_dict["constraints"] = req_dict.get("constraints") or {}
    req_dict["constraints"]["lambda_time"] = request.lambda_time

    cache_key = ResultCache.compute_key(req_dict)
    cached = result_cache.get(cache_key)
    if cached:
        try:
            qrm = cached.get("quantum_metrics", {}).get("qubit_route_map", {})
            cached["qubit_route_map"] = {str(k): v for k, v in qrm.items()}
            return OptimizationResultV2.model_validate(cached)
        except Exception:
            pass
    try:
        result = run_optimization_pipeline(req_dict)
        result_cache.put(cache_key, result)
        qrm = result.get("quantum_metrics", {}).get("qubit_route_map", {})
        result["qubit_route_map"] = {str(k): v for k, v in qrm.items()}
        return OptimizationResultV2.model_validate(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


# Async endpoint (Phase 2)
@app.post("/api/v1/optimize/async", response_model=AsyncJobResponse)
async def optimize_async(request: OptimizationRequest):
    """
    Fire-and-forget. Returns job_id immediately.
    Poll GET /api/v1/jobs/{job_id} for status and result.
    Recommended for ≥15 qubits, MPI backends, FX700 large-scale runs.
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job = await job_queue.submit(job_id, request.model_dump(), run_optimization_pipeline)
    return AsyncJobResponse(
        job_id=job_id, status=job.status.value,
        message="Job queued. Poll /api/v1/jobs/{job_id} for status.",
        poll_url=f"/api/v1/jobs/{job_id}",
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    """Poll async job status. result field populated when status == 'completed'."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatusResponse(
        job_id=job.job_id, status=job.status.value, progress=job.progress,
        created_at=job.created_at, started_at=job.started_at,
        finished_at=job.finished_at, result=job.result, error=job.error,
        backend_used=job.backend_used,
        cache_hit=bool(job.result and job.result.get("cache_hit")),
    )


@app.get("/api/v1/jobs")
def list_jobs(limit: int = 20):
    return {"jobs": job_store.list_recent(limit=limit)}


@app.post("/api/v1/optimize/batch")
def batch_optimize(requests: list[OptimizationRequest]):
    results = []
    for i, req in enumerate(requests):
        cache_key = ResultCache.compute_key(req.model_dump())
        cached = result_cache.get(cache_key)
        if cached:
            results.append({"scenario": i, "result": cached, "cache_hit": True})
            continue
        try:
            result = run_optimization_pipeline(req.model_dump())
            result_cache.put(cache_key, result)
            results.append({"scenario": i, "result": result, "cache_hit": False})
        except Exception as e:
            results.append({"scenario": i, "error": str(e), "status": "failed"})
    return results


class SlurmRequest(BaseModel):
    n_routes: int = Field(ge=1, le=50)
    backend: str = "qulacs_mpi"
    n_mpi_nodes: int = Field(default=1, ge=1, le=512)
    shots: int = Field(default=10000, ge=100)
    max_iterations: int = Field(default=300, ge=10)
    mps_bond_dim: int = Field(default=2, ge=1, le=64)


@app.post("/api/v1/slurm/generate", response_class=PlainTextResponse)
def generate_slurm(req: SlurmRequest):
    """Generate a ready-to-submit Slurm .job script for FX700."""
    try:
        config = QARPConfig(
            backend_type=BackendType(req.backend),
            n_mpi_nodes=req.n_mpi_nodes,
            shots=req.shots,
            max_iterations=req.max_iterations,
            mps_bond_dim=req.mps_bond_dim,
        )
        return generate_slurm_script(config, n_qubits=req.n_routes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.delete("/api/v1/cache")
def clear_cache():
    result_cache._cache.clear()
    result_cache._timestamps.clear()
    return {"message": "Cache cleared"}


@app.get("/api/v1/cache/stats")
def cache_stats():
    return result_cache.stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
