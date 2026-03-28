"""
problem_encoder.py
==================
PHASE 1 — Step 1: Problem Encoder

Translates the business problem (nodes, routes, demands) into an Ising
Hamiltonian that QAOA/VQE can minimise.

QUBIT MAPPING (fix #6 — transparency)
───────────────────────────────────────
Each possible *route* becomes exactly one qubit.  The mapping is
deterministic and ordered by the input list index:

    input routes[0]  →  qubit 0   (route_id = "WH-1→STORE-1#0")
    input routes[1]  →  qubit 1
    ...
    input routes[n-1] →  qubit n-1

After optimisation the quantum circuit is sampled to produce a bitstring
such as "101".  Reading left-to-right:
    bit 0 = '1'  →  routes[0] is SELECTED (ship full capacity)
    bit 1 = '0'  →  routes[1] is NOT selected
    bit 2 = '1'  →  routes[2] is SELECTED

The mapping is stored in IsingHamiltonian.qubit_map and also returned in
every route_assignment as {"qubit_index": i, "route_id": "..."}.

BINARY ROUTE DECISION → FLOW (fix #3)
───────────────────────────────────────
Each qubit is a binary decision variable  x_i ∈ {0, 1}:
    x_i = 1  → the route is *fully activated*: flow = route.capacity
    x_i = 0  → the route is *inactive*:         flow = 0

Post-processing (SolutionDecoder) maps the bitstring to:
    route_assignments[i].flow       = x_i * route.capacity
    route_assignments[i].total_cost = flow * cost_per_unit
    allocations[node].outflow       = Σ flow of outgoing selected routes
    allocations[node].inflow        = Σ flow of incoming selected routes
    demand_satisfaction[d].slack    = delivered - demand  (negative = shortage)

PENALTY SCALING (fix #5)
─────────────────────────
The Hamiltonian is:
    H = Σ_i c_i * x_i                    (objective cost terms)
      + λ_D * Σ_k P_k (D_k - Σ flow)²   (demand-satisfaction penalties)
      + λ_C * Σ_s (Σ outflow - I_s)²     (capacity/inventory penalties)

The constants λ_D and λ_C must be large enough that violations are never
energetically preferred over feasible solutions.  A principled default is:
    λ_D = demand_penalty_scale  * max(|c_i|)  (from constraints dict)
    λ_C = capacity_penalty_scale * max(|c_i|) (from constraints dict)

Both multipliers can be overridden via the API's `constraints` field:
    {"demand_penalty_scale": 5.0, "capacity_penalty_scale": 3.0}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Core domain data classes
# ---------------------------------------------------------------------------

@dataclass
class SupplyNode:
    id: str
    name: str
    type: str          # "warehouse" | "distribution_center" | "retail" | "supplier"
    capacity: float
    current_inventory: float
    coordinates: Optional[dict] = None


@dataclass
class Route:
    from_node: str
    to_node: str
    distance: float
    cost_per_unit: float
    time_hours: float
    capacity: float


@dataclass
class DemandForecast:
    node_id: str
    demand: float
    priority: int = 1    # 1–5; higher = more important to satisfy


@dataclass
class IsingHamiltonian:
    """
    Ising Hamiltonian  H_P = Σ_{i,j} J_ij Z_i Z_j + Σ_i h_i Z_i + offset

    Attributes
    ----------
    n_qubits    : number of qubits = number of route decisions
    J           : {(i,j): coupling_strength}  for ZZ interactions
    h           : {i: field_strength}          for single-qubit Z
    offset      : constant energy shift (bookkeeping)
    qubit_map   : {qubit_index: Route}  — the deterministic route↔qubit mapping
    route_ids   : {qubit_index: str}    — human-readable route identifier
    cost_scale  : the normalisation denominator used for h coefficients
                  (stored so objective_value can be de-normalised)
    """
    n_qubits:   int
    J:          dict = field(default_factory=dict)
    h:          dict = field(default_factory=dict)
    offset:     float = 0.0
    qubit_map:  dict = field(default_factory=dict)   # int → Route
    route_ids:  dict = field(default_factory=dict)   # int → str
    cost_scale: float = 1.0


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ProblemEncoder:
    """
    Converts supply chain data into an Ising Hamiltonian for QAOA/VQE.

    Parameters
    ----------
    penalty_weight : baseline multiplier applied to both demand and capacity
                     penalties.  Overridden per-type via the `constraints`
                     argument of `encode()`.
    """

    def __init__(self, penalty_weight: float = 10.0) -> None:
        self.penalty_weight = penalty_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        nodes: list[SupplyNode],
        routes: list[Route],
        demands: list[DemandForecast],
        objective: str = "balanced",
        lambda_time: float = 0.5,
        constraints: Optional[dict] = None,
        carbon_weight: float = 0.0,
    ) -> IsingHamiltonian:
        """
        Build and return an IsingHamiltonian for the given supply chain.

        Parameters
        ----------
        nodes      : supply network nodes
        routes     : possible shipping routes; routes[i] maps to qubit i
        demands    : demand forecasts per destination node
        objective  : "minimize_cost" | "minimize_time" |
                     "maximize_efficiency" | "balanced"
        lambda_time: weight of time component in "balanced" objective (0–1)
        constraints: optional dict to override penalty scales, e.g.
                     {"demand_penalty_scale": 5.0,
                      "capacity_penalty_scale": 3.0,
                      "lambda_time": 0.4}
        carbon_weight: weight for carbon-aware optimization (0.0–1.0).
                       When > 0, adds CO2 emission cost to objective terms.

        Returns
        -------
        IsingHamiltonian  (qubit_map, route_ids, and cost_scale populated)
        """
        constraints = constraints or {}
        if "lambda_time" in constraints:
            lambda_time = float(constraints["lambda_time"])
        if "carbon_weight" in constraints:
            carbon_weight = float(constraints["carbon_weight"])

        n = len(routes)

        # ── Step 1: deterministic qubit ↔ route mapping (fix #6) ──────────
        # Order is fixed by the input list index — never changes for the
        # same input, so callers can rely on bit i ↔ routes[i].
        qubit_map: dict[int, Route] = {i: r for i, r in enumerate(routes)}
        route_ids: dict[int, str]   = {
            i: f"{r.from_node}→{r.to_node}#{i}"
            for i, r in enumerate(routes)
        }

        # ── Step 2: objective cost coefficients ────────────────────────────
        raw_costs = self._compute_cost_coefficients(routes, objective, lambda_time)

        # ── Step 3: normalise to keep QAOA landscape numerically stable ────
        max_abs = max((abs(c) for c in raw_costs), default=1.0)
        if max_abs < 1e-12:
            max_abs = 1.0
        norm_costs = [c / max_abs for c in raw_costs]
        cost_scale = max_abs   # stored so we can de-normalise later

        # ── Step 3b: compute penalty scales early (needed for cost balance) ─
        demand_scale   = float(constraints.get("demand_penalty_scale",
                                                self.penalty_weight))
        capacity_scale = float(constraints.get(
            "capacity_penalty_scale",
            constraints.get("warehouse_penalty_scale", self.penalty_weight)
        ))
        demand_scale   = max(demand_scale,   2.0)
        capacity_scale = max(capacity_scale, 2.0)

        # ── Step 3c: amplify cost terms to compete with penalty terms ──────
        # Without amplification, cost terms are O(1) while penalty terms
        # are O(penalty_weight), making the energy gap between feasible
        # solutions negligibly small.  cost_objective_weight scales up the
        # objective relative to penalties so QAOA can actually resolve it.
        cost_objective_weight = float(constraints.get("cost_objective_weight",
                                                       max(demand_scale, capacity_scale)))
        norm_costs = [c * cost_objective_weight for c in norm_costs]

        # ── Step 4: QUBO → Ising conversion for linear cost terms ─────────
        # QUBO: min Σ c_i x_i,  x_i ∈ {0,1}
        # Ising substitution: x_i = (1 − Z_i)/2
        # → c_i * x_i  =  (c_i/2)(1 − Z_i)
        # → h contribution: −c_i/2  (Z coefficient)
        # → offset contribution: +c_i/2
        h: dict[int, float] = {}
        offset: float = 0.0
        for i, c in enumerate(norm_costs):
            h[i] = h.get(i, 0.0) - c / 2.0
            offset += c / 2.0

        # ── Step 5: demand-satisfaction penalty terms ──────────────────────
        # (penalty scales already computed in Step 3b above)
        # Without this, raw demands (e.g. 200) and capacities (300) produce
        # penalty offsets in the hundreds of thousands, making the Hamiltonian
        # landscape flat and causing the optimizer to converge to "00" (nothing
        # selected) as a spurious energy minimum.
        max_demand = max((d.demand for d in demands), default=1.0) or 1.0
        max_cap    = max((r.capacity for r in routes), default=1.0) or 1.0
        norm_factor = max(max_demand, max_cap)   # single shared normalization

        J: dict[tuple, float] = {}
        J, h, offset = self._add_demand_penalties(
            routes, demands, J=J, h=h, offset=offset,
            lam_scale=demand_scale,
            norm_factor=norm_factor,
        )

        # ── Step 7: capacity / inventory penalty terms ─────────────────────
        J, h, offset = self._add_capacity_penalties(
            nodes, routes, J=J, h=h, offset=offset,
            lam_scale=capacity_scale,
            norm_factor=norm_factor,
        )

        return IsingHamiltonian(
            n_qubits=n,
            J=J,
            h=h,
            offset=offset,
            qubit_map=qubit_map,
            route_ids=route_ids,
            cost_scale=cost_scale,
        )

    # ------------------------------------------------------------------
    # Cost coefficient computation (fix #4 helper)
    # ------------------------------------------------------------------

    # CO2 emission factors (kg CO2 per ton-km by transport mode)
    _EMISSION_FACTORS = {
        "road": 0.062, "rail": 0.022, "sea": 0.008, "air": 0.602,
    }

    @staticmethod
    def _estimate_mode(distance: float) -> str:
        if distance < 300: return "road"
        if distance < 1500: return "rail"
        if distance < 8000: return "sea"
        return "air"

    def _compute_cost_coefficients(
        self,
        routes: list[Route],
        objective: str,
        lambda_time: float,
        carbon_weight: float = 0.0,
    ) -> list[float]:
        """
        Return a scalar cost coefficient for each route.

        These are the *raw* (un-normalised) values used to form the
        linear terms of the Hamiltonian.  The chosen objective determines
        what quantity QAOA tries to minimise.

        Scaling note (fix #4):
          - minimize_cost  : cost_per_unit × capacity  [monetary units]
          - minimize_time  : time_hours                 [hours]
          - maximize_eff.  : −capacity/(cost×time)      [dimensionless, negated]
          - balanced       : (1−λ)×cost_term + λ×time_term
                             Both terms are normalised to their own maxima
                             before blending so neither dominates by scale.

        Carbon-aware mode (carbon_weight > 0):
          Adds CO2 emission cost to the objective so the quantum optimizer
          directly minimises carbon footprint alongside logistics cost.
          CO2 = emission_factor(mode) × (capacity/1000) × distance  [kg]
        """
        cost_terms = [r.cost_per_unit * r.capacity for r in routes]
        time_terms = [r.time_hours for r in routes]

        # Carbon footprint per route (kg CO2)
        carbon_terms = []
        for r in routes:
            mode = self._estimate_mode(r.distance)
            co2 = self._EMISSION_FACTORS[mode] * (r.capacity / 1000.0) * r.distance
            carbon_terms.append(co2)

        max_cost = max((abs(c) for c in cost_terms), default=1.0) or 1.0
        max_time = max((abs(t) for t in time_terms), default=1.0) or 1.0
        max_carbon = max((abs(c) for c in carbon_terms), default=1.0) or 1.0

        coefficients: list[float] = []
        for r, c_raw, t_raw, co2_raw in zip(routes, cost_terms, time_terms, carbon_terms):
            if objective == "minimize_cost":
                coeff = c_raw

            elif objective == "minimize_time":
                coeff = t_raw

            elif objective == "maximize_efficiency":
                denom = r.cost_per_unit * r.time_hours
                coeff = -(r.capacity / denom) if denom > 1e-12 else 0.0

            else:  # "balanced"
                # Normalise each dimension to [0,1] before weighting so
                # the blend is truly balanced regardless of unit difference.
                # lambda_time=0.5 → equal weight; 0=pure cost, 1=pure time.
                c_norm = c_raw / max_cost
                t_norm = t_raw / max_time
                coeff = (1.0 - lambda_time) * c_norm + lambda_time * t_norm

            # Add carbon penalty when carbon_weight > 0
            if carbon_weight > 0:
                co2_norm = co2_raw / max_carbon
                coeff = (1.0 - carbon_weight) * (coeff / max(abs(coeff), 1e-12) if abs(coeff) > 1e-12 else 0) + carbon_weight * co2_norm

            coefficients.append(coeff)

        return coefficients

    # ------------------------------------------------------------------
    # Demand-satisfaction penalties (fix #5: data-dependent lambda)
    # ------------------------------------------------------------------

    def _add_demand_penalties(
        self,
        routes: list[Route],
        demands: list[DemandForecast],
        J: dict,
        h: dict,
        offset: float,
        lam_scale: float,
        norm_factor: float = 1.0,
    ) -> tuple[dict, dict, float]:
        """
        For each demand node k (demand D_k, priority P_k), add Ising terms
        that reward selecting routes delivering to k and penalise shortfalls.

        Formulation
        -----------
        We use TWO complementary terms:

        1. Quadratic shortfall penalty  lam * (D_n - sum cap_in * x_i)^2
           Converted to Ising via x_i = (1-Z_i)/2 → a_i = cap_in/2
           where D_n = D/norm_factor, cap_in = cap/norm_factor.

           This term is symmetric: when D_n = sum_a (demand == cap/2),
           selecting or not gives equal penalty → degeneracy.

        2. Direct delivery reward  lam * D_n * sum cap_in * x_i
           Converted to Ising: lam * D_n * a_i * (1 - Z_i)
           h[i] += lam * D_n * a_i    (positive h → prefers Z=-1 → x=1)
           offset -= lam * D_n * a_i  (constant adjustment)

           This term ALWAYS drives routes toward selection, breaking the
           degeneracy that occurs when demand equals capacity.

        Together these produce a Hamiltonian whose global minimum is at
        the bitstring where all demand is satisfied (or as much as possible).
        All terms remain QUADRATIC in x (fix #1 constraint satisfied).
        """
        demand_map = {d.node_id: d for d in demands}

        for node_id, dem in demand_map.items():
            D_n = dem.demand / norm_factor          # normalized demand
            lam = lam_scale * dem.priority

            delivering = [(i, r) for i, r in enumerate(routes) if r.to_node == node_id]
            if not delivering:
                continue

            # Normalized Ising-space contributions: a_i = cap_in / 2
            contribs = [(i, (r.capacity / norm_factor) / 2.0) for i, r in delivering]
            sum_a    = sum(a for _, a in contribs)
            R = D_n - sum_a   # residual when no routes selected (Ising: all Z=+1)

            # ── Term 1: quadratic penalty ─────────────────────────────────
            # Constant: lam * R^2
            offset += lam * R ** 2

            # Linear: lam * 2 * R * a_i * Z_i  (positive = prefers Z=-1)
            for qi, a_i in contribs:
                h[qi] = h.get(qi, 0.0) + lam * 2.0 * R * a_i

            # Quadratic: lam * 2 * a_i * a_j * Z_i * Z_j
            for p in range(len(contribs)):
                for q in range(p + 1, len(contribs)):
                    qi, a_i = contribs[p]
                    qj, a_j = contribs[q]
                    key = (min(qi, qj), max(qi, qj))
                    J[key] = J.get(key, 0.0) + lam * 2.0 * a_i * a_j

            # Diagonal (i==i): lam * a_i^2 → goes to offset (Z_i^2 = 1)
            for qi, a_i in contribs:
                offset += lam * a_i ** 2

            # ── Term 2: direct delivery reward (breaks degeneracy) ────────
            # Adds h[i] += lam * D_n * a_i  so Z_i=-1 (selected) always wins
            for qi, a_i in contribs:
                h[qi] = h.get(qi, 0.0) + lam * D_n * a_i
            offset -= lam * D_n * sum_a   # offset adjustment for the constant part

        return J, h, offset

    # ------------------------------------------------------------------
    # Capacity / inventory penalties (fix #5)
    # ------------------------------------------------------------------

    def _add_capacity_penalties(
        self,
        nodes: list[SupplyNode],
        routes: list[Route],
        J: dict,
        h: dict,
        offset: float,
        lam_scale: float,
        norm_factor: float = 1.0,
    ) -> tuple[dict, dict, float]:
        """
        For each source node s (inventory I_s), penalise total outflow
        exceeding available inventory:

            λ_C * (Σ_{routes from s} cap_i * x_i − I_s)²

        Only applied when maximum possible outflow > I_s (soft constraint).
        Expansion identical to demand penalties; all terms remain quadratic.
        """
        node_map = {nd.id: nd for nd in nodes}

        for node_id, node in node_map.items():
            # Normalize inventory and capacities to O(1) (same norm_factor as demand).
            I_s  = node.current_inventory / norm_factor
            lam  = lam_scale

            outgoing = [(i, r) for i, r in enumerate(routes) if r.from_node == node_id]
            if not outgoing:
                continue

            contribs = [(i, (r.capacity / norm_factor) / 2.0) for i, r in outgoing]
            sum_a    = sum(a for _, a in contribs)

            # Skip if max possible outflow ≤ inventory (no violation possible)
            if sum_a <= I_s / 2.0:
                continue

            excess = sum_a - I_s / 2.0

            offset += lam * excess ** 2

            for qi, a_i in contribs:
                # Capacity penalty: lam*(excess - Σ a_i Z_i)^2
                # Linear term: -2*lam*excess*a_i (negative sign because
                # selecting a route reduces excess, which is desired).
                h[qi] = h.get(qi, 0.0) - lam * 2.0 * excess * a_i

            for p_idx in range(len(contribs)):
                for q_idx in range(p_idx + 1, len(contribs)):
                    qi, a_i = contribs[p_idx]
                    qj, a_j = contribs[q_idx]
                    key = (min(qi, qj), max(qi, qj))
                    J[key] = J.get(key, 0.0) + lam * 2.0 * a_i * a_j

            # Diagonal terms: Z_i^2 = 1, so lam * a_i^2 goes to offset
            for _, a_i in contribs:
                offset += lam * a_i ** 2

        return J, h, offset


# ---------------------------------------------------------------------------
# Greedy classical baseline (fix #7)
# ---------------------------------------------------------------------------

def greedy_classical_baseline(
    routes: list[Route],
    demands: list[DemandForecast],
    nodes: list[SupplyNode],
    objective: str = "balanced",
    lambda_time: float = 0.5,
) -> dict:
    """
    Greedy baseline: for each demand node, choose the cheapest (or fastest,
    or best-balanced) feasible inbound route that has available inventory,
    respecting source node inventory limits.

    Returns a dict with the same keys as run_optimization_pipeline's
    classical_comparison block:
        {
            "method":            "greedy",
            "objective":         objective,
            "selected_routes":   [...],  # list of (qubit_idx, Route)
            "objective_value":   float,
            "total_cost":        float,
            "total_time":        float,
            "demand_satisfaction": [...],
        }

    This is a *meaningful* baseline — it reflects what a reasonable human
    scheduler would do, not the trivially bad "activate every route" sum.
    """
    # Track remaining inventory per source node
    inventory: dict[str, float] = {n.id: n.current_inventory for n in nodes}

    # Sort routes by "score" matching the objective
    def route_score(r: Route) -> float:
        if objective == "minimize_cost":
            return r.cost_per_unit
        if objective == "minimize_time":
            return r.time_hours
        if objective == "maximize_efficiency":
            denom = r.cost_per_unit * r.time_hours
            return -(r.capacity / denom) if denom > 1e-12 else float("inf")
        # balanced: normalise then blend
        max_c = max((x.cost_per_unit for x in routes), default=1.0) or 1.0
        max_t = max((x.time_hours for x in routes), default=1.0) or 1.0
        return (1 - lambda_time) * r.cost_per_unit / max_c + lambda_time * r.time_hours / max_t

    # Index routes by destination
    routes_by_dest: dict[str, list[tuple[int, Route]]] = {}
    for idx, r in enumerate(routes):
        routes_by_dest.setdefault(r.to_node, []).append((idx, r))

    for dest in routes_by_dest:
        routes_by_dest[dest].sort(key=lambda ir: route_score(ir[1]))

    selected: list[tuple[int, Route]] = []   # (qubit_index, Route)
    delivered: dict[str, float] = {}

    # Sort demands by priority descending (serve most important first)
    for dem in sorted(demands, key=lambda d: -d.priority):
        remaining_demand = dem.demand
        for idx, r in routes_by_dest.get(dem.node_id, []):
            if remaining_demand <= 0:
                break
            avail = inventory.get(r.from_node, 0.0)
            if avail <= 0:
                continue
            flow = min(r.capacity, remaining_demand, avail)
            if flow <= 0:
                continue
            # Activate this route (partial flow — greedy may activate same route
            # multiple times if capacity < demand; we treat it as one selection)
            inventory[r.from_node] -= flow
            remaining_demand       -= flow
            delivered[dem.node_id]  = delivered.get(dem.node_id, 0.0) + flow
            if not any(qi == idx for qi, _ in selected):
                selected.append((idx, r))

    total_cost = sum(r.cost_per_unit * r.capacity for _, r in selected)
    total_time = max((r.time_hours for _, r in selected), default=0.0)

    if objective == "minimize_cost":
        obj_value = total_cost
    elif objective == "minimize_time":
        obj_value = total_time
    elif objective == "maximize_efficiency":
        obj_value = -sum(
            r.capacity / (r.cost_per_unit * r.time_hours)
            for _, r in selected
            if r.cost_per_unit * r.time_hours > 1e-12
        )
    else:  # balanced
        max_c = max((r.cost_per_unit for r in routes), default=1.0) or 1.0
        max_t = max((r.time_hours for r in routes), default=1.0) or 1.0
        obj_value = sum(
            (1 - lambda_time) * r.cost_per_unit / max_c
            + lambda_time * r.time_hours / max_t
            for _, r in selected
        )

    demand_satisfaction = []
    for dem in demands:
        actual   = delivered.get(dem.node_id, 0.0)
        shortage = max(0.0, dem.demand - actual)
        demand_satisfaction.append({
            "node_id":   dem.node_id,
            "demand":    dem.demand,
            "delivered": round(actual, 4),
            "shortage":  round(shortage, 4),
            "slack":     round(actual - dem.demand, 4),   # negative = unmet
            "satisfied": shortage < 1e-6,
        })

    return {
        "method":             "greedy",
        "objective":          objective,
        "selected_routes":    [{"qubit_index": qi, "route_id": f"{r.from_node}→{r.to_node}#{qi}"}
                                for qi, r in selected],
        "objective_value":    round(obj_value, 4),
        "total_cost":         round(total_cost, 4),
        "total_time":         round(total_time, 4),
        "demand_satisfaction": demand_satisfaction,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
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
    ham = encoder.encode(nodes, routes, demands, objective="balanced")

    print(f"Qubits: {ham.n_qubits}")
    print(f"Route IDs: {ham.route_ids}")
    print(f"h fields: {ham.h}")
    print(f"J couplings: {ham.J}")
    print(f"cost_scale: {ham.cost_scale}")

    # Test greedy baseline
    baseline = greedy_classical_baseline(routes, demands, nodes, "minimize_cost")
    print(f"\nGreedy baseline (minimize_cost):")
    print(f"  objective_value: {baseline['objective_value']}")
    print(f"  selected: {[r['route_id'] for r in baseline['selected_routes']]}")
    for d in baseline["demand_satisfaction"]:
        print(f"  {d['node_id']}: delivered={d['delivered']}, slack={d['slack']}")

    # Test data-dependent penalties from constraints
    ham2 = encoder.encode(nodes, routes, demands, objective="balanced",
                           constraints={"demand_penalty_scale": 20.0})
    print(f"\nWith demand_penalty_scale=20 → h: {ham2.h}")
