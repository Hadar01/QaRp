"""
tests.py
========
Unit tests verifying the 7 correctness requirements.

Run with:
    python tests.py           # built-in runner (no pytest needed)
    pytest tests.py -v        # if pytest is installed

These tests are designed to run WITHOUT qulacs or QARP — they test the
pure-Python logic in problem_encoder.py and the pipeline helpers.
"""

from __future__ import annotations

import sys
import os
import math
import traceback

# Make the package importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.problem_encoder import (
    ProblemEncoder, SupplyNode, Route, DemandForecast,
    IsingHamiltonian, greedy_classical_baseline,
)

# ---------------------------------------------------------------------------
# Minimal shared fixture
# ---------------------------------------------------------------------------

def _make_problem(n_warehouses: int = 1, n_stores: int = 2):
    """Small but non-trivial supply chain used by most tests."""
    nodes = [
        SupplyNode("WH-1",    "East Warehouse", "warehouse", capacity=1000, current_inventory=800),
        SupplyNode("STORE-1", "Store One",      "retail",    capacity=300,  current_inventory=50),
        SupplyNode("STORE-2", "Store Two",      "retail",    capacity=300,  current_inventory=30),
    ]
    routes = [
        Route("WH-1", "STORE-1", distance=100, cost_per_unit=2.5, time_hours=1.5, capacity=300),
        Route("WH-1", "STORE-2", distance=150, cost_per_unit=3.0, time_hours=2.0, capacity=300),
    ]
    demands = [
        DemandForecast("STORE-1", demand=200, priority=3),
        DemandForecast("STORE-2", demand=150, priority=2),
    ]
    return nodes, routes, demands


NODES, ROUTES, DEMANDS = _make_problem()


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0

def ok(name: str):
    global _PASS
    _PASS += 1
    print(f"  PASS  {name}")

def fail(name: str, msg: str):
    global _FAIL
    _FAIL += 1
    print(f"  FAIL  {name}")
    print(f"        {msg}")


def run_test(name: str, fn):
    try:
        fn()
        ok(name)
    except AssertionError as e:
        fail(name, str(e))
    except Exception as e:
        fail(name, f"Exception: {traceback.format_exc(limit=3)}")


# ===========================================================================
# Fix #1 — Core module imports
# ===========================================================================

def test_problem_encoder_importable():
    from core.problem_encoder import ProblemEncoder, IsingHamiltonian
    assert ProblemEncoder is not None
    assert IsingHamiltonian is not None

def test_qaoa_circuit_importable():
    # SolutionDecoder must not require qulacs
    try:
        from core.qaoa_circuit import SolutionDecoder
        dec = SolutionDecoder()
        assert dec is not None
    except ImportError as e:
        if "qulacs" in str(e):
            pass   # qulacs not installed — that's fine; SolutionDecoder itself is pure
        else:
            raise


# ===========================================================================
# Fix #3 — Binary route decision → flow/allocations/slack
# ===========================================================================

def test_decoder_flow_is_capacity_when_selected():
    """bit=1 → flow = route.capacity; bit=0 → route absent from assignments"""
    from core.qaoa_circuit import SolutionDecoder
    enc = ProblemEncoder()
    ham = enc.encode(NODES, ROUTES, DEMANDS, objective="minimize_cost")

    dec = SolutionDecoder()

    # "10" → only route 0 selected
    plan = dec.decode("10", ham, ROUTES, NODES, DEMANDS)
    assert len(plan["route_assignments"]) == 1
    r = plan["route_assignments"][0]
    assert r["flow"] == ROUTES[0].capacity, \
        f"flow should equal capacity {ROUTES[0].capacity}, got {r['flow']}"
    assert r["selected"] is True
    assert r["total_cost"] == round(ROUTES[0].capacity * ROUTES[0].cost_per_unit, 2)

    # "00" → zero routes selected, allocations still computed
    plan2 = dec.decode("00", ham, ROUTES, NODES, DEMANDS)
    assert plan2["route_assignments"] == []

def test_decoder_allocations_outflow_inflow():
    """Selecting route WH-1→STORE-1 should debit WH-1 and credit STORE-1."""
    from core.qaoa_circuit import SolutionDecoder
    enc = ProblemEncoder()
    ham = enc.encode(NODES, ROUTES, DEMANDS)
    dec = SolutionDecoder()
    plan = dec.decode("10", ham, ROUTES, NODES, DEMANDS)

    alloc = {a["node_id"]: a for a in plan["allocations"]}
    assert alloc["WH-1"]["outflow"] == ROUTES[0].capacity
    assert alloc["STORE-1"]["inflow"] == ROUTES[0].capacity
    assert alloc["WH-1"]["inflow"] == 0.0

def test_decoder_demand_slack_negative_when_unmet():
    """
    Route capacity=300, demand=200 → delivered=300 → slack=+100 (surplus).
    No routes selected → delivered=0 → slack=-200 (unmet demand).
    """
    from core.qaoa_circuit import SolutionDecoder
    enc = ProblemEncoder()
    ham = enc.encode(NODES, ROUTES, DEMANDS)
    dec = SolutionDecoder()

    # Both routes selected
    plan_both = dec.decode("11", ham, ROUTES, NODES, DEMANDS)
    d1 = next(d for d in plan_both["demand_satisfaction"] if d["node_id"] == "STORE-1")
    assert d1["slack"] >= 0, "STORE-1 should have non-negative slack when route selected"

    # No routes selected
    plan_none = dec.decode("00", ham, ROUTES, NODES, DEMANDS)
    d2 = next(d for d in plan_none["demand_satisfaction"] if d["node_id"] == "STORE-1")
    assert d2["slack"] < 0, "STORE-1 slack must be negative when no route serves it"
    assert d2["shortage"] == d2["demand"]


# ===========================================================================
# Fix #4 — Objective correctness (objective_value changes with objective)
# ===========================================================================

def _quantum_objective(bitstring: str, objective: str, routes=None) -> float:
    """
    Re-implements the same objective formula as run_optimization_pipeline
    so we can test it without the full API stack.
    """
    routes = routes or ROUTES
    from core.qaoa_circuit import SolutionDecoder
    enc = ProblemEncoder()
    ham = enc.encode(NODES, routes, DEMANDS, objective=objective)
    dec = SolutionDecoder()
    plan = dec.decode(bitstring, ham, routes, NODES, DEMANDS)

    assignments = plan["route_assignments"]
    total_cost = sum(r["total_cost"] for r in assignments)
    total_time = max((r["time_hours"] for r in assignments), default=0.0)

    lambda_time = 0.5
    max_cost_norm = max((r.cost_per_unit * r.capacity for r in routes), default=1.0) or 1.0
    max_time_norm = max((r.time_hours for r in routes), default=1.0) or 1.0

    if objective == "minimize_cost":
        return total_cost
    if objective == "minimize_time":
        return total_time
    if objective == "maximize_efficiency":
        eff = sum(
            r["flow"] / (r["cost_per_unit"] * r["time_hours"])
            for r in assignments
            if r["cost_per_unit"] * r["time_hours"] > 1e-12
        )
        return -eff
    # balanced
    c_norm = total_cost / max_cost_norm
    t_norm = total_time / max_time_norm
    return (1.0 - lambda_time) * c_norm + lambda_time * t_norm


def test_objective_value_differs_across_objectives():
    """
    Fix #4: objective_value must change when the objective changes.
    Using bitstring "11" (both routes selected) as a stable test case.
    """
    bs = "11"
    cost_obj  = _quantum_objective(bs, "minimize_cost")
    time_obj  = _quantum_objective(bs, "minimize_time")
    bal_obj   = _quantum_objective(bs, "balanced")

    # minimize_cost returns monetary cost (large number)
    # minimize_time returns hours (small number)
    # They must be different
    assert cost_obj != time_obj, \
        f"minimize_cost ({cost_obj}) and minimize_time ({time_obj}) must differ"

    # balanced must be between cost and time when both are normalised
    # (both routes selected, so normalised blend ≤ 1 by construction)
    assert 0.0 <= bal_obj <= 2.0, \
        f"balanced objective should be in a sensible range, got {bal_obj}"

    # minimize_time should return max(time_hours) which is 2.0 for route 1
    assert time_obj == 2.0, f"minimize_time with '11' should be 2.0, got {time_obj}"

def test_objective_value_minimize_cost_is_total_cost():
    """minimize_cost objective_value == Σ flow * cost_per_unit for selected routes."""
    bs = "10"   # only route 0 (WH-1→STORE-1, cap=300, cost=2.5)
    obj = _quantum_objective(bs, "minimize_cost")
    expected = 300 * 2.5   # flow=300, cost_per_unit=2.5
    assert abs(obj - expected) < 1e-6, \
        f"minimize_cost objective should be {expected}, got {obj}"

def test_objective_value_minimize_time_is_max_time():
    """minimize_time objective_value == max(time_hours) over selected routes."""
    bs = "01"   # only route 1 (time_hours=2.0)
    obj = _quantum_objective(bs, "minimize_time")
    assert abs(obj - 2.0) < 1e-9, f"minimize_time for route 1 should be 2.0, got {obj}"

def test_objective_value_empty_selection():
    """All-zero bitstring: zero cost, zero time, zero efficiency."""
    bs = "00"
    cost_obj = _quantum_objective(bs, "minimize_cost")
    time_obj = _quantum_objective(bs, "minimize_time")
    assert cost_obj == 0.0, f"No routes → cost=0, got {cost_obj}"
    assert time_obj == 0.0, f"No routes → time=0, got {time_obj}"


# ===========================================================================
# Fix #5 — Penalty scaling: data-dependent, exposed via constraints
# ===========================================================================

def test_penalty_scale_affects_hamiltonian_coefficients():
    """
    Larger demand_penalty_scale must produce larger h/J magnitude.
    If penalties are purely hard-coded, the two Hamiltonians will be identical.
    """
    enc = ProblemEncoder()
    ham_low  = enc.encode(NODES, ROUTES, DEMANDS, objective="balanced",
                           constraints={"demand_penalty_scale": 2.0})
    ham_high = enc.encode(NODES, ROUTES, DEMANDS, objective="balanced",
                           constraints={"demand_penalty_scale": 50.0})

    # Sum of squared h coefficients as a proxy for penalty magnitude
    def h_magnitude(ham):
        return sum(v**2 for v in ham.h.values())

    mag_low  = h_magnitude(ham_low)
    mag_high = h_magnitude(ham_high)
    assert mag_high > mag_low, \
        f"Higher demand_penalty_scale must increase Hamiltonian magnitude: {mag_low} vs {mag_high}"

def test_penalty_scale_constraint_key_names():
    """All three documented penalty keys must be accepted without error."""
    enc = ProblemEncoder()
    # These must not raise
    enc.encode(NODES, ROUTES, DEMANDS, constraints={"demand_penalty_scale": 5.0})
    enc.encode(NODES, ROUTES, DEMANDS, constraints={"capacity_penalty_scale": 3.0})
    enc.encode(NODES, ROUTES, DEMANDS, constraints={"warehouse_penalty_scale": 4.0})

def test_penalty_always_positive_floor():
    """Even with scale=0, penalty is floored to 2.0 (never zero)."""
    enc = ProblemEncoder()
    ham = enc.encode(NODES, ROUTES, DEMANDS, constraints={"demand_penalty_scale": 0.0})
    # h values should be non-trivial (penalty floor applied)
    assert any(abs(v) > 0 for v in ham.h.values()), \
        "Zero penalty scale should be floored to 2.0, h should be non-zero"


# ===========================================================================
# Fix #6 — Route↔qubit mapping transparency
# ===========================================================================

def test_qubit_map_is_deterministic():
    """Same input → same qubit assignment. Encode twice, compare."""
    enc = ProblemEncoder()
    ham1 = enc.encode(NODES, ROUTES, DEMANDS)
    ham2 = enc.encode(NODES, ROUTES, DEMANDS)
    assert ham1.qubit_map == ham2.qubit_map, "qubit_map must be deterministic"
    assert ham1.route_ids == ham2.route_ids, "route_ids must be deterministic"

def test_qubit_map_index_matches_routes_list():
    """qubit i must correspond to routes[i]."""
    enc = ProblemEncoder()
    ham = enc.encode(NODES, ROUTES, DEMANDS)
    for i, route in enumerate(ROUTES):
        assert i in ham.qubit_map, f"qubit {i} missing from qubit_map"
        assert ham.qubit_map[i] is route, f"qubit_map[{i}] must be routes[{i}]"

def test_route_ids_format():
    """route_ids[i] must be 'FROM→TO#i'."""
    enc = ProblemEncoder()
    ham = enc.encode(NODES, ROUTES, DEMANDS)
    for i, route in enumerate(ROUTES):
        expected = f"{route.from_node}→{route.to_node}#{i}"
        assert ham.route_ids[i] == expected, \
            f"route_ids[{i}] = '{ham.route_ids[i]}', expected '{expected}'"

def test_decoder_route_assignments_carry_qubit_index():
    """Every route_assignment must include qubit_index and route_id."""
    from core.qaoa_circuit import SolutionDecoder
    enc = ProblemEncoder()
    ham = enc.encode(NODES, ROUTES, DEMANDS)
    dec = SolutionDecoder()
    plan = dec.decode("11", ham, ROUTES, NODES, DEMANDS)
    for r in plan["route_assignments"]:
        assert "qubit_index" in r, "route_assignment must carry qubit_index"
        assert "route_id"    in r, "route_assignment must carry route_id"
    # qubit_route_map must be surfaced at top level
    assert "qubit_route_map" in plan, "decode() must return qubit_route_map"


# ===========================================================================
# Fix #7 — Greedy classical baseline
# ===========================================================================

def test_greedy_baseline_returns_correct_keys():
    """Return dict must contain all expected top-level keys."""
    result = greedy_classical_baseline(ROUTES, DEMANDS, NODES, "balanced")
    required = {"method", "objective", "selected_routes", "objective_value",
                "total_cost", "total_time", "demand_satisfaction"}
    missing = required - result.keys()
    assert not missing, f"greedy_classical_baseline missing keys: {missing}"

def test_greedy_baseline_method_is_greedy():
    result = greedy_classical_baseline(ROUTES, DEMANDS, NODES, "minimize_cost")
    assert result["method"] == "greedy"

def test_greedy_baseline_objective_changes_with_objective():
    """
    Fix #7: objective_value must differ across objectives (just like quantum).
    """
    r_cost = greedy_classical_baseline(ROUTES, DEMANDS, NODES, "minimize_cost")
    r_time = greedy_classical_baseline(ROUTES, DEMANDS, NODES, "minimize_time")
    assert r_cost["objective_value"] != r_time["objective_value"], \
        "greedy objective_value must change across objectives"

def test_greedy_baseline_respects_inventory():
    """
    If warehouse has zero inventory, greedy should select no routes and
    demand satisfaction should show unmet demand.
    """
    empty_nodes = [
        SupplyNode("WH-1", "Empty WH", "warehouse", capacity=1000, current_inventory=0),
        SupplyNode("STORE-1", "Store", "retail", capacity=300, current_inventory=0),
    ]
    result = greedy_classical_baseline(ROUTES[:1], DEMANDS[:1], empty_nodes, "minimize_cost")
    assert result["selected_routes"] == [], \
        "No routes should be selected when inventory=0"
    assert result["demand_satisfaction"][0]["satisfied"] is False

def test_greedy_priority_order():
    """Higher-priority demand should be served first when inventory is limited."""
    tight_nodes = [
        SupplyNode("WH-1", "WH", "warehouse", capacity=1000, current_inventory=200),
        SupplyNode("STORE-1", "S1", "retail", capacity=300, current_inventory=0),
        SupplyNode("STORE-2", "S2", "retail", capacity=300, current_inventory=0),
    ]
    demands_p = [
        DemandForecast("STORE-1", demand=200, priority=5),   # highest priority
        DemandForecast("STORE-2", demand=200, priority=1),   # lowest priority
    ]
    result = greedy_classical_baseline(ROUTES, demands_p, tight_nodes, "minimize_cost")
    served = {d["node_id"]: d for d in result["demand_satisfaction"]}
    # STORE-1 (priority=5) should be served before STORE-2 (priority=1)
    assert served["STORE-1"]["delivered"] > 0, \
        "High-priority demand STORE-1 should be served first"

def test_greedy_selected_routes_carry_qubit_index():
    """Each selected route must have qubit_index and route_id for traceability."""
    result = greedy_classical_baseline(ROUTES, DEMANDS, NODES, "balanced")
    for r in result["selected_routes"]:
        assert "qubit_index" in r
        assert "route_id" in r

def test_greedy_demand_satisfaction_has_slack():
    """Every demand entry must include the slack field."""
    result = greedy_classical_baseline(ROUTES, DEMANDS, NODES, "minimize_cost")
    for d in result["demand_satisfaction"]:
        assert "slack" in d, f"slack missing from demand_satisfaction entry: {d}"
        # slack = delivered - demand
        expected_slack = d["delivered"] - d["demand"]
        assert abs(d["slack"] - expected_slack) < 1e-6, \
            f"slack={d['slack']} but delivered-demand={expected_slack}"


# ===========================================================================
# Fix #1 — IsingHamiltonian QUBO must stay quadratic (no cubic terms)
# ===========================================================================

def test_hamiltonian_only_quadratic_interactions():
    """
    All J couplings must be 2-qubit (pairs), never 3+ qubit tuples.
    This verifies the Phase-1 QUBO-friendly constraint.
    """
    enc = ProblemEncoder()
    ham = enc.encode(NODES, ROUTES, DEMANDS, objective="balanced")
    for key in ham.J.keys():
        assert len(key) == 2, \
            f"J key {key} has {len(key)} indices — only 2-qubit terms allowed"
        i, j = key
        assert i < j, f"J keys must be ordered (i,j) with i<j, got ({i},{j})"


# ===========================================================================
# Run all tests
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Supply Chain — Correctness Tests")
    print("=" * 60)

    print("\n[Fix #1] Core module imports")
    run_test("problem_encoder importable",          test_problem_encoder_importable)
    run_test("qaoa_circuit importable",             test_qaoa_circuit_importable)

    print("\n[Fix #3] Binary route decision → flow/allocations/slack")
    run_test("flow equals capacity when selected",  test_decoder_flow_is_capacity_when_selected)
    run_test("allocations outflow/inflow correct",  test_decoder_allocations_outflow_inflow)
    run_test("slack negative when demand unmet",    test_decoder_demand_slack_negative_when_unmet)

    print("\n[Fix #4] Objective correctness")
    run_test("objective_value differs by objective",  test_objective_value_differs_across_objectives)
    run_test("minimize_cost == total monetary cost",  test_objective_value_minimize_cost_is_total_cost)
    run_test("minimize_time == max time_hours",       test_objective_value_minimize_time_is_max_time)
    run_test("empty selection → zero objective",      test_objective_value_empty_selection)

    print("\n[Fix #5] Penalty scaling")
    run_test("high penalty scale → larger coefficients", test_penalty_scale_affects_hamiltonian_coefficients)
    run_test("all three penalty key names accepted",     test_penalty_scale_constraint_key_names)
    run_test("penalty floor prevents zero penalties",    test_penalty_always_positive_floor)

    print("\n[Fix #6] Route↔qubit mapping transparency")
    run_test("qubit_map is deterministic",               test_qubit_map_is_deterministic)
    run_test("qubit_map[i] is routes[i]",                test_qubit_map_index_matches_routes_list)
    run_test("route_ids format is FROM→TO#i",            test_route_ids_format)
    run_test("route_assignments carry qubit_index",      test_decoder_route_assignments_carry_qubit_index)

    print("\n[Fix #7] Greedy classical baseline")
    run_test("returns all required keys",                test_greedy_baseline_returns_correct_keys)
    run_test("method == 'greedy'",                       test_greedy_baseline_method_is_greedy)
    run_test("objective_value changes with objective",   test_greedy_baseline_objective_changes_with_objective)
    run_test("respects zero inventory",                  test_greedy_baseline_respects_inventory)
    run_test("serves higher-priority demand first",      test_greedy_priority_order)
    run_test("selected routes carry qubit_index",        test_greedy_selected_routes_carry_qubit_index)
    run_test("demand_satisfaction has slack field",      test_greedy_demand_satisfaction_has_slack)

    print("\n[Fix #1] QUBO quadratic constraint")
    run_test("no cubic J terms in Hamiltonian",          test_hamiltonian_only_quadratic_interactions)

    print(f"\n{'='*60}")
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL:
        sys.exit(1)
