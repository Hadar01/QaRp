"""
run_all_tests.py
================
Final comprehensive test suite for Fujitsu Challenge submission.
Runs ALL verifications and benchmarks, saving results to log files.

Usage on FX700:
    export LD_LIBRARY_PATH=/home/share/developer/gcc-14.1.0/lib64:$LD_LIBRARY_PATH
    source ~/QARPdemo/venv/bin/activate
    cd ~/QARPdemo/QaRp
    mpirun -np 1 python run_all_tests.py 2>&1 | tee final_results.log
"""

import json
import time
import sys
import numpy as np
import traceback

# ── Core imports ──────────────────────────────────────────────────────────
from core.problem_encoder import (
    ProblemEncoder, SupplyNode, Route, DemandForecast,
)
from core.qaoa_circuit import QAOACircuit, VQEOptimizer, ExactGroundStateOptimizer

# ── QARP imports ──────────────────────────────────────────────────────────
try:
    from qarp.engines import QulacsEngine
    QARP_OK = True
    print("[OK] QARP QulacsEngine available")
except ImportError:
    QARP_OK = False
    print("[SKIP] QARP not available")

# ── Tenet imports ─────────────────────────────────────────────────────────
try:
    from pytket.extensions.tenet import (
        InnerProductBackend, MPSInnerProductBackend, Config,
    )
    from pytket.circuit import Circuit, Qubit
    from pytket.pauli import Pauli, QubitPauliString
    from pytket.utils import QubitPauliOperator
    TENET_OK = True
    print("[OK] pytket-tenet available")
except ImportError as e:
    TENET_OK = False
    print(f"[SKIP] pytket-tenet not available: {e}")


def load_problem(path):
    with open(path) as f:
        req = json.load(f)
    nodes = [SupplyNode(n["id"], n["name"], n["type"], n["capacity"], n["current_inventory"])
             for n in req["nodes"]]
    routes = [Route(r["from_node"], r["to_node"], r["distance"],
                    r["cost_per_unit"], r["time_hours"], r["capacity"])
              for r in req["routes"]]
    demands = [DemandForecast(d["node_id"], d["demand"], d.get("priority", 1))
               for d in req["demands"]]
    constraints = req.get("constraints", {})
    encoder = ProblemEncoder(penalty_weight=constraints.get("demand_penalty_scale", 10.0))
    ham = encoder.encode(nodes, routes, demands,
                         objective=req.get("objective", "minimize_cost"),
                         constraints=constraints)
    return nodes, routes, demands, ham, req


def ising_energy(ham, bits):
    spins = np.array([1 - 2 * int(b) for b in bits])
    energy = sum(h_val * spins[i] for i, h_val in ham.h.items())
    for (qi, qj), J_val in ham.J.items():
        energy += J_val * spins[qi] * spins[qj]
    return float(energy + ham.offset)


def compute_cost(bitstring, routes):
    return sum(routes[i].cost_per_unit * routes[i].capacity
               for i, b in enumerate(bitstring) if b == '1')


def ham_to_qubit_pauli_operator(ham):
    terms = {QubitPauliString(): ham.offset}
    for i, h_val in ham.h.items():
        if abs(h_val) > 1e-15:
            terms[QubitPauliString({Qubit(i): Pauli.Z})] = h_val
    for (i, j), J_val in ham.J.items():
        if abs(J_val) > 1e-15:
            terms[QubitPauliString({Qubit(i): Pauli.Z, Qubit(j): Pauli.Z})] = J_val
    return QubitPauliOperator(terms)


def run_rqaoa(ham, p_layers=2, threshold=3, max_iter=100, n_restarts=3):
    """Run RQAOA using Qulacs (fast)."""
    from core.advanced_optimizers import (
        _reduce_hamiltonian_pair, _reduce_hamiltonian_single,
        _build_compact_hamiltonian, _compute_correlations,
    )
    n_original = ham.n_qubits
    h, J, offset = dict(ham.h), dict(ham.J), ham.offset
    active = list(range(n_original))
    substitutions = []

    while len(active) > threshold:
        n_active = len(active)
        compact_ham = _build_compact_hamiltonian(h, J, offset, active)
        p_use = min(p_layers, max(1, n_active - 1))
        qaoa = QAOACircuit(compact_ham, p_layers=p_use)
        vqe = VQEOptimizer(qaoa, max_iterations=max_iter, n_restarts=n_restarts)
        res = vqe.optimize()
        single, pair = _compute_correlations(qaoa, res["best_params"])
        idx_to_orig = {ci: active[ci] for ci in range(n_active)}

        best_strength, best_action = -1.0, None
        for ci, val in single.items():
            if abs(val) > best_strength:
                best_strength = abs(val)
                best_action = ("single", idx_to_orig[ci], 1 if val > 0 else -1)
        for (ci, cj), val in pair.items():
            if abs(val) > best_strength:
                best_strength = abs(val)
                best_action = ("pair", idx_to_orig[cj], idx_to_orig[ci], 1 if val > 0 else -1)

        if best_action[0] == "single":
            _, q, s = best_action
            h, J, offset, active = _reduce_hamiltonian_single(h, J, offset, active, q, s)
            substitutions.append(best_action)
        else:
            _, elim, ref, sign = best_action
            h, J, offset, active = _reduce_hamiltonian_pair(h, J, offset, active, elim, ref, sign)
            substitutions.append(best_action)
        print(f"    Reduction {len(substitutions)}: {n_active}->{len(active)}q (|s|={best_strength:.4f})")

    compact_ham = _build_compact_hamiltonian(h, J, offset, active)
    exact = ExactGroundStateOptimizer(compact_ham)
    exact_res = exact.optimize()

    compact_bits = [int(b) for b in exact_res["best_bitstring"]]
    spins = {active[ci]: 1 - 2 * compact_bits[ci] for ci in range(len(active))}
    for sub in reversed(substitutions):
        if sub[0] == "single":
            spins[sub[1]] = sub[2]
        else:
            spins[sub[1]] = sub[3] * spins[sub[2]]

    bits = [0 if spins.get(i, 1) == 1 else 1 for i in range(n_original)]
    bitstring = "".join(str(b) for b in bits)
    return {"best_energy": ising_energy(ham, bits), "best_bitstring": bitstring,
            "n_reductions": len(substitutions)}


def tenet_verify(ham, bitstring, use_mps=False, bond_dim=64):
    """Verify a solution's energy through Tenet tensor network."""
    ham_op = ham_to_qubit_pauli_operator(ham)
    n = ham.n_qubits

    if use_mps:
        backend = MPSInnerProductBackend(Config(mps_bond_dim=bond_dim))
    else:
        backend = InnerProductBackend()

    circ = Circuit(n)
    for i, b in enumerate(bitstring):
        if b == '1':
            circ.X(i)

    compiled = backend.get_compiled_circuit(circ)
    t0 = time.time()
    energy = float(backend.get_operator_expectation_value(compiled, ham_op).real)
    t = time.time() - t0
    return energy, t


# ══════════════════════════════════════════════════════════════════════════
#  MAIN TEST SUITE
# ══════════════════════════════════════════════════════════════════════════

def main():
    results = {}

    problems = [
        ("data/request_advantage.json", "6q Supply Chain"),
        ("data/request_12q.json", "12q Supply Chain"),
        ("data/request_36q.json", "36q Supply Chain"),
    ]

    print("\n" + "=" * 80)
    print("  FINAL COMPREHENSIVE TEST SUITE — Fujitsu Challenge 2025-26")
    print(f"  Team G-147 | QaRp Quantum Supply Chain Optimization")
    print(f"  QARP v0.4.4 | pytket-tenet v0.5.0 | Qulacs (MPI)")
    print("=" * 80)

    for path, name in problems:
        print(f"\n{'=' * 80}")
        print(f"  TEST: {name} ({path})")
        print(f"{'=' * 80}")

        try:
            nodes, routes, demands, ham, req = load_problem(path)
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            continue

        n = ham.n_qubits
        print(f"  Qubits: {n}, Routes: {len(routes)}")

        prob_results = {"qubits": n, "routes": len(routes)}

        # ── 1. RQAOA via Qulacs ──────────────────────────────────────────
        print(f"\n  [1] RQAOA via Qulacs ({n}q)...")
        try:
            t0 = time.time()
            rqaoa_res = run_rqaoa(ham, p_layers=2, threshold=3,
                                   max_iter=100, n_restarts=3)
            rqaoa_time = time.time() - t0
            rqaoa_cost = compute_cost(rqaoa_res["best_bitstring"], routes)

            if n <= 20:
                exact = ExactGroundStateOptimizer(ham)
                exact_res = exact.optimize()
                ar = rqaoa_res["best_energy"] / exact_res["best_energy"] if exact_res["best_energy"] < 0 else "N/A"
            else:
                ar = "N/A"

            print(f"      Energy:     {rqaoa_res['best_energy']:.4f}")
            print(f"      Cost:       ${rqaoa_cost:,.0f}")
            print(f"      AR:         {ar:.4f}" if isinstance(ar, float) else f"      AR:         {ar}")
            print(f"      Bitstring:  {rqaoa_res['best_bitstring']}")
            print(f"      Reductions: {rqaoa_res['n_reductions']}")
            print(f"      Time:       {rqaoa_time:.1f}s")

            prob_results["rqaoa"] = {
                "energy": rqaoa_res["best_energy"],
                "cost": rqaoa_cost,
                "ar": ar if isinstance(ar, float) else None,
                "bitstring": rqaoa_res["best_bitstring"],
                "time": rqaoa_time,
            }
        except Exception as e:
            print(f"      [ERROR] {e}")
            traceback.print_exc()
            prob_results["rqaoa"] = {"error": str(e)}

        # ── 2. Tenet Verification (General TN) ───────────────────────────
        if TENET_OK and "rqaoa" in prob_results and "bitstring" in prob_results.get("rqaoa", {}):
            bitstring = prob_results["rqaoa"]["bitstring"]

            if n <= 20:
                print(f"\n  [2] Tenet Verification — General TN ({n}q)...")
                try:
                    tenet_e, tenet_t = tenet_verify(ham, bitstring, use_mps=False)
                    direct_e = prob_results["rqaoa"]["energy"]
                    match = abs(tenet_e - direct_e) < 0.01
                    print(f"      Tenet energy:  {tenet_e:.4f}")
                    print(f"      Direct energy: {direct_e:.4f}")
                    print(f"      MATCH:         {match}")
                    print(f"      Time:          {tenet_t:.1f}s")
                    prob_results["tenet_general"] = {
                        "energy": tenet_e, "time": tenet_t, "match": match
                    }
                except Exception as e:
                    print(f"      [ERROR] {e}")
                    traceback.print_exc()

            # ── 3. Tenet Verification (MPS) ───────────────────────────────
            print(f"\n  [3] Tenet Verification — MPS (bond_dim=64, {n}q)...")
            try:
                tenet_mps_e, tenet_mps_t = tenet_verify(
                    ham, bitstring, use_mps=True, bond_dim=64)
                direct_e = prob_results["rqaoa"]["energy"]
                match = abs(tenet_mps_e - direct_e) < 0.01
                print(f"      MPS energy:    {tenet_mps_e:.4f}")
                print(f"      Direct energy: {direct_e:.4f}")
                print(f"      MATCH:         {match}")
                print(f"      Time:          {tenet_mps_t:.1f}s")
                prob_results["tenet_mps"] = {
                    "energy": tenet_mps_e, "time": tenet_mps_t,
                    "match": match, "bond_dim": 64
                }
            except Exception as e:
                print(f"      [ERROR] {e}")
                traceback.print_exc()

        results[path] = prob_results

    # ── 4. Full benchmark suite on 6q ─────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  [4] Full 12-Algorithm Benchmark (6q)")
    print(f"{'=' * 80}")
    import subprocess
    try:
        cmd = ["mpirun", "-np", "1", "python", "benchmark_suite.py",
               "-i", "data/request_advantage.json",
               "-a", "exact,rqaoa,scalable_rqaoa,qaoa,warm_start"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.stderr:
            print(f"  STDERR: {result.stderr[-500:]}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ── 5. QARP Benchmark ─────────────────────────────────────────────────
    if QARP_OK:
        print(f"\n{'=' * 80}")
        print(f"  [5] QARP Benchmark (6q)")
        print(f"{'=' * 80}")
        try:
            cmd = ["mpirun", "-np", "1", "python", "qarp_benchmark.py",
                   "-i", "data/request_advantage.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        except Exception as e:
            print(f"  [ERROR] {e}")

    # ── Save results ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")

    for path, res in results.items():
        n = res.get("qubits", "?")
        print(f"\n  {path} ({n}q):")
        if "rqaoa" in res and "energy" in res["rqaoa"]:
            r = res["rqaoa"]
            print(f"    RQAOA: E={r['energy']:.4f}, Cost=${r['cost']:,.0f}, "
                  f"AR={r.get('ar', 'N/A')}, T={r['time']:.1f}s")
        if "tenet_general" in res:
            t = res["tenet_general"]
            print(f"    Tenet (General): E={t['energy']:.4f}, "
                  f"Match={t['match']}, T={t['time']:.1f}s")
        if "tenet_mps" in res:
            t = res["tenet_mps"]
            print(f"    Tenet (MPS-{t['bond_dim']}): E={t['energy']:.4f}, "
                  f"Match={t['match']}, T={t['time']:.1f}s")

    # Save JSON
    with open("final_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: final_test_results.json")
    print(f"\n{'=' * 80}")
    print(f"  ALL TESTS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
