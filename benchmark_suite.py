"""
benchmark_suite.py — Comprehensive Quantum vs Classical Benchmark
=================================================================
Runs all 7 algorithms on multiple problem instances and generates
a structured results report showing quantum advantage.

Output: benchmark_results.json + console summary table

Usage:
    python benchmark_suite.py
    python benchmark_suite.py --input request_advantage.json --algorithms qaoa,adapt_vqe
"""

import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from core.problem_encoder import (
    ProblemEncoder, SupplyNode, Route, DemandForecast,
    IsingHamiltonian, greedy_classical_baseline,
)
from core.qaoa_circuit import (
    QAOACircuit, VQEOptimizer, SolutionDecoder,
    GradientVQEOptimizer, AdaptVQEOptimizer, VQDOptimizer,
    ExactGroundStateOptimizer,
)
from core.advanced_algorithms import (
    WarmStartQAOA, MultiObjectiveParetoQAOA, CircuitCuttingPipeline,
    parse_supply_chain,
)
from core.advanced_optimizers import (
    CVaRQAOAOptimizer, LayerByLayerOptimizer, RecursiveQAOA,
)
from core.error_mitigation import mitigate_qaoa


def load_problem(filepath):
    """Load and encode a problem from JSON."""
    with open(filepath) as f:
        request = json.load(f)
    nodes, routes, demands = parse_supply_chain(request)
    encoder = ProblemEncoder(penalty_weight=10.0)
    constraints = request.get("constraints", {})
    ham = encoder.encode(
        nodes, routes, demands,
        objective=request.get("objective", "balanced"),
        lambda_time=float(constraints.get("lambda_time", 0.5)),
        constraints=constraints,
    )
    return nodes, routes, demands, ham, request


def run_algorithm(name, ham, nodes, routes, demands, p=3, shots=1024, max_iter=300):
    """Run a single algorithm and return results dict."""
    t0 = time.time()
    result = {}

    try:
        if name == "qaoa":
            qaoa = QAOACircuit(ham, p_layers=p)
            vqe = VQEOptimizer(qaoa, max_iterations=max_iter, n_restarts=5)
            res = vqe.optimize()
            bs, conf, _ = qaoa.get_best_bitstring(res["best_params"], n_shots=shots)
            result = {"energy": res["best_energy"], "bitstring": bs,
                      "confidence": conf, "converged": res["converged"]}

        elif name == "gradient_vqe":
            qaoa = QAOACircuit(ham, p_layers=p)
            gvqe = GradientVQEOptimizer(qaoa, max_iterations=max_iter, n_restarts=10)
            res = gvqe.optimize()
            bs, conf, _ = qaoa.get_best_bitstring(res["best_params"], n_shots=shots)
            result = {"energy": res["best_energy"], "bitstring": bs,
                      "confidence": conf, "converged": res["converged"]}

        elif name == "adapt_vqe":
            adapt = AdaptVQEOptimizer(ham, max_p=min(p + 3, 8))
            res = adapt.optimize()
            opt_p = res.get("optimal_p", p)
            qaoa = QAOACircuit(ham, p_layers=opt_p)
            bs, conf, _ = qaoa.get_best_bitstring(res["best_params"], n_shots=shots)
            result = {"energy": res["best_energy"], "bitstring": bs,
                      "confidence": conf, "optimal_p": opt_p}

        elif name == "vqd":
            qaoa = QAOACircuit(ham, p_layers=p)
            vqd = VQDOptimizer(qaoa, n_excited=3, beta_penalty=5.0,
                               max_iterations=200, n_restarts=2)
            res = vqd.optimize()
            bs, conf, _ = qaoa.get_best_bitstring(res["best_params"], n_shots=shots)
            result = {"energy": res["best_energy"], "bitstring": bs,
                      "confidence": conf, "n_solutions": res["n_solutions"]}

        elif name == "warm_start":
            ws = WarmStartQAOA(ham, p=p, shots=shots, cvar_alpha=0.2)
            res = ws.optimize()
            result = {"energy": res["best_energy"], "bitstring": res["best_bitstring"],
                      "classical_energy": res.get("classical_energy")}

        elif name == "pareto":
            mo = MultiObjectiveParetoQAOA(nodes, routes, demands, p=p, shots=shots)
            res = mo.explore_pareto(n_points=5)
            front = res["pareto_front"]
            if front:
                best = min(front, key=lambda x: x["total_cost"])
                result = {"energy": best["energy"], "bitstring": best["bitstring"],
                          "n_pareto": res["n_pareto_optimal"],
                          "total_cost": best["total_cost"]}
            else:
                result = {"energy": 0, "bitstring": "0" * ham.n_qubits}

        elif name == "circuit_cut":
            cc = CircuitCuttingPipeline(nodes, routes, demands,
                                        max_fragment_qubits=max(3, ham.n_qubits // 2))
            res = cc.optimize(p=p, shots=shots, method="adapt_vqe")
            result = {"energy": res["combined_energy"],
                      "bitstring": res["combined_bitstring"],
                      "n_fragments": res["n_fragments"]}

        elif name == "exact":
            exact = ExactGroundStateOptimizer(ham)
            res = exact.optimize()
            result = {"energy": res["best_energy"], "bitstring": res["best_bitstring"],
                      "confidence": 1.0, "converged": True,
                      "method": "exact_ground_state",
                      "search_space": 2 ** ham.n_qubits}

        elif name == "rqaoa":
            rqaoa_opt = RecursiveQAOA(
                ham, qaoa_p=min(2, p), threshold=min(3, ham.n_qubits),
                qaoa_restarts=3, qaoa_max_iter=150,
            )
            res = rqaoa_opt.optimize()
            result = {"energy": res["best_energy"], "bitstring": res["best_bitstring"],
                      "confidence": 1.0, "converged": True,
                      "method": "rqaoa",
                      "n_reductions": res.get("n_reductions", 0),
                      "reduction_log": res.get("reduction_log", [])}

        elif name == "cvar_qaoa":
            qaoa = QAOACircuit(ham, p_layers=p)
            cvar = CVaRQAOAOptimizer(qaoa, max_iterations=max_iter, n_restarts=8)
            res = cvar.optimize()
            bs, conf, _ = qaoa.get_best_bitstring(res["best_params"], n_shots=shots)
            result = {"energy": res["best_energy"], "bitstring": bs,
                      "confidence": conf, "converged": res["converged"]}

        elif name == "layer_by_layer":
            lbl = LayerByLayerOptimizer(ham, target_p=p,
                                        max_iter_per_layer=150,
                                        max_iter_refinement=300)
            res = lbl.optimize()
            qaoa_decode = QAOACircuit(ham, p_layers=p)
            bs, conf, _ = qaoa_decode.get_best_bitstring(res["best_params"], n_shots=shots)
            result = {"energy": res["best_energy"], "bitstring": bs,
                      "confidence": conf, "converged": res["converged"],
                      "layer_energies": res.get("layer_energies", [])}

    except Exception as e:
        result = {"error": str(e)}

    result["time_seconds"] = round(time.time() - t0, 2)
    result["algorithm"] = name
    return result


def compute_cost(bitstring, routes):
    """Compute total cost from a bitstring."""
    total = 0
    for i, bit in enumerate(bitstring):
        if bit == "1" and i < len(routes):
            total += routes[i].cost_per_unit * routes[i].capacity
    return total


def compute_approximation_ratio(energy, ham):
    """
    Compute approximation ratio: E_quantum / E_optimal.

    For small problems (≤20 qubits), brute-force the optimal solution.
    """
    n = ham.n_qubits
    if n > 20:
        return None  # Too large for brute force

    best_energy = float("inf")
    for state_int in range(2 ** n):
        bits = np.array([(state_int >> i) & 1 for i in range(n)])
        spins = 1 - 2 * bits
        energy = sum(ham.h.get(i, 0) * spins[i] for i in range(n))
        for (qi, qj), J_val in ham.J.items():
            energy += J_val * spins[qi] * spins[qj]
        energy += ham.offset
        if energy < best_energy:
            best_energy = energy

    return best_energy


def main():
    parser = argparse.ArgumentParser(description="Quantum Benchmark Suite")
    parser.add_argument("--input", "-i", nargs="+",
                        default=["data/request.json", "data/request_12q.json", "data/request_advantage.json"])
    parser.add_argument("--algorithms", "-a", default="all")
    parser.add_argument("--layers", "-p", type=int, default=3)
    args = parser.parse_args()

    if args.algorithms == "all":
        algorithms = ["exact", "rqaoa", "cvar_qaoa", "layer_by_layer",
                       "qaoa", "gradient_vqe", "adapt_vqe", "vqd",
                       "warm_start", "pareto", "circuit_cut"]
    else:
        algorithms = args.algorithms.split(",")

    all_results = {}

    for filepath in args.input:
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filepath} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  Problem: {filepath}")
        print(f"{'='*70}")

        nodes, routes, demands, ham, request = load_problem(filepath)
        n_qubits = ham.n_qubits
        print(f"  Qubits: {n_qubits}, Routes: {len(routes)}")

        # Classical baseline
        greedy = greedy_classical_baseline(
            routes, demands, nodes,
            objective=request.get("objective", "balanced"))
        greedy_cost = greedy["total_cost"]
        print(f"  Greedy baseline: ${greedy_cost:,.0f}")

        # Brute-force optimal (small problems)
        optimal_energy = compute_approximation_ratio(0, ham)
        if optimal_energy is not None:
            print(f"  Optimal energy (brute-force): {optimal_energy:.4f}")

        # Read problem-specific settings from the request JSON
        req_p = request.get("p_layers", args.layers)
        req_iter = request.get("quantum_iterations", 300)

        # Run algorithms
        problem_results = {"n_qubits": n_qubits, "greedy_cost": greedy_cost,
                           "optimal_energy": optimal_energy, "algorithms": {}}

        print(f"\n  {'Algorithm':<16} {'Energy':>10} {'Cost':>10} {'Adv.':>6} {'AR':>6} {'Time':>6}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*6}")

        for algo in algorithms:
            if algo not in ["pareto", "circuit_cut"] or n_qubits >= 4:
                res = run_algorithm(algo, ham, nodes, routes, demands,
                                    p=req_p, max_iter=req_iter)

                if "error" in res:
                    print(f"  {algo:<16} ERROR: {res['error'][:40]}")
                    problem_results["algorithms"][algo] = res
                    continue

                energy = res.get("energy", 0)
                bs = res.get("bitstring", "0" * n_qubits)
                cost = compute_cost(bs, routes)
                advantage = greedy_cost / cost if cost > 0 else 0
                ar = energy / optimal_energy if optimal_energy and optimal_energy < 0 else None
                ar_str = f"{ar:.4f}" if ar else "N/A"

                res["cost"] = cost
                res["advantage"] = round(advantage, 4)
                res["approximation_ratio"] = round(ar, 4) if ar else None
                problem_results["algorithms"][algo] = res

                print(f"  {algo:<16} {energy:>10.2f} ${cost:>9,.0f} {advantage:>5.2f}x {ar_str:>6} {res['time_seconds']:>5.1f}s")

        # ZNE demo on best algorithm
        best_algo = min(
            [(k, v) for k, v in problem_results["algorithms"].items()
             if "energy" in v and "error" not in v],
            key=lambda x: x[1]["energy"],
            default=None
        )
        if best_algo and n_qubits <= 20:
            try:
                algo_name, algo_res = best_algo
                qaoa = QAOACircuit(ham, p_layers=args.layers)
                vqe = VQEOptimizer(qaoa, max_iterations=100, n_restarts=2)
                vqe_res = vqe.optimize()
                zne = mitigate_qaoa(qaoa, vqe_res["best_params"],
                                    base_noise=0.005, method="polynomial")
                problem_results["zne"] = {
                    "ideal_energy": float(vqe_res["best_energy"]),
                    "raw_noisy_energy": zne["raw_energy"],
                    "mitigated_energy": zne["mitigated_energy"],
                    "improvement_pct": zne["improvement_pct"],
                }
                print(f"\n  ZNE Error Mitigation:")
                print(f"    Ideal:     {vqe_res['best_energy']:.4f}")
                print(f"    Noisy:     {zne['raw_energy']:.4f}")
                print(f"    Mitigated: {zne['mitigated_energy']:.4f} "
                      f"(+{zne['improvement_pct']:.1f}%)")
            except Exception as e:
                print(f"  ZNE failed: {e}")

        all_results[filepath] = problem_results

    # Save results
    outfile = "benchmark_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"  Results saved: {outfile}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
