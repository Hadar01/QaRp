#!/usr/bin/env python3
"""
run_optimization.py
====================
FX700 command-line entry-point for QARP-based quantum optimization.

This is the script invoked by the generated Slurm .job file:
    python -u run_optimization.py --backend qulacs_mpi --n_qubits 12 ...

It reads a problem JSON from stdin or a file, runs the full pipeline,
and writes the result JSON to stdout (captured by Slurm's --output log).

Usage:
    # Local testing:
    python run_optimization.py --input request_12q.json

    # FX700 via Slurm:
    python -u run_optimization.py --backend qulacs_mpi --n_qubits 12 \\
           --shots 10000 --max_iter 300 --input request_12q.json

    # Read from stdin:
    cat request_12q.json | python run_optimization.py --backend qulacs
"""

import argparse
import json
import sys
import os
import time
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_optimization")


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantum Supply Chain Optimization — FX700 CLI"
    )
    p.add_argument(
        "--input", "-i",
        type=str, default=None,
        help="Path to input JSON file. Reads stdin if omitted.",
    )
    p.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Path to write result JSON. Writes stdout if omitted.",
    )
    p.add_argument(
        "--backend", "-b",
        type=str, default=None,
        choices=["qulacs", "qulacs_mpi", "qiskit_aer", "pytket_tenet", "local_sim"],
        help="Override backend (default: auto-detect).",
    )
    p.add_argument(
        "--n_qubits",
        type=int, default=None,
        help="Expected qubit count (informational only).",
    )
    p.add_argument(
        "--shots",
        type=int, default=10000,
        help="Measurement shots for sampling (default: 10000).",
    )
    p.add_argument(
        "--max_iter",
        type=int, default=300,
        help="Max VQE iterations per restart (default: 300).",
    )
    p.add_argument(
        "--p_layers",
        type=int, default=None,
        help="QAOA circuit depth. Default: auto from quantum_iterations.",
    )
    p.add_argument(
        "--objective",
        type=str, default=None,
        choices=["minimize_cost", "minimize_time", "maximize_efficiency", "balanced"],
        help="Override objective from JSON.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    start = time.time()

    # ── Load problem ──────────────────────────────────────────────────────
    if args.input:
        logger.info(f"Loading problem from: {args.input}")
        with open(args.input, "r") as f:
            request_dict = json.load(f)
    else:
        logger.info("Reading problem from stdin...")
        request_dict = json.load(sys.stdin)

    # ── Apply CLI overrides ───────────────────────────────────────────────
    if args.backend:
        request_dict["backend"] = args.backend
    if args.objective:
        request_dict["objective"] = args.objective
    if args.max_iter:
        request_dict["quantum_iterations"] = max(
            request_dict.get("quantum_iterations", 100),
            args.max_iter,
        )

    n_routes = len(request_dict.get("routes", []))
    logger.info(f"Problem: {n_routes} routes ({n_routes} qubits), "
                f"objective={request_dict.get('objective', 'balanced')}")

    # ── Import and run pipeline ───────────────────────────────────────────
    from main import run_optimization_pipeline

    logger.info("Starting quantum optimization pipeline...")
    result = run_optimization_pipeline(request_dict)

    elapsed = time.time() - start
    status = result.get("status", "unknown")
    obj = result.get("objective_value", "?")
    qm = result.get("quantum_metrics", {})
    converged = qm.get("convergence_achieved", False)
    advantage = qm.get("quantum_advantage", "?")

    logger.info(f"Pipeline completed in {elapsed:.1f}s")
    logger.info(f"  Status:            {status}")
    logger.info(f"  Objective value:   {obj}")
    logger.info(f"  Converged:         {converged}")
    logger.info(f"  Quantum advantage: {advantage}")
    logger.info(f"  Bitstring:         {qm.get('bitstring', '?')}")
    logger.info(f"  Backend:           {qm.get('backend', '?')}")

    # ── Print selected routes ─────────────────────────────────────────────
    selected = [r for r in result.get("route_assignments", []) if r.get("selected")]
    logger.info(f"  Selected routes ({len(selected)}/{n_routes}):")
    for r in selected:
        logger.info(f"    qubit {r['qubit_index']}: {r['route_id']}  "
                     f"flow={r['flow']}  cost=${r['total_cost']:.2f}")

    # ── Print demand satisfaction ─────────────────────────────────────────
    logger.info("  Demand satisfaction:")
    for ds in result.get("demand_satisfaction", []):
        status_icon = "✓" if ds.get("satisfied") else "✗"
        logger.info(f"    {status_icon} {ds['node_id']}: "
                     f"delivered={ds['delivered']}/{ds['demand']}  "
                     f"slack={ds['slack']}")

    # ── Write output ──────────────────────────────────────────────────────
    result_json = json.dumps(result, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json)
        logger.info(f"Result written to: {args.output}")
    else:
        print(result_json)

    return 0 if status in ("success", "fallback_classical") else 1


if __name__ == "__main__":
    sys.exit(main())
