Create a CLAUDE.md file in the repo root with this content:

# QaRp — Quantum Supply Chain Optimization

## Project
Fujitsu Global Quantum Simulator Challenge submission.
RQAOA-based supply chain optimization with Ising Hamiltonian encoding.

## Stack
- Python 3.10+
- numpy, scipy (core dependencies)
- No external quantum frameworks (custom simulator)

## Key files
- core/problem_encoder.py — Hamiltonian construction (QUBO → Ising)
- core/qaoa_circuit.py — QAOA ansatz and exact optimizer
- core/advanced_optimizers.py — RQAOA, CVaR-QAOA
- benchmark_suite.py — benchmark and advantage computation
- tests/tests.py — test suite (run with: python tests/tests.py)

## Test command
python tests/tests.py

## Rules
- All tests must pass before committing
- Don't change the RQAOA algorithm logic
- Keep the 6q advantage problem result: bitstring 011010, cost $850