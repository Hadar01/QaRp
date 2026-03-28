#!/usr/bin/env python3
"""Debug script to isolate segfault on FX700 compute nodes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def step(msg):
    print(f"[STEP] {msg}", flush=True)

# --- 1. Basic imports ---
step("1. import numpy")
import numpy as np
step("2. import scipy")
from scipy.optimize import minimize
step("3. import qulacs")
from qulacs import QuantumState, QuantumCircuit, Observable

# --- 2. Basic qulacs operations ---
step("4. QuantumState(6)")
state = QuantumState(6)
state.set_zero_state()
step(f"   state OK: {state.get_vector()[:4]}")

step("5. QuantumCircuit(6) with gates")
circuit = QuantumCircuit(6)
for i in range(6):
    circuit.add_H_gate(i)
circuit.add_CNOT_gate(0, 1)
circuit.update_quantum_state(state)
step("   circuit OK")

step("6. Observable(6)")
obs = Observable(6)
obs.add_operator(1.0, "Z 0")
obs.add_operator(0.5, "Z 0 Z 1")
ev = obs.get_expectation_value(state)
step(f"   observable OK: E={ev:.4f}")

# --- 3. scipy minimize with qulacs ---
step("7. scipy.optimize.minimize + qulacs")
def cost_fn(params):
    s = QuantumState(2)
    s.set_zero_state()
    c = QuantumCircuit(2)
    c.add_RX_gate(0, params[0])
    c.add_RY_gate(1, params[1])
    c.update_quantum_state(s)
    o = Observable(2)
    o.add_operator(1.0, "Z 0 Z 1")
    return float(o.get_expectation_value(s))

result = minimize(cost_fn, x0=[0.5, 0.5], method="COBYLA", options={"maxiter": 20})
step(f"   minimize OK: energy={result.fun:.4f}")

# --- 4. Problem encoder ---
step("8. import ProblemEncoder")
from core.problem_encoder import ProblemEncoder, SupplyNode, Route, DemandForecast
step("9. encode problem")
nodes = [
    SupplyNode('WH-A', 'Hub', 'warehouse', 500, 250),
    SupplyNode('S1', 'Store', 'retail', 300, 10),
]
routes = [Route('WH-A', 'S1', 50, 1.0, 1.0, 200)]
demands = [DemandForecast('S1', 100, 1)]
encoder = ProblemEncoder()
ham = encoder.encode(nodes, routes, demands)
step(f"   encoded: n_qubits={ham.n_qubits}")

# --- 5. QAOA circuit ---
step("10. import QAOACircuit")
from core.qaoa_circuit import QAOACircuit, VQEOptimizer, SolutionDecoder
step("11. create QAOACircuit")
qaoa = QAOACircuit(ham, p_layers=1)
step("12. expectation_value")
params = np.array([0.5, 0.5])
ev = qaoa.expectation_value(params)
step(f"    E={ev:.4f}")

step("13. VQEOptimizer.optimize()")
vqe = VQEOptimizer(qaoa, max_iterations=20, n_restarts=1)
vqe_result = vqe.optimize()
step(f"    energy={vqe_result['best_energy']:.4f}")

# --- 6. QARP mock path ---
step("14. import QARP mock")
from backends.qarp_backend import build_qarp_engine, build_qarp_hamiltonian, build_qarp_ansatz
from backends.qarp_backend import QARPConfig, BackendType, QARPOptimizer
from backends.qarp_mock.algorithms import StateVector

step("15. build_qarp_engine (QULACS_SINGLE)")
cfg = QARPConfig()
cfg.backend_type = BackendType.QULACS_SINGLE
engine = build_qarp_engine(cfg)
step(f"    engine={engine}")

step("16. build_qarp_hamiltonian")
qarp_ham = build_qarp_hamiltonian(ham)
step(f"    ham terms={len(qarp_ham.terms)}")

step("17. build_qarp_ansatz")
ansatz = build_qarp_ansatz(ham, p_layers=1)
symbols = ansatz.get_symbols()
step(f"    symbols={symbols}")

step("18. StateVector.evaluate")
measurement = StateVector(bra=ansatz, operator=qarp_ham, ket=ansatz)
engine.build(measurements=[measurement])
ev = engine.run(parameters={s: 0.5 for s in symbols})
step(f"    E={ev:.4f}")

step("19. QARPOptimizer (full VQE)")
cfg.max_iterations = 20
optimizer = QARPOptimizer(ham, cfg, p_layers=1)
result = optimizer.optimize()
step(f"    energy={result['best_energy']:.4f}")

# --- 7. Full pipeline ---
step("20. run_optimization_pipeline")
import json
with open("data/request_advantage.json") as f:
    req = json.load(f)
req["backend"] = "qulacs"
req["quantum_iterations"] = 50

from main import run_optimization_pipeline
result = run_optimization_pipeline(req)
step(f"    status={result['status']}")
step(f"    advantage={result.get('quantum_metrics',{}).get('quantum_advantage','?')}")

step("ALL DONE — no segfault!")
