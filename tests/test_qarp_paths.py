"""
test_qarp_paths.py — Verifies ALL QARP v0.4.3 code paths execute via mock
==========================================================================

Run: python tests/test_qarp_paths.py

Tests the QARP v0.4.3 API upgrade:
  - QubitOperator (replaces PauliHamiltonian)
  - CompositeBlock ansatz (replaces ParametricCircuit)
  - engine.build()/run() pattern (replaces engine.get_expectation_value())
  - EAPartitioning (replaces CircuitCutter)
  - AdaptVQE composite (replaces ADAPT_VQE)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backends.qarp_backend import (
    QARP_AVAILABLE, QARP_MOCK,
    build_qarp_engine, build_qarp_hamiltonian, build_qarp_ansatz,
    QARPConfig, BackendType, QARPOptimizer, QARPAdaptOptimizer,
    QARPCircuitCuttingOptimizer, detect_environment, generate_slurm_script,
)
from core.problem_encoder import ProblemEncoder, SupplyNode, Route, DemandForecast
from backends.qarp_mock.algorithms import StateVector


def main():
    passed = 0
    failed = 0

    def check(name, condition, detail=''):
        nonlocal passed, failed
        if condition:
            print(f'  [PASS] {name}')
            passed += 1
        else:
            print(f'  [FAIL] {name} — {detail}')
            failed += 1

    print('=' * 65)
    print('  QARP v0.4.3 Integration Test Suite (Mock-Verified Code Paths)')
    print('=' * 65)

    env = detect_environment()
    print(f'\nEnvironment: qarp={QARP_AVAILABLE}, mock={QARP_MOCK}')
    check('QARP available', QARP_AVAILABLE, 'qarp_mock should load if real QARP absent')

    nodes = [
        SupplyNode('WH-1', 'Warehouse', 'warehouse', 1000, 800),
        SupplyNode('S-1', 'Store 1', 'retail', 300, 50),
        SupplyNode('S-2', 'Store 2', 'retail', 300, 30),
    ]
    routes = [
        Route('WH-1', 'S-1', 100, 2.5, 1.5, 300),
        Route('WH-1', 'S-2', 150, 3.0, 2.0, 300),
    ]
    demands = [DemandForecast('S-1', 200, 3), DemandForecast('S-2', 150, 2)]
    encoder = ProblemEncoder()
    ham = encoder.encode(nodes, routes, demands)

    print('\n[1] build_qarp_hamiltonian() → QubitOperator')
    qarp_ham = build_qarp_hamiltonian(ham)
    check('Returns QubitOperator', qarp_ham is not None)
    check('Correct n_qubits', getattr(qarp_ham, 'n_qubits', None) == 2)
    check('Has Pauli terms', hasattr(qarp_ham, 'terms') and len(qarp_ham.terms) > 0)
    check('Has identity (offset) term', () in getattr(qarp_ham, 'terms', {}))

    print('\n[2] build_qarp_ansatz() → CompositeBlock')
    circuit = build_qarp_ansatz(ham, p_layers=2)
    check('Returns CompositeBlock', circuit is not None)
    check('Correct n_qubits', getattr(circuit, 'n_qubits', None) == 2)
    check('Has named symbols', len(circuit.get_symbols()) > 0)
    symbols = circuit.get_symbols()
    check('Has gamma symbols', any('gamma' in s for s in symbols))
    check('Has beta symbols', any('beta' in s for s in symbols))
    check('Symbol count = 2*p', len(symbols) == 4,
          f'Expected 4 symbols (2 layers), got {len(symbols)}')

    print('\n[3] build_qarp_engine()')
    for bt in [BackendType.QULACS_SINGLE, BackendType.QULACS_MPI, BackendType.LOCAL_SIM]:
        cfg = QARPConfig()
        cfg.backend_type = bt
        engine = build_qarp_engine(cfg)
        if bt == BackendType.LOCAL_SIM:
            check(f'Backend {bt.value} returns None (local path)', engine is None)
        else:
            check(f'Backend {bt.value} returns engine', engine is not None)

    print('\n[4] Engine build(measurements) + run(parameters) pattern')
    engine = build_qarp_engine(QARPConfig())
    measurement = StateVector(bra=circuit, operator=qarp_ham, ket=circuit)
    engine.build(measurements=[measurement])
    ev = engine.run(parameters={s: 0.5 for s in symbols})
    check('engine.run() returns float', isinstance(ev, (int, float)))
    check('Value is finite', abs(ev) < 1e6)
    print(f'    E = {ev:.6f}')

    print('\n[5] Engine sampling (v0.4.3 format)')
    circuit.set_symbols({s: 0.5 for s in symbols})
    samples = engine.sample(circuit=circuit, n_shots=200)
    check('Returns dict', isinstance(samples, dict))
    check('Keys are tuples', all(isinstance(k, tuple) for k in samples))
    check('Values are probabilities', all(0 <= v <= 1 for v in samples.values()))
    total_prob = sum(samples.values())
    check('Probabilities sum to ~1', abs(total_prob - 1.0) < 0.1,
          f'Sum={total_prob:.3f}')
    print(f'    Samples: {samples}')

    print('\n[6] QARPOptimizer (QARP path)')
    cfg = QARPConfig()
    cfg.backend_type = BackendType.QULACS_SINGLE
    cfg.max_iterations = 20
    optimizer = QARPOptimizer(ham, cfg, p_layers=2)
    check('Mode = qarp (not local)', getattr(optimizer, '_mode', '') == 'qarp')
    result = optimizer.optimize()
    check('Returns best_energy', 'best_energy' in result)
    check('Energy is finite', abs(result['best_energy']) < 1e6)
    check('Has valid result', result.get('converged', False) or result['best_energy'] < 0)
    print(f'    Energy: {result["best_energy"]:.4f}')

    print('\n[7] QARPOptimizer.get_best_bitstring()')
    bs, conf, counts = optimizer.get_best_bitstring(result['best_params'], n_shots=200)
    check('Bitstring length = n_qubits', len(bs) == 2)
    check('Bitstring is binary', all(c in '01' for c in bs))
    check('Confidence > 0', conf > 0)
    print(f'    Bitstring: {bs}, Confidence: {conf:.1%}')

    print('\n[8] QARPAdaptOptimizer (QARP AdaptVQE)')
    adapt = QARPAdaptOptimizer(ham, cfg, max_layers=4, gradient_threshold=0.01)
    adapt_result = adapt.optimize()
    check('Returns best_energy', 'best_energy' in adapt_result)
    check('Method contains adapt', 'adapt' in adapt_result.get('method', ''))
    print(f'    Energy: {adapt_result["best_energy"]:.4f}, Method: {adapt_result.get("method", "")}')

    print('\n[9] QARPCircuitCuttingOptimizer (QARP EAPartitioning)')
    cutter = QARPCircuitCuttingOptimizer(ham, routes, cfg, max_fragment_qubits=1)
    cut_result = cutter.optimize(p_layers=2)
    check('Method contains qarp', 'qarp' in cut_result.get('method', ''))
    check('Has n_fragments', 'n_fragments' in cut_result)
    check('n_fragments >= 2', cut_result.get('n_fragments', 0) >= 2)
    print(f'    Fragments: {cut_result.get("n_fragments")}, Method: {cut_result.get("method", "")}')

    print('\n[10] generate_slurm_script()')
    for bt in [BackendType.QULACS_MPI, BackendType.PYTKET_TENET]:
        script = generate_slurm_script(QARPConfig(), n_qubits=20)
        check(f'Slurm script ({bt.value})', 'SBATCH' in script and 'qsc_supply_chain' in script)

    print('\n[11] detect_environment()')
    env = detect_environment()
    check('Has qarp_available', 'qarp_available' in env)
    check('Has qarp_mock', 'qarp_mock' in env)
    check('Has suggested_backend', 'suggested_backend' in env)

    print('\n[12] QubitOperator API (openfermion mock)')
    from backends.qarp_mock.openfermion_mock import QubitOperator
    op1 = QubitOperator('Z0 Z1', 0.5)
    op2 = QubitOperator('Z0', -0.3)
    combined = op1 + op2
    check('QubitOperator addition works', len(combined.terms) == 2)
    check('QubitOperator n_qubits', combined.n_qubits == 2)
    op3 = QubitOperator('', 1.0)
    check('Identity term', op3.get_constant() == 1.0)

    print('\n' + '=' * 65)
    total = passed + failed
    print(f'  QARP v0.4.3 Integration: {passed}/{total} passed, {failed} failed')
    if QARP_MOCK:
        print('  Mode: qarp_mock v0.4.3 (local Qulacs) — identical API to FX700 QARP')
    else:
        print('  Mode: real QARP v0.4.3')
    print('=' * 65)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
