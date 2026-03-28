"""
qarp_mock — Local Mock of Fujitsu QARP v0.4.3
==============================================
This package mirrors the QARP v0.4.3 API surface using Qulacs as the backend.
All QARP code paths in qarp_backend.py execute identically through this mock,
ensuring correctness before deploying to FX700.

QARP v0.4.3 API Changes from v1.6.2:
  - Hamiltonians: openfermion.QubitOperator replaces PauliHamiltonian/PauliTerm
  - Circuits: qarp.blocks.CompositeBlock replaces ParametricCircuit
  - Engines:  engine.build(measurements) + engine.run(params) pattern
  - Algorithms: VQE, QAOA, AdaptVQE composite classes
  - Cutting: EAPartitioning replaces CircuitCutter
  - Devices: Device + NoiseModel for noise simulation
  - Optimizers: ScipyOptimizer, RotosolveOptimizer

On FX700, the real ``qarp`` package takes priority (installed in site-packages).
Locally, this mock enables full integration testing of every QARP feature.
"""

__version__ = "mock-2.0.0"
__qarp_compat__ = "0.4.3"  # Compatible with QARP v0.4.3 (QSC2025)

# Re-export all mock classes for convenient imports
from backends.qarp_mock.openfermion_mock import QubitOperator
from backends.qarp_mock.blocks import (
    Block, CompositeBlock, HnBlock, IsingCostBlock, MixerBlock, LayeredHEABlock,
)
from backends.qarp_mock.engines import QulacsEngine, TketEngine
from backends.qarp_mock.algorithms import (
    StateVector, PauliAveraging, Sampler,
    VQE, QAOA, AdaptVQE,
)
from backends.qarp_mock.optimizers import ScipyOptimizer, RotosolveOptimizer
from backends.qarp_mock.cutting import EAPartitioning, CutterResult
from backends.qarp_mock.devices import (
    Device, NoiseModel,
    get_all_to_all_architecture, get_nearest_neighbour_architecture,
)
