"""
qulacs_compat.py
================
Platform-aware import shim for qulacs / numpy_simulator.

All qarp_mock modules should import from HERE instead of from qulacs directly.
On aarch64 (FX700 compute nodes) where qulacs segfaults, this transparently
provides the numpy_simulator replacements.
"""

import platform

_ARCH = platform.machine().lower()

if _ARCH in ('aarch64', 'arm64'):
    from core.numpy_simulator import QuantumState, QuantumCircuit, Observable
else:
    try:
        from qulacs import QuantumState, QuantumCircuit, Observable
    except ImportError:
        from core.numpy_simulator import QuantumState, QuantumCircuit, Observable
