"""
qulacs_compat.py
================
Platform-aware import shim for qulacs / numpy_simulator.

All qarp_mock modules should import from HERE instead of from qulacs directly.
On aarch64 (FX700 compute nodes) where the pre-installed qulacs binary may
segfault, this first attempts to import a natively-built qulacs, and only
falls back to the pure-numpy simulator if that import raises ImportError.

Note: segfaults cannot be caught by try/except (they kill the process).
If qulacs was built from source on aarch64 and installed correctly,
the import will succeed without segfault. If only the broken pre-compiled
binary exists, importing it will segfault — meaning the pip upgrade +
source build step is a prerequisite.
"""

import platform
import logging

logger = logging.getLogger(__name__)
_ARCH = platform.machine().lower()

try:
    from qulacs import QuantumState, QuantumCircuit, Observable
    _BACKEND = "qulacs-native"
except ImportError:
    from core.numpy_simulator import QuantumState, QuantumCircuit, Observable
    _BACKEND = "numpy-simulator"

if _ARCH in ('aarch64', 'arm64'):
    logger.info(f"aarch64 detected — quantum backend: {_BACKEND}")
else:
    logger.info(f"x86_64 detected — quantum backend: {_BACKEND}")
