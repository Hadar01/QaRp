"""
qarp_mock.devices — Device and NoiseModel
==========================================
Mirrors ``qarp.devices`` from QARP v0.4.3.

Provides Device specification and NoiseModel for simulating
quantum hardware noise characteristics.
"""

from __future__ import annotations
from typing import List, Optional


class NoiseModel:
    """
    Mock noise model for quantum device simulation.
    Mirrors qarp.devices.NoiseModel.
    """

    def __init__(self):
        self.errors: list = []

    def add_bit_flip_error(self, param: float, gate_set=None):
        self.errors.append(("bit_flip", param, gate_set))

    def add_amplitude_damping_error(self, param: float, gate_set=None):
        self.errors.append(("amplitude_damping", param, gate_set))

    def add_depolarizing_error(self, param: float, gate_set=None):
        self.errors.append(("depolarizing", param, gate_set))

    def __repr__(self) -> str:
        return f"NoiseModel({len(self.errors)} error channels)"


class Device:
    """
    Mock quantum device specification.
    Mirrors qarp.devices.Device.
    """

    def __init__(self, n_qubits: int, architecture=None,
                 noise_model: Optional[NoiseModel] = None,
                 gate_set=None, directedness: bool = False):
        self.n_qubits = n_qubits
        self.architecture = architecture
        self.noise_model = noise_model
        self.gate_set = gate_set
        self.directedness = directedness

    def transform_circuit(self, circuit):
        """Apply device constraints to circuit (no-op in mock)."""
        return circuit

    def __repr__(self) -> str:
        return f"Device({self.n_qubits}q, noise={self.noise_model is not None})"


def get_all_to_all_architecture(n_qubits: int):
    """All-to-all qubit connectivity."""
    return {"type": "all_to_all", "n_qubits": n_qubits}


def get_nearest_neighbour_architecture(xdim: int, ydim: int):
    """Grid nearest-neighbor connectivity."""
    return {"type": "nearest_neighbour", "xdim": xdim, "ydim": ydim}
