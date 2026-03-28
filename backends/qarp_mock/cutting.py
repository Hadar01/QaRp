"""
qarp_mock.cutting — EAPartitioning circuit cutting
===================================================
Mirrors ``qarp.cutting.EAPartitioning`` from QARP v0.4.3.

In mock mode, uses simple qubit-index-based partitioning.
On FX700, QARP's real EAPartitioning uses evolutionary algorithms
for optimal gate-level wire cutting.
"""

from __future__ import annotations
from typing import List, Optional


class CutterResult:
    """Result container from circuit cutting."""

    def __init__(self, subcircuits: list, cut_circuit=None):
        self.subcircuits = subcircuits
        self.cut_circuit = cut_circuit

    @property
    def n_fragments(self) -> int:
        return len(self.subcircuits)


class EAPartitioning:
    """
    Mock of qarp.cutting.EAPartitioning.

    Evolutionary Algorithm-based circuit partitioning.
    Mock uses simple qubit-index-based partitioning.

    Parameters
    ----------
    original_circuit     : Block or Circuit to be partitioned
    max_size_subcircuits : list of max qubits per subcircuit (uses first element)
    verbose              : whether to print partitioning details
    """

    def __init__(self, original_circuit, max_size_subcircuits: list = None,
                 verbose: bool = True):
        self.circuit = original_circuit
        if max_size_subcircuits is None:
            max_size_subcircuits = [20]
        self.max_size = (max_size_subcircuits[0]
                         if isinstance(max_size_subcircuits, list)
                         else int(max_size_subcircuits))
        self.verbose = verbose
        self._result: Optional[CutterResult] = None

    def cut(self, manual_setting=None) -> CutterResult:
        """Partition circuit into fragments of ≤ max_size qubits."""
        n = self.circuit.n_qubits
        subcircuits = []

        for start in range(0, n, self.max_size):
            end = min(start + self.max_size, n)
            chunk_qubits = list(range(start, end))
            frag = self._build_fragment(chunk_qubits)
            subcircuits.append(frag)

        self._result = CutterResult(subcircuits=subcircuits)
        return self._result

    def _build_fragment(self, qubit_indices: List[int]):
        """Build a sub-block for the given qubit indices."""
        from backends.qarp_mock.blocks import (
            CompositeBlock, HnBlock, IsingCostBlock, MixerBlock, Block,
        )

        n_frag = len(qubit_indices)
        qubit_set = set(qubit_indices)
        local_map = {q: i for i, q in enumerate(qubit_indices)}

        # If the circuit is a CompositeBlock, rebuild each sub-block
        if hasattr(self.circuit, 'blocks'):
            sub_blocks = []
            for block in self.circuit.blocks:
                if isinstance(block, HnBlock):
                    sub_blocks.append(HnBlock(n_frag))
                elif isinstance(block, IsingCostBlock):
                    sub_h = {}
                    for qi, val in block.h.items():
                        if qi in qubit_set:
                            sub_h[local_map[qi]] = val
                    sub_J = {}
                    for (qi, qj), val in block.J.items():
                        if qi in qubit_set and qj in qubit_set:
                            sub_J[(local_map[qi], local_map[qj])] = val
                    sub_blocks.append(
                        IsingCostBlock(n_frag, sub_h, sub_J, block.gamma_symbol))
                elif isinstance(block, MixerBlock):
                    sub_blocks.append(
                        MixerBlock(n_frag, block.beta_symbol))
                # else: skip unknown block types in fragment

            frag = CompositeBlock(blocks=sub_blocks, n_qubits=n_frag)
        else:
            # Fallback: create a simple block placeholder
            frag = Block(n_frag, name=f"Fragment_{qubit_indices[0]}-{qubit_indices[-1]}")

        frag._original_qubit_map = {i: q for i, q in enumerate(qubit_indices)}
        return frag

    @property
    def subcircuits(self):
        return self._result.subcircuits if self._result else None

    def __repr__(self) -> str:
        n_frag = self._result.n_fragments if self._result else 0
        return (f"EAPartitioning({self.circuit.n_qubits}q → "
                f"{n_frag} fragments, max={self.max_size}q)")
