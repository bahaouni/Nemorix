"""KV-cache block — the atomic unit of agent memory.

A KV block is modeled as an abstract memory object characterized by its size,
precision, access history, storage tier, and importance score.  Internal tensor
dimensions (e.g., attention heads and head dimension) are abstracted into the
block size, as they are not required for evaluating memory management policies.

``size_bytes`` represents the combined Key and Value tensors for one
transformer layer of one agent's KV-cache.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import uuid


COMPRESSION_RATIOS = {"fp16": 1.0, "fp8": 0.5, "int4": 0.25}


@dataclass
class KVBlock:
    """One layer's KV-cache (Keys + Values) for a single agent."""

    block_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_id: str = ""
    layer_idx: int = 0
    num_tokens: int = 0
    size_bytes: int = 0  # Combined K + V tensor size at current dtype
    dtype: str = "fp16"
    importance_score: float = 0.5  # Policy-agnostic importance (not necessarily attention)
    last_accessed: float = 0.0
    tier: str = "gpu"

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def compressed_size(self, target_dtype: str) -> int:
        if target_dtype not in COMPRESSION_RATIOS or self.dtype not in COMPRESSION_RATIOS:
            return self.size_bytes
        factor = COMPRESSION_RATIOS[target_dtype] / COMPRESSION_RATIOS[self.dtype]
        return max(1, int(self.size_bytes * factor))

    def compress_to(self, target_dtype: str) -> None:
        self.size_bytes = self.compressed_size(target_dtype)
        self.dtype = target_dtype

    def copy(self, new_id: bool = True) -> "KVBlock":
        """Create a copy. By default assigns a new block_id to avoid aliasing."""
        return KVBlock(
            block_id=uuid.uuid4().hex[:8] if new_id else self.block_id,
            agent_id=self.agent_id,
            layer_idx=self.layer_idx,
            num_tokens=self.num_tokens,
            size_bytes=self.size_bytes,
            dtype=self.dtype,
            importance_score=self.importance_score,
            last_accessed=self.last_accessed,
            tier=self.tier,
        )
