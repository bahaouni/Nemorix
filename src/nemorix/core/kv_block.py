"""KV-cache block — the atomic unit of agent memory."""

from __future__ import annotations
from dataclasses import dataclass, field
import uuid


COMPRESSION_RATIOS = {"fp16": 1.0, "fp8": 0.5, "int4": 0.25}


@dataclass
class KVBlock:
    block_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_id: str = ""
    layer_idx: int = 0
    num_tokens: int = 0
    size_bytes: int = 0
    dtype: str = "fp16"
    attention_score: float = 0.5
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

    def copy(self) -> "KVBlock":
        return KVBlock(
            block_id=self.block_id,
            agent_id=self.agent_id,
            layer_idx=self.layer_idx,
            num_tokens=self.num_tokens,
            size_bytes=self.size_bytes,
            dtype=self.dtype,
            attention_score=self.attention_score,
            last_accessed=self.last_accessed,
            tier=self.tier,
        )
