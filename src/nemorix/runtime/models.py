"""Data models for the measured KV offload prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EncodedTensor:
    """A host-resident encoded tensor and the metadata needed to restore it."""

    payload: Any
    shape: tuple[int, ...]
    source_dtype: str
    storage_dtype: str
    nbytes: int


@dataclass
class StoredLayer:
    """Encoded key/value tensors for one agent layer."""

    agent_id: str
    layer_idx: int
    key: EncodedTensor
    value: EncodedTensor
    num_tokens: int
    importance_score: float = 0.5
    priority: int = 5
    last_accessed: float = 0.0
    tier: str = "ram"

    @property
    def nbytes(self) -> int:
        return self.key.nbytes + self.value.nbytes

    @property
    def record_id(self) -> str:
        return f"{self.agent_id}:{self.layer_idx}"


@dataclass
class TransferEvent:
    """One measured encode, transfer, storage, or restore operation."""

    operation: str
    agent_id: str
    layer_idx: int | None
    source: str
    destination: str
    bytes_moved: int
    elapsed_ms: float
    storage_dtype: str = ""

    @property
    def bandwidth_gbps(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return (self.bytes_moved / 1024**3) / (self.elapsed_ms / 1000.0)


@dataclass
class ResumeMetrics:
    """Critical-path and complete-resume measurements for one activation."""

    agent_id: str
    critical_layers: int
    total_layers: int
    critical_ms: float
    total_ms: float = 0.0
    bytes_critical: int = 0
    bytes_total: int = 0
    source_tiers: set[str] = field(default_factory=set)


@dataclass
class ResumeHandle:
    """Partial page-in result; call ``wait()`` before using all layers."""

    critical: dict[int, tuple[Any, Any]]
    metrics: ResumeMetrics
    _future: Any = None

    def wait(self) -> dict[int, tuple[Any, Any]]:
        remaining = self._future.result() if self._future is not None else {}
        self._future = None
        return {**self.critical, **remaining}
