"""Measured GPU↔RAM/NVMe KV-cache prototype manager."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import perf_counter, time
from typing import Any

from nemorix.core.kv_block import KVBlock
from nemorix.runtime.codec import TensorCodec
from nemorix.runtime.models import ResumeHandle, ResumeMetrics, StoredLayer, TransferEvent
from nemorix.runtime.store import TieredLayerStore


class RuntimeKVManager:
    """Own tensor KV layers and move them through a measured external store.

    A layer is represented as ``(key_tensor, value_tensor)``. The manager is
    independent of tensor layout, so it can own synthetic tensors or tensors
    exposed by vLLM layer hooks. FP8 encoding and device copies are delegated to
    the codec. Restoring the first ``critical_fraction`` of layers is synchronous;
    remaining layers are restored in a background worker and joined with
    :meth:`ResumeHandle.wait`.
    """

    def __init__(
        self,
        codec: TensorCodec,
        store: TieredLayerStore,
        policy: object,
        *,
        device: str = "cuda",
        gpu_capacity_bytes: int | None = None,
        critical_fraction: float = 0.10,
    ) -> None:
        if not 0 < critical_fraction <= 1:
            raise ValueError("critical_fraction must be in (0, 1]")
        self.codec = codec
        self.store = store
        self.policy = policy
        self.device = device
        self.gpu_capacity_bytes = gpu_capacity_bytes
        self.critical_fraction = critical_fraction
        self.resident: dict[str, dict[int, tuple[Any, Any]]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.events: list[TransferEvent] = []
        self.resume_history: list[ResumeMetrics] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemorix-pagein")
        self._lock = Lock()

    def register_agent(
        self,
        agent_id: str,
        layers: dict[int, tuple[Any, Any]],
        *,
        num_tokens: int,
        priority: int = 5,
        importance: dict[int, float] | None = None,
        current_time: float | None = None,
    ) -> None:
        now = time() if current_time is None else current_time
        self.resident[agent_id] = dict(layers)
        self.metadata[agent_id] = {
            "num_tokens": num_tokens,
            "priority": priority,
            "importance": dict(importance or {}),
            "last_accessed": now,
        }
        set_priority = getattr(self.policy, "set_agent_priority", None)
        if set_priority is not None:
            set_priority(agent_id, priority)

    @property
    def gpu_used_bytes(self) -> int:
        return sum(self._tensor_nbytes(tensor) for layers in self.resident.values() for pair in layers.values() for tensor in pair)

    def ensure_gpu_capacity(self, required_bytes: int, current_time: float | None = None) -> list[str]:
        """Offload policy-selected resident layers until the GPU budget fits."""
        if self.gpu_capacity_bytes is None:
            return []
        deficit = self.gpu_used_bytes + required_bytes - self.gpu_capacity_bytes
        if deficit <= 0:
            return []
        now = time() if current_time is None else current_time
        candidates = self._resident_blocks()
        victims = self.policy.select_victims(candidates, deficit, now)
        evicted: list[str] = []
        for victim in victims:
            self.offload_layer(victim.agent_id, victim.layer_idx, current_time=now)
            evicted.append(victim.block_id)
        return evicted

    def offload_agent(self, agent_id: str, current_time: float | None = None) -> list[TransferEvent]:
        """Encode all resident layers as FP8 and move them to RAM/NVMe."""
        now = time() if current_time is None else current_time
        before = len(self.events)
        for layer_idx in sorted(list(self.resident.get(agent_id, {}))):
            self.offload_layer(agent_id, layer_idx, current_time=now)
        return self.events[before:]

    def offload_layer(self, agent_id: str, layer_idx: int, current_time: float | None = None) -> TransferEvent:
        now = time() if current_time is None else current_time
        pair = self.resident[agent_id][layer_idx]
        source_bytes = sum(self._tensor_nbytes(tensor) for tensor in pair)
        start = perf_counter()
        key = self.codec.encode(pair[0])
        value = self.codec.encode(pair[1])
        self.codec.synchronize(self.device)
        elapsed = (perf_counter() - start) * 1000.0
        meta = self.metadata[agent_id]
        record = StoredLayer(
            agent_id=agent_id,
            layer_idx=layer_idx,
            key=key,
            value=value,
            num_tokens=meta["num_tokens"],
            importance_score=meta["importance"].get(layer_idx, 0.5),
            priority=meta["priority"],
            last_accessed=meta["last_accessed"],
        )
        self.store.put(record, now)
        with self._lock:
            del self.resident[agent_id][layer_idx]
        event = TransferEvent(
            "offload",
            agent_id,
            layer_idx,
            self.device,
            record.tier,
            source_bytes,
            elapsed,
            key.storage_dtype,
        )
        self.events.append(event)
        return event

    def resume_agent(self, agent_id: str, *, critical_layers: int | None = None) -> ResumeHandle:
        """Synchronously load critical layers and asynchronously load the rest."""
        layer_indices = self.store.layers_for(agent_id)
        if not layer_indices:
            raise KeyError(f"No offloaded layers for agent {agent_id!r}")
        count = critical_layers
        if count is None:
            count = max(1, math.ceil(len(layer_indices) * self.critical_fraction))
        count = min(max(1, count), len(layer_indices))
        critical_ids = layer_indices[:count]
        remaining_ids = layer_indices[count:]

        started = perf_counter()
        critical, critical_bytes, tiers = self._restore_layers(agent_id, critical_ids)
        critical_ms = (perf_counter() - started) * 1000.0
        metrics = ResumeMetrics(
            agent_id=agent_id,
            critical_layers=count,
            total_layers=len(layer_indices),
            critical_ms=critical_ms,
            bytes_critical=critical_bytes,
            source_tiers=tiers,
        )
        self.resume_history.append(metrics)

        def restore_remaining() -> dict[int, tuple[Any, Any]]:
            remaining, remaining_bytes, remaining_tiers = self._restore_layers(agent_id, remaining_ids)
            metrics.total_ms = (perf_counter() - started) * 1000.0
            metrics.bytes_total = critical_bytes + remaining_bytes
            metrics.source_tiers.update(remaining_tiers)
            self.metadata[agent_id]["last_accessed"] = time()
            observe = getattr(self.policy, "observe_access", None)
            if observe is not None:
                observe(agent_id, self.metadata[agent_id]["last_accessed"])
            return remaining

        if remaining_ids:
            future = self._executor.submit(restore_remaining)
        else:
            metrics.total_ms = critical_ms
            metrics.bytes_total = critical_bytes
            future = None
        return ResumeHandle(critical=critical, metrics=metrics, _future=future)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _restore_layers(self, agent_id: str, layer_indices: list[int]) -> tuple[dict[int, tuple[Any, Any]], int, set[str]]:
        restored: dict[int, tuple[Any, Any]] = {}
        total_bytes = 0
        tiers: set[str] = set()
        for layer_idx in layer_indices:
            record = self.store.get(agent_id, layer_idx)
            start = perf_counter()
            key = self.codec.decode(record.key, self.device)
            value = self.codec.decode(record.value, self.device)
            self.codec.synchronize(self.device)
            elapsed = (perf_counter() - start) * 1000.0
            pair = (key, value)
            with self._lock:
                self.resident.setdefault(agent_id, {})[layer_idx] = pair
            restored[layer_idx] = pair
            total_bytes += record.nbytes
            tiers.add(record.tier)
            self.events.append(
                TransferEvent("page_in", agent_id, layer_idx, record.tier, self.device, record.nbytes, elapsed, record.key.storage_dtype)
            )
            self.store.remove(agent_id, layer_idx)
        return restored, total_bytes, tiers

    def _resident_blocks(self) -> list[KVBlock]:
        blocks: list[KVBlock] = []
        for agent_id, layers in self.resident.items():
            meta = self.metadata[agent_id]
            for layer_idx, pair in layers.items():
                blocks.append(
                    KVBlock(
                        block_id=f"{agent_id}:{layer_idx}",
                        agent_id=agent_id,
                        layer_idx=layer_idx,
                        num_tokens=meta["num_tokens"],
                        size_bytes=sum(self._tensor_nbytes(tensor) for tensor in pair),
                        importance_score=meta["importance"].get(layer_idx, 0.5),
                        last_accessed=meta["last_accessed"],
                    )
                )
        return blocks

    @staticmethod
    def _tensor_nbytes(tensor: Any) -> int:
        nbytes = getattr(tensor, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if callable(numel) and callable(element_size):
            return int(numel() * element_size())
        raise TypeError("Tensor must expose nbytes or numel()/element_size()")
