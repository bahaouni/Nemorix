"""Thin adapter between vLLM paged KV buffers and ``RuntimeKVManager``.

The adapter intentionally avoids importing private vLLM classes. vLLM's V1
connector API changes rapidly, while ``register_kv_caches`` consistently exposes
layer-name → tensor mappings. This class handles the common cache layout whose
first axis is K/V and second axis is physical block. A production connector can
call these methods from ``save_kv_layer`` and ``wait_for_layer_load``.
"""

from __future__ import annotations

import re
from time import time
from typing import Any

from nemorix.runtime.manager import RuntimeKVManager
from nemorix.runtime.models import ResumeHandle


class VLLMPagedCacheAdapter:
    """Capture and restore request-owned physical blocks in vLLM KV tensors."""

    def __init__(self, manager: RuntimeKVManager) -> None:
        self.manager = manager
        self.kv_caches: dict[str, Any] = {}
        self._layer_names: list[str] = []

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register the layer buffers supplied by vLLM's V1 connector API."""
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")
        self.kv_caches = dict(kv_caches)
        self._layer_names = sorted(kv_caches, key=self._layer_sort_key)

    def capture_request(
        self,
        agent_id: str,
        block_ids_by_layer: dict[str, list[int]],
        *,
        num_tokens: int,
        priority: int = 5,
        importance: dict[int, float] | None = None,
    ) -> None:
        """Gather request blocks from paged GPU buffers and register the agent.

        The gathered tensors are compact copies, so releasing vLLM's physical
        blocks after this method does not invalidate the offload operation.
        """
        self._require_registered()
        layers: dict[int, tuple[Any, Any]] = {}
        for layer_idx, layer_name in enumerate(self._layer_names):
            ids = block_ids_by_layer.get(layer_name)
            if not ids:
                continue
            cache = self.kv_caches[layer_name]
            self._validate_cache(cache, layer_name)
            indices = self._index_tensor(cache, ids)
            gathered = cache.index_select(1, indices).contiguous()
            layers[layer_idx] = (gathered[0].clone(), gathered[1].clone())
        if not layers:
            raise ValueError("No vLLM blocks were selected for capture")
        self.manager.register_agent(
            agent_id,
            layers,
            num_tokens=num_tokens,
            priority=priority,
            importance=importance,
            current_time=time(),
        )

    def offload_request(self, agent_id: str) -> None:
        """FP8-encode a previously captured request into RAM/NVMe."""
        self.manager.offload_agent(agent_id)

    def start_restore(self, agent_id: str, critical_layers: int | None = None) -> ResumeHandle:
        """Start partial page-in; critical layers are available on return."""
        return self.manager.resume_agent(agent_id, critical_layers=critical_layers)

    def inject_restored(
        self,
        restored: dict[int, tuple[Any, Any]],
        block_ids_by_layer: dict[str, list[int]],
    ) -> None:
        """Scatter compact restored tensors into newly allocated vLLM blocks."""
        self._require_registered()
        for layer_idx, pair in restored.items():
            layer_name = self._layer_names[layer_idx]
            ids = block_ids_by_layer.get(layer_name)
            if not ids:
                raise ValueError(f"No destination blocks for {layer_name}")
            cache = self.kv_caches[layer_name]
            self._validate_cache(cache, layer_name)
            indices = self._index_tensor(cache, ids)
            if pair[0].shape[0] != len(ids) or pair[1].shape[0] != len(ids):
                raise ValueError(f"Restored block count does not match {layer_name}")
            cache[0].index_copy_(0, indices, pair[0])
            cache[1].index_copy_(0, indices, pair[1])

    @staticmethod
    def restore_all(handle: ResumeHandle) -> dict[int, tuple[Any, Any]]:
        """Join background page-in and return critical plus remaining layers."""
        return handle.wait()

    def _require_registered(self) -> None:
        if not self.kv_caches:
            raise RuntimeError("Call register_kv_caches() before capture or restore")

    @staticmethod
    def _validate_cache(cache: Any, layer_name: str) -> None:
        if getattr(cache, "ndim", 0) < 3 or cache.shape[0] != 2:
            raise ValueError(
                f"{layer_name} must use [2, num_blocks, ...] K/V layout; "
                f"received shape {getattr(cache, 'shape', None)}"
            )

    @staticmethod
    def _index_tensor(cache: Any, ids: list[int]) -> Any:
        torch = __import__("torch")
        return torch.tensor(ids, dtype=torch.long, device=cache.device)

    @staticmethod
    def _layer_sort_key(name: str) -> tuple[int, str]:
        numbers = re.findall(r"\d+", name)
        return (int(numbers[-1]) if numbers else 0, name)
