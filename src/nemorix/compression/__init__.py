"""Compression utilities for KV-cache quantization during tier migration."""

from __future__ import annotations
from nemorix.core.kv_block import KVBlock, COMPRESSION_RATIOS


def estimate_compressed_size(size_bytes: int, from_dtype: str, to_dtype: str) -> int:
    if from_dtype not in COMPRESSION_RATIOS or to_dtype not in COMPRESSION_RATIOS:
        return size_bytes
    factor = COMPRESSION_RATIOS[to_dtype] / COMPRESSION_RATIOS[from_dtype]
    return max(1, int(size_bytes * factor))


def quality_loss_estimate(from_dtype: str, to_dtype: str) -> float:
    """Estimated quality loss percentage based on published benchmarks."""
    loss_table = {
        ("fp16", "fp8"): 0.3,
        ("fp16", "int4"): 1.8,
        ("fp8", "int4"): 1.5,
        ("fp8", "fp16"): 0.0,  # no loss going up
        ("int4", "fp16"): 0.0,
        ("int4", "fp8"): 0.0,
    }
    return loss_table.get((from_dtype, to_dtype), 0.0)
