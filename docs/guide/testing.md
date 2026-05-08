# Running Tests

Nemorix has **74 tests** across 5 files. All tests are deterministic and run in ~60 seconds.

## Run All Tests

```bash
python -m pytest tests/ -v
```

## Test Files

| File | Tests | What it validates |
|---|---|---|
| `test_accuracy.py` | 39 | Hardware physics formulas: bandwidth, latency, cost — each test cites the hardware source it validates |
| `test_eviction.py` | 5 | LRU correctness, semantic policy correctness, priority protection |
| `test_tier_manager.py` | 5 | Tier capacity math, migration latency, compression, cost formula |
| `test_scheduler.py` | 3 | Agent lifecycle transitions, idle suspension |
| `test_simulation_integrity.py` | 22 | Full simulation physical invariants, policy ordering guarantees, scaling behavior |

## What Each Test Verifies

### `test_accuracy.py` — Hardware physics

```python
def test_cxl_bandwidth_1gib_block():
    """
    CXL bandwidth: 64 GB/s (Samsung CMM-D CXL 2.0 PCIe 5.0 x16 datasheet)
    Transfer of 1 GiB should take: 1 / 64 * 1000 + 0.005 = 15.630 ms
    """
    block = KVBlock(..., size_bytes=1024**3)
    latency = compute_transfer_latency(block, tier="cxl")
    assert abs(latency - 15.630) < 0.001
```

Every test cites the source spec — this is how you know the numbers trace back to hardware.

### `test_eviction.py` — Policy correctness

```python
def test_lru_evicts_oldest():
    # 3 agents with different last_accessed times
    # LRU must always select the one with oldest timestamp
    ...

def test_semantic_protects_high_attention():
    # Agent A: attention score 0.9 (important blocks)
    # Agent B: attention score 0.1 (low importance)
    # Semantic policy must evict B even if A is older
    ...

def test_priority_protection():
    # priority-9 agent must never be evicted over priority-1 agent
    # when all else is equal
    ...
```

### `test_simulation_integrity.py` — Physical invariants

```python
def test_no_offload_worse_than_lru():
    # no_offload avg latency must always exceed LRU avg latency
    ...

def test_nemorix_sla_geq_lru():
    # Nemorix serves >= as many agents under SLA as LRU
    ...

def test_gpu_utilization_bounded():
    # GPU utilization must be between 0% and 100%
    ...

def test_cost_positive():
    # Cost per agent-hour must be > 0 for all policies
    ...
```

## Running Specific Tests

```bash
# One file
python -m pytest tests/test_eviction.py -v

# One test by name (substring match)
python -m pytest -k "test_semantic" -v

# Tests matching a pattern
python -m pytest -k "latency or bandwidth" -v

# Stop after first failure
python -m pytest tests/ -x -v

# With coverage (requires pytest-cov)
pip install pytest-cov
python -m pytest tests/ --cov=src/nemorix --cov-report=html
```

## Continuous Integration

Every push runs the full suite via GitHub Actions on:
- Python 3.10, 3.11, 3.12, 3.13
- Ubuntu, Windows, macOS

See `.github/workflows/ci.yml` for the full workflow.
