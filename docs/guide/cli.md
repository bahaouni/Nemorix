# CLI Reference

## `run_simulation.py` — Main Benchmark

Run a full 3-policy comparison simulation (No Offload vs LRU vs Nemorix):

```bash
python benchmarks/run_simulation.py [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--agents N` | `50` | Number of AI agents |
| `--tokens N` | `65536` | Max context window per agent (tokens) |
| `--hours N` | `24` | Simulation duration (simulated hours) |
| `--gpu-gb N` | `80` | GPU VRAM size (GB) |
| `--cxl-gb N` | `512` | CXL memory pool (GB) |
| `--ram-gb N` | `256` | CPU RAM (GB) |
| `--ssd-gb N` | `4000` | NVMe SSD (GB) |
| `--seed N` | `42` | Random seed (simulation is deterministic) |
| `--json` | off | Print full JSON output to stdout |

### Examples

```bash
# Default run (50 agents, 64K tokens, 24h)
python benchmarks/run_simulation.py

# Scale test: 100 agents, 32K token window
python benchmarks/run_simulation.py --agents 100 --tokens 32768

# Larger CXL pool: 1 TB
python benchmarks/run_simulation.py --cxl-gb 1024

# JSON output (pipe to jq, save to file, etc.)
python benchmarks/run_simulation.py --json > results.json

# One-week simulation for stable P99 estimates
python benchmarks/run_simulation.py --hours 168

# Reproduce the exact benchmark from the paper
python benchmarks/run_simulation.py --agents 50 --tokens 65536 --hours 24 --seed 42
```

### Output

```
  Metric                         No Offload        LRU      Nemorix
  ─────────────────────────────────────────────────────────────────
  Agents under SLA (<200ms)               0         50         50
  Max GPU-resident agents                 6          6          6
  Avg resume latency               1205.5ms      10.4ms      9.9ms
  P50 resume latency               1279.3ms      10.0ms      9.6ms
  P99 resume latency               1638.1ms      15.6ms     15.6ms
  GPU utilization                       90%        90%        91%
  Eviction accuracy                     N/A        42%        41%
  Cost per agent-hour (incl. GPU)     $7.01       $0.17      $0.17
```

Results are also written to `benchmarks/results.json`.

---

## `compare_policies.py` — Policy Sweep

Runs all three policies across multiple agent counts (10, 25, 50, 75, 100):

```bash
python benchmarks/compare_policies.py
```

Shows how each policy scales as the number of agents increases beyond what GPU VRAM can hold.

---

## `plot_results.py` — Generate Charts

Requires `matplotlib`:

```bash
pip install matplotlib
python benchmarks/plot_results.py
```

Reads `benchmarks/results.json` and saves a 4-panel comparison chart to
`benchmarks/nemorix_comparison.png`.

---

## `pytest` — Test Suite

```bash
# Full suite with verbose output
python -m pytest tests/ -v

# Quick pass/fail only
python -m pytest tests/ -q

# Specific file
python -m pytest tests/test_accuracy.py -v

# Specific test by name
python -m pytest tests/test_eviction.py -k "test_lru_evicts_oldest" -v
```

---

## REST API Server

```bash
pip install -e ".[api]"
uvicorn nemorix.api.server:app --reload --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/agents` | `POST` | Register a new agent |
| `/agents/{id}` | `GET` | Get agent state and metrics |
| `/agents` | `GET` | List all agents |
| `/agents/{id}/resume` | `POST` | Activate agent, returns latency |
| `/metrics` | `GET` | Live tier utilization and cost |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
