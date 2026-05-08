# Installation

## Requirements

- **Python 3.10, 3.11, 3.12, or 3.13**
- No external dependencies for the core simulation (pure Python standard library)

## Install Options

### Core + tests (recommended for development)

```bash
git clone https://github.com/bahaouni/nemorix.git
cd nemorix
pip install -e ".[dev]"
```

### Core only (minimal)

```bash
pip install -e .
```

### With charts

```bash
pip install -e ".[plot]"
# enables: python benchmarks/plot_results.py → saves benchmarks/nemorix_comparison.png
```

### With REST API

```bash
pip install -e ".[api]"
# enables: uvicorn nemorix.api.server:app --reload --port 8000
```

### Everything

```bash
pip install -e ".[all]"
```

## Verify Installation

```bash
python -m pytest tests/ -q
# Expected: 74 passed in ~60s
```

```bash
python benchmarks/run_simulation.py
# Expected: prints comparison table in ~10s
```

## No-install Quick Run

If you don't want to install, set `PYTHONPATH` manually:

```bash
# Windows
set PYTHONPATH=src
python benchmarks/run_simulation.py

# Linux / macOS
PYTHONPATH=src python benchmarks/run_simulation.py
```

## Continuous Integration

The project runs its full test suite on every commit via GitHub Actions across:

| Python | Ubuntu | Windows | macOS |
|---|---|---|---|
| 3.10 | ✅ | ✅ | ✅ |
| 3.11 | ✅ | ✅ | ✅ |
| 3.12 | ✅ | ✅ | ✅ |
| 3.13 | ✅ | ✅ | ✅ |
