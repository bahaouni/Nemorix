# Contributing to Nemorix

Thank you for your interest in contributing to Nemorix! This document provides guidelines
for contributing to the project.

## Getting Started

1. **Fork the repository** and clone your fork locally.

2. **Set up the development environment:**
   ```bash
   cd nemorix
   pip install -e ".[dev]"
   ```

3. **Run the tests** to make sure everything works:
   ```bash
   python -m pytest tests/ -v
   ```

## How to Contribute

### Reporting Bugs

- Use the [GitHub Issues](https://github.com/nemorix-project/nemorix/issues) page.
- Include a minimal reproducible example.
- Describe what you expected to happen vs. what actually happened.
- Include your Python version and OS.

### Suggesting Features

- Open an issue with the **Feature Request** template.
- Describe the use case and why existing functionality doesn't cover it.

### Submitting Code

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines below.

3. Add or update tests for your changes.

4. Run the full test suite:
   ```bash
   python -m pytest tests/ -v
   ```

5. Submit a pull request against `main`.

## Code Style

- **Python 3.10+** — use modern type hints (`X | None` instead of `Optional[X]`).
- **No external dependencies** in the core (`src/nemorix/core/`, `src/nemorix/policies/`,
  `src/nemorix/simulation/`). The stdlib-only policy keeps the simulator lightweight.
- **Dataclasses** over plain classes for data containers.
- Keep functions focused — one function, one responsibility.
- Use descriptive variable names. Avoid single-letter names except in tight loops.

## Testing Guidelines

- Every physics constant must have a test that validates it against a published source.
- Eviction policy tests should use explicit block setups, not random data.
- Simulation integrity tests should verify invariants (GPU utilization ≤ 100%,
  cost > 0, etc.), not exact numeric outputs.
- Tests must be deterministic — use fixed seeds for any randomness.

## Project Structure

```
src/nemorix/
├── core/           # Data structures: KVBlock, AgentMemoryObject, TierManager, Scheduler
├── policies/       # Eviction policies: LRU (baseline), Semantic (innovation)
├── compression/    # Quantization utilities (FP16 → FP8 → INT4)
├── simulation/     # Discrete-event simulator and workload generator
└── api/            # Optional FastAPI server (requires extra dependencies)
```

## Hardware Constants

If you update any hardware constant (bandwidth, latency, cost), you **must**:

1. Add a comment citing the source (datasheet, benchmark, spec).
2. Update the corresponding test in `tests/test_accuracy.py`.
3. Re-run `benchmarks/run_simulation.py` and update `benchmarks/results.json`.

## License

By contributing to Nemorix, you agree that your contributions will be licensed
under the [MIT License](LICENSE).
