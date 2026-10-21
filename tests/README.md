# PyGFMC tests

Two test suites for the `pygfmc` package in `../codes`.

| Folder | Cases | Runtime (CPU) | Purpose |
|---|---|---|---|
| [`core/`](core/README.md) | 587 | about 30 minutes | Exactness checks of the building blocks, checkpointing, guides, multi-device runs, and benchmark reference contracts. Run after every change. |
| [`benchmark/`](benchmark/README.md) | 72 | about 140 minutes | Broad external-reference regressions plus multi-seed small-system accuracy and convergence checks. Opt-in. |

**Each folder has its own README** that lists every test with what it checks
(`core/README.md`) and every benchmark case with its model, method, reference
and tolerance (`benchmark/README.md`).

## Running

```bash
pip install -e "./codes[test]"            # from the repository root
cd tests
pytest core -q                            # core suite
pytest core -q -m "not slow"              # without the 13 cases marked slow (about 20-25 min)
RUN_GFMC_BENCH=1 pytest benchmark -q      # benchmark suite (about 140 minutes)
pytest benchmark -q                       # without the flag: only the cheap reference-data check
```

The NetKet training tests in `core/` are skipped when NetKet is not installed or does
not import (it needs `pip install -e "./codes[netket]"` from the repository root).
QuSpin and TeNPy are needed only to regenerate the benchmark references, not to
run the benchmark: the reference values are stored in
`benchmark/reference_data/`.

The extended validation workflow runs both complete suites weekly, on published
releases, and on manual dispatch, with NetKet installed and slow tests enabled
(including simulated four-device tests). This is CPU validation, not GPU testing.
Use `pytest core benchmark --collect-only -q` for current case counts.

## Layout

```
tests/
├── conftest.py            path setup, per-test jax_enable_x64 scope, x64 fixture, NetKet sharding switch
├── core/                  core suite, conftest.py, reference.py, rates.py, guides.py, walkers.py, reference_data/
└── benchmark/
    ├── test_gfmc_vs_quspin.py, test_gfmc_vs_tenpy.py, test_gfmc_vs_pyrochlore_ed.py
    ├── test_small_system_accuracy.py  multi-seed ED and convergence checks
    ├── helpers/           model registry, run harness, scoring, reference generators
    └── reference_data/    stored references (quspin/, tenpy/, pyrochlore/)
```
