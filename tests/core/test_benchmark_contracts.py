"""Fast negative tests for the benchmark harness, plus reference compatibility."""
import copy
import importlib
import json

import h5py
import pytest

from benchmark.helpers.models import QUSPIN_MODELS, TENPY_MODELS, PYROCHLORE_MODELS
from benchmark.helpers.reference_contract import ROOT, validate_reference
from benchmark.helpers import scoring
from benchmark.helpers import reference_contract


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(scoring, "expected_observables", lambda cfg: {"z_tot"})
    return {"id": "test", "N": 4, "params": {}}


def test_zero_reference_is_asserted(cfg):
    with pytest.raises(AssertionError):
        scoring.check_observables(cfg, "multi_step", {"z_tot": 0.0},
                                  {"z_tot": {8: 1.0}}, 0.08)


@pytest.mark.parametrize("refs,outputs", [({"z_tot": 0.0}, {}),
                                        ({}, {"z_tot": {8: 0.0}})])
def test_missing_observable_fails(cfg, refs, outputs):
    with pytest.raises(AssertionError):
        scoring.check_observables(cfg, "multi_step", refs, outputs, 0.08)


def test_small_zero_reference_error_passes(cfg):
    scoring.check_observables(cfg, "multi_step", {"z_tot": 0.0},
                              {"z_tot": {8: 0.01}}, 0.08)


def test_reference_exclusion_must_be_explicit_and_current(cfg):
    cfg["reference_unavailable"] = {"z_tot": "not exported by external solver"}
    assert not scoring.check_observable_contract(cfg, {}, {"z_tot": {8: 0.0}})
    with pytest.raises(AssertionError, match="obsolete"):
        scoring.check_observable_contract(cfg, {"z_tot": 0.0}, {"z_tot": {8: 0.0}})


@pytest.mark.parametrize("windows", [{}, {8: float("nan")}, {8: float("inf")}])
def test_invalid_observable_results_fail(cfg, windows):
    with pytest.raises(AssertionError):
        scoring.check_observable_contract(cfg, {"z_tot": 0.0}, {"z_tot": windows})


def test_missing_projection_window_fails(cfg):
    cfg["gfmc"] = {"p_array": [8, 16]}
    with pytest.raises(AssertionError, match="missing windows"):
        scoring.check_observable_contract(cfg, {"z_tot": 0.0}, {"z_tot": {8: 0.0}})


CASES = [(group, cfg) for group, models in
         (("quspin", QUSPIN_MODELS), ("tenpy", TENPY_MODELS), ("pyrochlore", PYROCHLORE_MODELS))
         for cfg in models]


@pytest.mark.parametrize("group,model", CASES, ids=[c[1]["id"] for c in CASES])
def test_committed_reference_contract(group, model):
    with h5py.File(ROOT / group / (model["id"] + ".h5"), "r") as f:
        validate_reference(f, model)


@pytest.mark.parametrize("change", ["params", "sector", "geometry"])
def test_stale_reference_contract_is_rejected(change):
    model = copy.deepcopy(QUSPIN_MODELS[0])
    original = model["build"]
    if change == "params":
        model["params"]["J_z"] += 0.25
    elif change == "sector":
        model["target_charge"] = 123
    else:
        model["build"] = lambda: (original()[0], list(reversed(original()[1])))
    with h5py.File(ROOT / "quspin" / (model["id"] + ".h5"), "r") as f:
        with pytest.raises(AssertionError):
            validate_reference(f, model)


@pytest.mark.parametrize("change", ["sha256", "normalization"])
def test_reference_bytes_and_normalization_are_pinned(tmp_path, monkeypatch, change):
    model = QUSPIN_MODELS[0]
    manifest = json.loads(reference_contract.MANIFEST.read_text())
    if change == "sha256":
        manifest[model["id"]]["sha256"] = "0" * 64
    else:
        manifest[model["id"]]["contract"]["normalization"] = "per-site energy"
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(reference_contract, "MANIFEST", path)
    with h5py.File(ROOT / "quspin" / (model["id"] + ".h5"), "r") as f:
        with pytest.raises(AssertionError):
            validate_reference(f, model)


@pytest.mark.parametrize("corruption", [None, "early_energy", "flippable_frac", "ring_flip_tot"])
def test_rk_assertions_check_all_windows_and_signed_observables(monkeypatch, corruption):
    from benchmark import test_gfmc_vs_pyrochlore_ed as bench
    model = next(m for m in PYROCHLORE_MODELS if m["phase"] == "rk_point")
    windows = model["gfmc"]["p_array"]
    energy = dict.fromkeys(windows, 0.0)
    with h5py.File(ROOT / "pyrochlore" / (model["id"] + ".h5"), "r") as f:
        obs = {name: dict.fromkeys(windows, float(f.attrs[name]))
               for name in ("flippable_frac", "ring_flip_tot")}
    if corruption == "early_energy":
        energy[min(windows)] = 1.0
    elif corruption:
        obs[corruption][max(windows)] *= -1
    monkeypatch.setattr(bench.gb, "run_benchmark", lambda *a, **k: (energy, obs))
    if corruption:
        with pytest.raises(AssertionError):
            bench.test_gfmc_matches_pyrochlore_reference(model)
    else:
        bench.test_gfmc_matches_pyrochlore_reference(model)


@pytest.mark.parametrize("module,attribute,models,function", [
    ("test_gfmc_vs_quspin", "OUTPUT_DIR", QUSPIN_MODELS, "test_gfmc_matches_quspin"),
    ("test_gfmc_vs_tenpy", "OUTPUT_DIR", TENPY_MODELS, "test_gfmc_matches_tenpy"),
    ("test_gfmc_vs_pyrochlore_ed", "REFERENCE_DIR", PYROCHLORE_MODELS,
     "test_gfmc_matches_pyrochlore_reference"),
])
def test_missing_requested_reference_fails_not_skips(tmp_path, monkeypatch, module,
                                                  attribute, models, function):
    bench = importlib.import_module("benchmark." + module)
    monkeypatch.setattr(bench, attribute, str(tmp_path))
    args = (models[0],) if attribute == "REFERENCE_DIR" else (models[0], "multi_step")
    with pytest.raises(pytest.fail.Exception, match="missing reference"):
        getattr(bench, function)(*args)
