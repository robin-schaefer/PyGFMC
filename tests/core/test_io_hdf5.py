"""write_to_HDF5 / run_from_HDF5 / evaluate_observables: the per-run storage format.

Per-run grouping, the walker ensemble and PRNG stream, must-match validation
(including operator identity by content hash), bit-identical continuation,
and re-estimation from the stored raw series.
"""
import copy
import functools
import os
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import h5py
import pytest

from pygfmc import run_gfmc
from pygfmc.walkers import prepare_walkers_from_configs
from pygfmc import (write_to_HDF5, run_from_HDF5, evaluate_observables,
                    load_raw_from_HDF5)
from pygfmc.engine.io import (_read_from_HDF5, resume_from_HDF5, check_run_compatible,
                              pytree_fingerprint, MUST_MATCH, MAY_DIFFER)
from pygfmc.hamiltonian import SpinHalf, SzSzStructureFactor
from pygfmc.wavefunction import JastrowSpinHalf
from pygfmc.engine.estimators import dense_and_names, _estimate_observables

L = 8
EDGES = [(i, (i + 1) % L) for i in range(L)]
LAM = 8.0
P_ARRAY = [4, 8]


def _model():
    return SpinHalf.create(EDGES, J_x=0.0, J_y=0.0, J_z=-4.0, h_x=1.5)


def _zz(edge):
    return SpinHalf.create([edge], J_x=0.0, J_y=0.0, J_z=1.0, sign_check=False)


def _guide(beta=-0.2):
    return JastrowSpinHalf(L, EDGES, beta=beta)


def _sx_op():
    """Total transverse field as an off-diagonal measurement operator."""
    return SpinHalf.create(EDGES, J_x=0.0, J_y=0.0, J_z=0.0, h_x=-2.0,
                           sign_check=False)


def _write(path, out, **kw):
    """write_to_HDF5 for a run_gfmc 5-tuple ``(walkers, E, obs, stats, raw)``."""
    w, E, obs, stats, raw = out
    return write_to_HDF5(path, stats, w, E, obs, raw, **kw)


def _run(n_steps=200, n_walkers=256, step_size=1, diag=None, seed=0,
         guide=None, **kw):
    walkers = prepare_walkers_from_configs(jnp.ones((n_walkers, L)))
    kw.setdefault("rng_key", jax.random.PRNGKey(seed))
    kw.setdefault("warmup_steps", 40)
    return run_gfmc(walkers, n_steps, _model(), step_size, guide, P_ARRAY,
                    lam=LAM, diag_operators=diag, verbosity=0, **kw)


def test_run_groups_are_zero_based_and_self_contained(tmp_path):
    """Runs land in run_0, run_1, ... each self-contained; /raw is one shared series."""
    path = str(tmp_path / "a.h5")
    for seed in (0, 1):
        out = _run(seed=seed, diag={"zz": _zz(EDGES[0])})
        with warnings.catch_warnings():     # independent runs, not one chain
            warnings.filterwarnings("ignore", message=".*continuous chain.*")
            assert _write(path, out) == seed

    with h5py.File(path, "r") as f:
        assert f.attrs["format"] == "pygfmc"
        assert int(f.attrs["n_runs"]) == 2
        assert sorted(k for k in f if k.startswith("run")) == ["run_0", "run_1"]
        for n in (0, 1):
            g = f[f"run_{n}"]
            for key in ("energy", "observables", "parameters", "statistics"):
                assert key in g, f"run_{n}/{key} missing"
            assert g["energy"].shape == (len(P_ARRAY),)
            assert list(g["energy"].attrs["p_array"]) == P_ARRAY
            assert "raw" not in g
            assert g.attrs["raw_step_begin"] == 200 * n
            assert g.attrs["raw_step_end"] == 200 * (n + 1)
            assert "walker_hash" in g.attrs
        for key in ("energy", "log_weights", "diag"):
            assert f["raw/{}".format(key)].shape[0] == 400


def test_all_stats_entries_are_stored(tmp_path):
    """Nothing from stats is dropped -- only routed by the parameters/ prefix."""
    path = str(tmp_path / "a.h5")
    out = _run(diag={"zz": _zz(EDGES[0])})
    stats = out[3]
    _write(path, out)
    d = _read_from_HDF5(path, run=0)

    stored = set(d["statistics"]) | {"parameters/" + k for k in d["parameters"]}
    assert set(stats) == stored

    # the entries a campaign actually reads back
    for key in ("it_per_sec", "runtime", "total_wall_time", "avg_log_weight",
                "autocorrelation/energy_estimator", "autocorrelation/log_weights",
                "autocorrelation/diag/zz"):
        assert key in d["statistics"], key
    for key in ("n_steps", "n_walkers", "step_size", "lam", "p_array",
                "method", "hamiltonian", "verbosity", "config_dtype",
                "weight_dtype", "jax_enable_x64", "rng_key",
                "rng_key_next", "diag_operators"):
        assert key in d["parameters"], key


def test_guide_is_recorded_with_its_path(tmp_path):
    """The guide class and a has_guide flag are stored with the run."""
    path = str(tmp_path / "a.h5")
    _write(path, _run(n_steps=150, guide=_guide()))
    p = _read_from_HDF5(path)["parameters"]
    assert int(p["has_guide"]) == 1
    assert p["guide"] == "JastrowSpinHalf"

    _write(str(tmp_path / "b.h5"), _run(n_steps=120))
    p2 = _read_from_HDF5(str(tmp_path / "b.h5"))["parameters"]
    assert int(p2["has_guide"]) == 0


def test_resume_advances_the_stream_and_skips_the_warmup(tmp_path):
    """Resume skips the warm-up and starts on the key the previous run ended on."""
    path = str(tmp_path / "a.h5")
    _write(path, _run(n_steps=200))

    kw = resume_from_HDF5(path, hamiltonian=_model())
    assert kw["warmup_steps"] == 0          # already equilibrated
    assert kw["avg_log_weight"] is not None
    assert kw["rng_key"] is not None        # the key alone is the resume state
    assert kw["walkers"].config.shape == (256, L)

    _write(path, run_gfmc(kw.pop("walkers"), 150, _model(), 1, None,
                          P_ARRAY, lam=LAM, verbosity=0, **kw))

    p0 = _read_from_HDF5(path, 0)["parameters"]
    p1 = _read_from_HDF5(path, 1)["parameters"]
    assert int(p1["warmup_steps"]) == 0
    # one stream: run 1 starts on the key run 0 ended on
    assert np.array_equal(p0["rng_key_next"], p1["rng_key"])
    assert not np.array_equal(p0["rng_key"], p1["rng_key"])   # it advanced

    with h5py.File(path, "r") as f:
        assert int(f["current_walkers"].attrs["run"]) == 1
        assert np.array_equal(np.asarray(f["current_walkers"]["rng_key"]),
                              np.asarray(p1["rng_key_next"]))


def test_external_walker_file(tmp_path):
    """walker_filename stores the ensemble in a separate file that resume finds."""
    path, wf = str(tmp_path / "a.h5"), str(tmp_path / "walk.h5")
    _write(path, _run(n_steps=150), walker_filename=wf)
    with h5py.File(path, "r") as f:
        assert "current_walkers" not in f
        assert f.attrs["walker_filename"] == os.path.abspath(wf)
    assert resume_from_HDF5(path, hamiltonian=_model()
                            )["walkers"].config.shape == (256, L)


def test_resume_without_a_stored_ensemble_is_refused(tmp_path):
    """A file written without walkers cannot be resumed."""
    path = str(tmp_path / "a.h5")
    w, E, obs, stats, raw = _run(n_steps=120)
    write_to_HDF5(path, stats, None, E, obs, raw)      # no walkers
    with pytest.raises(ValueError, match="no walker ensemble"):
        resume_from_HDF5(path, hamiltonian=_model())


@pytest.mark.parametrize("param,value", [
    ("n_walkers", 512),
    ("lam", 9.0),
    ("step_size", 4),
    ("method", "single_step"),
    ("p_array", [4, 8, 16]),
    ("weight_dtype", "float64"),
])
def test_must_match_mismatch_is_refused(tmp_path, param, value):
    """Changing a MUST_MATCH parameter is refused on append."""
    path = str(tmp_path / "a.h5")
    _write(path, _run(n_steps=120))
    assert param in MUST_MATCH
    with pytest.raises(ValueError, match="parameter mismatch"):
        check_run_compatible(path, **{param: value})


@pytest.mark.parametrize("param,value", [
    ("n_steps", 999999),
    ("warmup_steps", 0),
    ("verbosity", 3),
])
def test_may_differ_is_allowed(tmp_path, param, value):
    """Changing a MAY_DIFFER parameter is accepted."""
    path = str(tmp_path / "a.h5")
    _write(path, _run(n_steps=120))
    assert param in MAY_DIFFER
    check_run_compatible(path, **{param: value})          # must not raise


def test_operators_are_compared_by_name_not_count(tmp_path):
    """Same count, different operators -> identical shapes, different meaning."""
    path = str(tmp_path / "a.h5")
    ops = {"zz01": _zz(EDGES[0]), "zz12": _zz(EDGES[1])}
    _write(path, _run(n_steps=120, diag=ops))

    check_run_compatible(path, diag_operators=ops)                # same
    with pytest.raises(ValueError, match="diag_operators"):        # renamed
        check_run_compatible(path, diag_operators={"zz01": None, "zz34": None})
    with pytest.raises(ValueError, match="diag_operators"):        # reordered
        check_run_compatible(path, diag_operators={"zz12": None, "zz01": None})
    with pytest.raises(ValueError, match="off_diag_operators"):
        check_run_compatible(path, off_diag_operators=["sx"])


def test_write_is_atomic_on_mismatch(tmp_path):
    """A refused append must leave the file exactly as it was."""
    path = str(tmp_path / "a.h5")
    _write(path, _run(n_steps=120, step_size=1))
    with pytest.raises(ValueError, match="parameter mismatch"):
        _write(path, _run(n_steps=120, step_size=4, seed=1))
    with h5py.File(path, "r") as f:
        assert int(f.attrs["n_runs"]) == 1
        assert "run_1" not in f


def test_rng_key_replay_is_refused(tmp_path):
    """Starting again on a key an earlier run started from (a replay) is refused; continuing is not."""
    path = str(tmp_path / "a.h5")
    _write(path, _run(n_steps=120))
    p = _read_from_HDF5(path)["parameters"]
    with pytest.raises(ValueError, match="RNG stream replay"):
        check_run_compatible(path, rng_key=p["rng_key"], n_steps=120)
    # continuing from where it ended is fine
    check_run_compatible(path, rng_key=p["rng_key_next"], n_steps=120)


def test_no_raw_means_no_rewindowing(tmp_path):
    """Without raw there is no /raw group and load_raw_from_HDF5 raises."""
    path = str(tmp_path / "a.h5")
    w, E, obs, stats, raw = _run(n_steps=120)
    write_to_HDF5(path, stats, w, E, obs, raw=None)
    with h5py.File(path, "r") as f:
        assert "raw" not in f                   # nothing to re-estimate from
    with pytest.raises(ValueError, match="no raw series"):
        load_raw_from_HDF5(path)


def test_fingerprint_is_stable_and_content_sensitive():
    """Equal contents give equal digests; any content change changes the digest."""
    assert pytree_fingerprint(_model()) == pytree_fingerprint(_model())
    assert pytree_fingerprint(None) == ""
    other = SpinHalf.create(EDGES, J_x=0.0, J_y=0.0, J_z=-4.1, h_x=1.5)
    assert pytree_fingerprint(_model()) != pytree_fingerprint(other)
    assert pytree_fingerprint(_zz(EDGES[0])) != pytree_fingerprint(_zz(EDGES[1]))
    assert pytree_fingerprint(_guide()) != pytree_fingerprint(_guide(-0.21))


def test_resume_verifies_the_objects_it_is_given(tmp_path):
    """The check belongs at resume time, not after the run has been paid for."""
    path = str(tmp_path / "a.h5")
    guide = _guide()
    ops = {"zz01": _zz(EDGES[0])}
    _write(path, _run(n_steps=150, guide=guide, diag=ops))

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # unchanged -> must not warn
        resume_from_HDF5(path, hamiltonian=_model(), guide=guide,
                         diag_operators=ops)

    with pytest.raises(ValueError, match="hamiltonian_hash"):
        resume_from_HDF5(path, hamiltonian=SpinHalf.create(
            EDGES, J_x=0.0, J_y=0.0, J_z=-4.1, h_x=1.5))
    with pytest.raises(ValueError, match="guide_hash"):
        resume_from_HDF5(path, guide=_guide(-0.21))
    with pytest.raises(ValueError, match="diag_operator_hashes"):
        resume_from_HDF5(path, diag_operators={"zz01": _zz(EDGES[1])})

    # nothing supplied -> nothing can be verified, and it says so
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resume_from_HDF5(path)
    assert any("could not verify" in str(c.message) for c in caught)


def test_fixed_parameters_live_once_at_the_file_level(tmp_path):
    """MUST_MATCH parameters are stored once at the top, per-run ones inside each run."""
    path = str(tmp_path / "a.h5")
    for seed in (0, 1):
        with warnings.catch_warnings():     # independent runs, not one chain
            warnings.filterwarnings("ignore", message=".*continuous chain.*")
            _write(path, _run(seed=seed))

    with h5py.File(path, "r") as f:
        # exactly one copy, at the top, and nothing fixed inside a run
        assert "parameters" in f
        fixed, per_run = set(f["parameters"]), set(f["run_0/parameters"])
        assert fixed <= set(MUST_MATCH)
        assert not (per_run & set(MUST_MATCH))
        assert not (fixed & per_run)
        for key in ("lam", "step_size", "p_array", "hamiltonian_hash",
                    "guide_hash", "diag_operator_hashes"):
            assert key in fixed, key
        for key in ("n_steps", "warmup_steps", "rng_key",
                    "verbosity"):
            assert key in per_run, key
        assert set(f["run_1/parameters"]) == per_run

    # a reader still sees one merged dict
    d = _read_from_HDF5(path, run=1)
    assert "lam" in d["parameters"] and "n_steps" in d["parameters"]
    assert set(d["parameters"]) == (set(d["fixed_parameters"])
                                    | set(d["run_parameters"]))


def test_run_from_HDF5_continues_the_chain(tmp_path):
    """One call: verify, load, run without warmup, same tuple as run_gfmc."""
    path = str(tmp_path / "a.h5")
    guide = _guide()
    ops = {"zz01": _zz(EDGES[0])}
    H = _model()

    _write(path, _run(step_size=2, guide=guide, diag=ops))

    out = run_from_HDF5(path, H, guide, ops, verbosity=0)
    assert len(out) == 5                                   # run_gfmc's tuple
    _write(path, out)

    p0 = _read_from_HDF5(path, 0)["parameters"]
    p1 = _read_from_HDF5(path, 1)["parameters"]
    assert int(p1["n_steps"]) == int(p0["n_steps"]) == 200
    assert int(p1["warmup_steps"]) == 0                    # never re-warm
    assert float(p1["lam"]) == LAM
    assert float(p1["step_size"]) == 2.0
    assert [int(v) for v in np.asarray(p1["p_array"]).ravel()] == P_ARRAY
    # one stream: run 1 starts on the key run 0 ended on
    assert np.array_equal(p0["rng_key_next"], p1["rng_key"])

    # explicit n_steps overrides the inherited length
    out2 = run_from_HDF5(path, H, guide, ops, n_steps=120, verbosity=0)
    _write(path, out2)
    p2 = _read_from_HDF5(path, 2)["parameters"]
    assert int(p2["n_steps"]) == 120
    assert np.array_equal(p1["rng_key_next"], p2["rng_key"])


def test_run_from_HDF5_refuses_a_different_calculation(tmp_path):
    """Changed objects are refused by hash, and the propagator cannot be overridden."""
    path = str(tmp_path / "a.h5")
    guide = _guide()
    ops = {"zz01": _zz(EDGES[0])}
    _write(path, _run(n_steps=150, guide=guide, diag=ops))

    with pytest.raises(ValueError, match="hamiltonian_hash"):
        run_from_HDF5(path, SpinHalf.create(EDGES, J_x=0.0, J_y=0.0,
                                            J_z=-4.1, h_x=1.5),
                      guide, ops, verbosity=0)
    with pytest.raises(ValueError, match="guide_hash"):
        run_from_HDF5(path, _model(), _guide(-0.21),
                      ops, verbosity=0)
    with pytest.raises(ValueError, match="diag_operators"):
        run_from_HDF5(path, _model(), guide, {"other": _zz(EDGES[0])},
                      verbosity=0)
    for param, value in (("lam", 9.0), ("step_size", 4), ("p_array", [4]),
                         ("method", "single_step")):
        with pytest.raises(ValueError, match="cannot be overridden"):
            run_from_HDF5(path, _model(), guide, ops, verbosity=0,
                          **{param: value})


def test_raw_is_one_continuous_series(tmp_path):
    """Runs append into a single /raw, and each run records its own slice."""
    path = str(tmp_path / "a.h5")
    H, ops = _model(), {"zz01": _zz(EDGES[0])}
    _write(path, _run(diag=ops))
    for _ in range(2):
        r = run_from_HDF5(path, H, None, ops, n_steps=150, verbosity=0)
        _write(path, r)

    with h5py.File(path, "r") as f:
        assert f["raw/energy"].shape == (500,)          # 200 + 150 + 150
        assert f["raw/log_weights"].shape == (500,)
        assert f["raw/diag"].shape[0] == 500
        spans = [(int(f[f"run_{n}"].attrs["raw_step_begin"]),
                  int(f[f"run_{n}"].attrs["raw_step_end"])) for n in range(3)]
        assert spans == [(0, 200), (200, 350), (350, 500)]
        hashes = [f[f"run_{n}"].attrs["walker_hash"] for n in range(3)]
        assert len(set(hashes)) == 3

    # the whole chain re-estimates as one series
    E_all, obs_all = evaluate_observables(load_raw_from_HDF5(path))
    assert sorted(E_all) == P_ARRAY
    assert set(obs_all) == {"zz01"}
    assert all(np.all(np.isfinite(np.asarray(v))) for v in obs_all["zz01"].values())
    # energy at windows the runs never used
    E_other, _ = evaluate_observables(load_raw_from_HDF5(path),
                                   p_array=[2, 6, 12])
    assert sorted(E_other) == [2, 6, 12]


def _walk_in_pieces(path, pieces, p_array, diag_ops, off_ops, seed=7,
                    step_size=2, **run_kwargs):
    """Walk sum(pieces) steps as len(pieces) runs through one file; return its /raw and walkers."""
    common = dict(diag_operators=diag_ops, off_diag_operators=off_ops,
                  verbosity=0, warmup_steps=0, avg_log_weight=2.8,
                  **run_kwargs)
    H = _model()
    out = run_gfmc(prepare_walkers_from_configs(jnp.ones((256, L))), pieces[0], H,
                   step_size, None, p_array, lam=LAM,
                   rng_key=jax.random.PRNGKey(seed), **common)
    _write(path, out)
    for n in pieces[1:]:
        out = run_from_HDF5(path, H, None, diag_ops, off_ops, n_steps=n,
                            verbosity=0)
        _write(path, out)
    with h5py.File(path, "r") as f:
        raw = {k: f["raw/" + k][()] for k in f["raw"]}
    return raw, out[0]


RAW_KEYS = ("energy", "log_weights", "diag", "off_diag", "off_diag_norm")


@pytest.mark.slow
def test_split_run_matches_contiguous_for_equal_and_unequal_pieces(tmp_path):
    """600 steps as one run, four equal and three unequal pieces give bit-identical /raw and estimates."""
    p_array = [4, 8, 32]
    diag_ops = {"zz01": _zz(EDGES[0]), "zz12": _zz(EDGES[1])}
    off_ops = {"sx": _sx_op()}
    splittings = {
        "contiguous":    [600],
        "four_equal":    [150, 150, 150, 150],
        "three_unequal": [100, 350, 150],
    }

    results, walkers = {}, {}
    for name, pieces in splittings.items():
        assert sum(pieces) == 600
        results[name], walkers[name] = _walk_in_pieces(
            str(tmp_path / f"{name}.h5"), pieces, p_array, diag_ops, off_ops)

    ref = results["contiguous"]
    for key in RAW_KEYS:
        assert ref[key].shape[0] == 600, key
    for name in ("four_equal", "three_unequal"):
        for key in RAW_KEYS:
            assert np.array_equal(ref[key], results[name][key]), (
                f"raw/{key} differs between 'contiguous' and '{name}'")
        assert bool(jnp.all(walkers["contiguous"].config
                            == walkers[name].config)), name
        assert bool(jnp.all(walkers["contiguous"].log_weight
                            == walkers[name].log_weight)), name

    # ... and so must the pooled re-estimate over the whole chain
    ref_E, ref_obs = evaluate_observables(
        load_raw_from_HDF5(str(tmp_path / "contiguous.h5")))
    for name in ("four_equal", "three_unequal"):
        E, obs = evaluate_observables(
            load_raw_from_HDF5(str(tmp_path / f"{name}.h5")))
        for p in p_array:
            assert float(E[p]) == pytest.approx(float(ref_E[p]), rel=0,
                                                abs=1e-12), f"{name} E p={p}"
            for op in ("zz01", "zz12", "sx"):
                assert float(obs[op][p]) == pytest.approx(
                    float(ref_obs[op][p]), rel=0, abs=1e-12), f"{name} {op} p={p}"


def test_split_run_with_off_diag_stride(tmp_path):
    """With a stride, a split run keeps the main chain bit-identical and every row labeled with its true step."""
    stride = 4
    p_array = [4, 8]
    diag_ops = {"zz01": _zz(EDGES[0])}
    off_ops = {"sx": _sx_op()}
    kw = dict(p_array=p_array, diag_ops=diag_ops, off_ops=off_ops,
              off_diag_stride=stride)

    ref, w_ref = _walk_in_pieces(str(tmp_path / "one.h5"), [400], **kw)
    for name, pieces in (("equal_multiple", [100, 100, 100, 100]),
                         ("unequal_multiple", [200, 80, 120]),
                         ("not_a_multiple", [150, 130, 120])):
        got, w = _walk_in_pieces(str(tmp_path / f"{name}.h5"), pieces, **kw)

        for key in ("energy", "log_weights", "diag"):
            assert np.array_equal(ref[key], got[key]), (
                f"raw/{key} differs between one run and '{name}'")
        assert bool(jnp.all(w_ref.config == w.config)), name

        # the grid is per call, so boundaries off the stride may drop a
        # measurement, but every stored row must decode to its true step
        steps = np.asarray(got["off_diag_steps"])
        got_od, _ = dense_and_names(got["off_diag"])
        assert steps.shape[0] == got_od.shape[0]
        assert np.all(np.diff(steps) > 0), f"{name}: step indices not increasing"
        assert steps[-1] < 400, f"{name}: step index past the chain"
        if all(x % stride == 0 for x in pieces):
            # boundaries on the grid: the measurement set is preserved
            assert np.array_equal(ref["off_diag_steps"], steps), name
            ref_od, _ = dense_and_names(ref["off_diag"])
            assert np.array_equal(ref_od, got_od), name

    with h5py.File(str(tmp_path / "one.h5"), "r") as f:
        # the stride is a fixed (must-match) parameter of the file
        assert int(np.ravel(f["parameters/off_diag_stride"][()])[0]) == stride
        assert f["raw/off_diag"].shape[0] == 400 // stride
        assert f["raw/energy"].shape[0] == 400          # energy stays per step


def test_relaunch_with_changed_hashes_is_refused(tmp_path):
    """Same names, changed contents: each change is refused by its own hash, before any stepping."""
    path = str(tmp_path / "a.h5")
    H = _model()
    guide = _guide()
    diag_ops = {"zz01": _zz(EDGES[0]), "zz12": _zz(EDGES[1])}
    off_ops = {"sx": _sx_op()}

    out = run_gfmc(prepare_walkers_from_configs(jnp.ones((256, L))), 150, H, 2,
                   guide, [4, 8], lam=LAM, diag_operators=diag_ops,
                   off_diag_operators=off_ops,
                   rng_key=jax.random.PRNGKey(0), warmup_steps=30,
                   verbosity=0)
    _write(path, out)

    run_from_HDF5(path, H, guide, diag_ops, off_ops, n_steps=20, verbosity=0)

    H_bad = SpinHalf.create(EDGES, J_x=0.0, J_y=0.0, J_z=-4.1, h_x=1.5)
    guide_bad = _guide(-0.21)
    diag_bad = {"zz01": _zz(EDGES[0]), "zz12": _zz(EDGES[3])}   # moved bond
    off_bad = {"sx": SpinHalf.create(EDGES, J_x=0.0, J_y=0.0, J_z=0.0,
                                     h_x=-1.0, sign_check=False)}

    for label, args, expected in (
            ("hamiltonian", (H_bad, guide, diag_ops, off_ops),
             "hamiltonian_hash"),
            ("guide",       (H, guide_bad, diag_ops, off_ops), "guide_hash"),
            ("diag op",     (H, guide, diag_bad, off_ops),
             "diag_operator_hashes"),
            ("off-diag op", (H, guide, diag_ops, off_bad),
             "off_diag_operator_hashes"),
            ("all at once", (H_bad, guide_bad, diag_bad, off_bad),
             "hamiltonian_hash"),
    ):
        with pytest.raises(ValueError, match="parameter mismatch") as exc:
            run_from_HDF5(path, *args, n_steps=20, verbosity=0)
        msg = str(exc.value)
        assert expected in msg, f"{label}: {expected} not named in\n{msg}"
        assert "pytree_fingerprint" in msg, label
        assert "retrained guide" in msg or "contents differ" in msg, label

    with h5py.File(path, "r") as f:
        assert int(f.attrs["n_runs"]) == 1


def test_diag_lead_in_matches_the_coherent_rows(tmp_path):
    """raw['additional_diag'] equals the next m_max diag rows of an uninterrupted run, without leaking into the run."""
    p_array = [8, 32]
    m_max = max(p_array) // 2
    H, ops = _model(), {"zz": _zz(EDGES[0])}
    common = dict(diag_operators=ops, verbosity=0, warmup_steps=0,
                  avg_log_weight=2.8)

    def fresh():
        return prepare_walkers_from_configs(jnp.ones((256, L)))

    _, _, _, _, raw_long = run_gfmc(fresh(), 400, H, 2, None, p_array, lam=LAM,
                                    rng_key=jax.random.PRNGKey(7), **common)
    w_short, _, _, _, raw_short = run_gfmc(fresh(), 200, H, 2, None,
                                           p_array, lam=LAM,
                                           rng_key=jax.random.PRNGKey(7),
                                           **common)
    lead_in = raw_short["additional_diag"]
    assert set(lead_in) == {"zz"}
    assert lead_in["zz"].shape == (m_max, len(p_array))
    assert np.array_equal(lead_in["zz"],
                          np.asarray(raw_long["diag"]["zz"])[200:200 + m_max])

    w_ref, _, _, _, raw_ref = run_gfmc(fresh(), 200, H, 2, None, p_array, lam=LAM,
                                       rng_key=jax.random.PRNGKey(7),
                                       verbosity=0, warmup_steps=0,
                                       avg_log_weight=2.8)
    assert raw_ref["additional_diag"] is None      # nothing diagonal to trace
    assert bool(jnp.all(w_short.config == w_ref.config))
    assert bool(jnp.all(w_short.log_weight == w_ref.log_weight))
    assert np.array_equal(np.asarray(raw_short["energy"]),
                          np.asarray(raw_ref["energy"]))


def test_stats_is_required(tmp_path):
    """Without parameters/ there is nothing to validate or continue from."""
    path = str(tmp_path / "a.h5")
    w, E, obs, stats, raw = _run(n_steps=120)
    with pytest.raises(ValueError, match="requires stats"):
        write_to_HDF5(path, None, w, E, obs, raw)
    with pytest.raises(ValueError, match="no 'parameters/' entries"):
        write_to_HDF5(path, {"it_per_sec": 1.0}, w, E, obs, raw)
    assert not os.path.exists(path)


def test_estimator_never_reads_the_invalid_leading_diag_rows():
    """Poisoning the zero-filled leading diag rows (below row p) leaves every estimate unchanged."""
    p_array = [4, 8, 32]
    _, _, _, _, raw = run_gfmc(
        prepare_walkers_from_configs(jnp.ones((256, L))), 400, _model(), 2, None,
        p_array, lam=LAM, diag_operators={"zz": _zz(EDGES[0])},
        rng_key=jax.random.PRNGKey(7), verbosity=0, warmup_steps=0,
        avg_log_weight=2.8)
    logw = np.asarray(raw["log_weights"])
    e_series = np.asarray(raw["energy"])
    diag = np.asarray(raw["diag"]["zz"])          # (n_steps, n_p), per name

    # the zero-fill is exactly p/2 deep, per column
    for j, p in enumerate(p_array):
        nonzero = np.flatnonzero(diag[:, j] != 0.0)
        assert nonzero.min() == p // 2, f"p={p}"

    _, base = _estimate_observables(p_array, logw, e_series,
                                   diag_op_np={"zz": diag})
    for p in p_array:
        poisoned = diag.copy()
        poisoned[:p] = 1e6                  # everything strictly below row p
        _, got = _estimate_observables(p_array, logw, e_series,
                                      diag_op_np={"zz": poisoned})
        assert got["zz"][p] == base["zz"][p], (
            f"p={p} reads a row below p, so the zero-filled lead-in leaks in")


@functools.lru_cache(maxsize=None)
def _cached_run_with_both_operators(n_steps, stride, seed):
    return run_gfmc(
        prepare_walkers_from_configs(jnp.ones((256, L))), n_steps, _model(), 1,
        None, P_ARRAY, lam=LAM, diag_operators={"zz": _zz(EDGES[0])},
        off_diag_operators={"sx": _sx_op()}, off_diag_stride=stride,
        rng_key=jax.random.PRNGKey(seed), warmup_steps=40, verbosity=0)


def _run_with_both_operators(n_steps=600, stride=1, seed=3):
    """One chain carrying a diagonal AND an off-diagonal observable.

    The run is deterministic, so it is computed once per argument set; each
    caller gets a deep copy, so no test can change what another one reads.
    """
    return copy.deepcopy(_cached_run_with_both_operators(n_steps, stride, seed))


def test_evaluate_observables_reproduces_the_run_it_came_from():
    """evaluate_observables(raw) with no further arguments reproduces the run's numbers exactly."""
    _, E, obs, _, raw = _run_with_both_operators()
    assert raw["p_array"] == P_ARRAY
    assert raw["diag_operators"] == ["zz"] and raw["off_diag_operators"] == ["sx"]
    E2, obs2 = evaluate_observables(raw)
    assert sorted(E2) == P_ARRAY and set(obs2) == {"zz", "sx"}
    for p in P_ARRAY:
        assert float(E2[p]) == pytest.approx(float(E[p]), rel=0, abs=1e-12)
        for name in ("zz", "sx"):
            assert float(obs2[name][p]) == pytest.approx(
                float(obs[name][p]), rel=0, abs=1e-12), f"{name} p={p}"


def test_load_raw_from_HDF5_round_trips_the_in_memory_raw(tmp_path):
    """load_raw_from_HDF5 returns raw under run_gfmc's keys (plus grid metadata) and estimates identically."""
    path = str(tmp_path / "a.h5")
    _, E, obs, stats, raw = _run_with_both_operators()
    write_to_HDF5(path, stats, None, E, obs, raw)

    loaded = load_raw_from_HDF5(path)
    for key in ("log_weights", "energy", "off_diag_steps"):
        assert np.array_equal(np.asarray(loaded[key]),
                              np.asarray(raw[key])), key
    for key in ("diag", "off_diag", "off_diag_norm"):
        assert isinstance(loaded[key], dict), key
        assert set(loaded[key]) == set(raw[key]), key
        for name, arr in raw[key].items():
            assert np.array_equal(np.asarray(loaded[key][name]),
                                  np.asarray(arr)), f"{key}[{name}]"
    assert loaded["p_array"] == P_ARRAY
    assert loaded["m_array"] == [p // 2 for p in P_ARRAY]
    assert loaded["diag_operators"] == ["zz"]
    assert loaded["off_diag_operators"] == ["sx"]

    E2, obs2 = evaluate_observables(loaded)          # no p_array needed
    for p in P_ARRAY:
        assert float(E2[p]) == pytest.approx(float(E[p]), rel=0, abs=1e-12)
        for name in ("zz", "sx"):
            assert float(obs2[name][p]) == pytest.approx(
                float(obs[name][p]), rel=0, abs=1e-12), f"{name} p={p}"


def test_evaluate_observables_rejects_a_grid_the_operators_are_not_on():
    """A p grid of the wrong length (or none) raises; unnamed dense series fall back to positional names."""
    _, _, _, _, raw = _run_with_both_operators()
    hand = {k: raw[k] for k in ("energy", "log_weights", "diag", "off_diag",
                                "off_diag_norm", "off_diag_steps")}
    with pytest.raises(ValueError, match="window"):
        evaluate_observables(hand, [4])
    with pytest.raises(ValueError, match="p_array"):
        evaluate_observables(hand)
    _, obs = evaluate_observables(hand, P_ARRAY)
    assert set(obs) == {"zz", "sx"}
    bare = {**hand, "off_diag": np.stack([hand["off_diag"]["sx"]], axis=1),
            "off_diag_norm": np.stack([hand["off_diag_norm"]["sx"]], axis=1)}
    _, obs_bare = evaluate_observables(bare, P_ARRAY)
    assert set(obs_bare) == {"zz", "off_diag_0"}


def test_evaluate_observables_energy_at_windows_the_run_never_used():
    """A different grid re-windows the energy and leaves the operators alone."""
    _, _, obs, _, raw = _run_with_both_operators()
    E, obs2 = evaluate_observables(raw, p_array=[2, 6, 12])
    assert sorted(E) == [2, 6, 12]
    assert all(np.isfinite(float(v)) for v in E.values())
    for name in ("zz", "sx"):
        assert sorted(obs2[name]) == P_ARRAY
        for p in P_ARRAY:
            assert float(obs2[name][p]) == pytest.approx(
                float(obs[name][p]), rel=0, abs=1e-12)


@pytest.mark.parametrize("stride", [1, 4])
def test_jackknife_two_blocks_equals_the_half_chain_split(stride):
    """n_blocks=2 jackknife error equals |x_first_half - x_second_half| / 2.

    The halves are rebuilt here by slicing raw (renumbering the second half's
    off-diagonal steps), so this checks the replica machinery independently.
    """
    _, _, _, _, raw = _run_with_both_operators(stride=stride)
    n = np.asarray(raw["energy"]).shape[0]
    blen = n // 2
    steps = np.asarray(raw["off_diag_steps"], dtype=np.int64)

    def half(keep_lo, keep_hi, shift):
        rows = (steps >= keep_lo) & (steps < keep_hi)
        return {
            "energy":       np.asarray(raw["energy"])[keep_lo:keep_hi],
            "log_weights":  np.asarray(raw["log_weights"])[keep_lo:keep_hi],
            "diag":         {nm: np.asarray(a)[keep_lo:keep_hi]
                             for nm, a in raw["diag"].items()},
            "off_diag":     {nm: np.asarray(a)[rows]
                             for nm, a in raw["off_diag"].items()},
            "off_diag_norm": {nm: np.asarray(a)[rows]
                              for nm, a in raw["off_diag_norm"].items()},
            "off_diag_steps": steps[rows] - shift,
            "diag_operators": ["zz"], "off_diag_operators": ["sx"],
            "p_array": P_ARRAY,
        }

    # replica 0 deletes the FIRST block, so it keeps the second half
    x0, o0 = evaluate_observables(half(blen, n, blen))
    x1, o1 = evaluate_observables(half(0, blen, 0))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_blocks=2 is few.*")
        E, obs, E_err, obs_err = evaluate_observables(raw, P_ARRAY, n_blocks=2)

    # the point estimates are untouched by asking for errors
    E_ref, obs_ref = evaluate_observables(raw)
    for p in P_ARRAY:
        assert float(E[p]) == float(E_ref[p])
        assert float(E_err[p]) == pytest.approx(
            abs(float(x0[p]) - float(x1[p])) / 2, rel=1e-9, abs=1e-12), f"p={p}"
        for name in ("zz", "sx"):
            assert float(obs[name][p]) == float(obs_ref[name][p])
            assert float(obs_err[name][p]) == pytest.approx(
                abs(float(o0[name][p]) - float(o1[name][p])) / 2,
                rel=1e-9, abs=1e-12), f"{name} p={p}"
            assert float(obs_err[name][p]) > 0.0


def test_jackknife_block_count_is_validated():
    """Too few usable blocks raise; a small block count warns."""
    _, _, _, _, raw = _run_with_both_operators(n_steps=200)
    with pytest.raises(ValueError, match="at least 2 usable blocks"):
        evaluate_observables(raw, P_ARRAY, n_blocks=1)
    with pytest.raises(ValueError, match="at least 2 usable blocks"):
        evaluate_observables(raw, P_ARRAY, n_blocks=400)   # fewer than 1 step/block
    with pytest.warns(UserWarning, match="is few"):
        evaluate_observables(raw, P_ARRAY, n_blocks=4)


def test_mixed_width_diagonal_operators_share_a_run(tmp_path):
    """A scalar and an n_k-valued diagonal operator share a run.

    Merging must not change any number (bitwise), both storage layouts must
    round-trip, and the jackknife must handle the ragged shapes.
    """
    pos = np.arange(L, dtype=float)[:, None]
    ks = 2 * np.pi * np.arange(3)[:, None] / L
    ops = {"zz": _zz(EDGES[0]), "Szz": SzSzStructureFactor(pos, ks)}

    w, E, obs, stats, raw = _run(diag=ops)
    assert isinstance(raw["diag"], dict), "raw['diag'] must be keyed by name"
    assert {k: v.shape for k, v in raw["diag"].items()} == {
        "zz": (200, len(P_ARRAY)), "Szz": (200, 3, len(P_ARRAY))}

    for name in ops:
        _, _, obs_one, _, _ = _run(diag={name: ops[name]})
        for p in P_ARRAY:
            np.testing.assert_array_equal(np.asarray(obs[name][p]),
                                          np.asarray(obs_one[name][p]))

    assert np.shape(stats["autocorrelation/diag/zz"]) == (len(P_ARRAY),)
    assert np.shape(stats["autocorrelation/diag/Szz"]) == (3, len(P_ARRAY))

    # ragged -> stored per name; uniform -> the dense layout
    mixed, uniform = str(tmp_path / "m.h5"), str(tmp_path / "u.h5")
    write_to_HDF5(mixed, stats, w, E, obs, raw)
    w2, E2, obs2, stats2, raw2 = _run(diag={"zz": _zz(EDGES[0]),
                                            "zz2": _zz(EDGES[1])})
    write_to_HDF5(uniform, stats2, w2, E2, obs2, raw2)
    with h5py.File(mixed, "r") as f:
        assert "diag_by_name" in f["raw"] and "diag" not in f["raw"]
        assert f["raw/diag_by_name/Szz"].shape == (200, 3, len(P_ARRAY))
    with h5py.File(uniform, "r") as f:
        assert "diag" in f["raw"] and "diag_by_name" not in f["raw"]
        assert f["raw/diag"].shape == (200, 2, len(P_ARRAY))

    # both layouts read back as a mapping and re-estimate exactly
    for path, ref in ((mixed, obs), (uniform, obs2)):
        back = load_raw_from_HDF5(path)
        assert isinstance(back["diag"], dict)
        _, obs_back = evaluate_observables(back)
        for name in ref:
            for p in P_ARRAY:
                np.testing.assert_array_equal(np.asarray(obs_back[name][p]),
                                              np.asarray(ref[name][p]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_blocks.*")
        _, _, e_err, o_err = evaluate_observables(load_raw_from_HDF5(mixed),
                                          P_ARRAY, n_blocks=4)
    assert np.shape(o_err["zz"][P_ARRAY[0]]) == ()
    assert np.shape(o_err["Szz"][P_ARRAY[0]]) == (3,)
    assert np.isfinite(e_err[P_ARRAY[0]])
    assert np.all(np.asarray(o_err["Szz"][P_ARRAY[0]]) >= 0)


def test_malformed_measurement_steps_are_rejected():
    """``off_diag_steps`` must be one-dimensional, one per row, integer-valued,
    strictly increasing and within the chain; nothing is cast or clipped."""
    from pygfmc import evaluate_observables
    rng = np.random.default_rng(0)
    n = 40
    raw = dict(energy=rng.normal(size=n), log_weights=rng.normal(0, .1, n), p_array=[2, 4],
               off_diag={"a": rng.normal(size=(n, 2))}, off_diag_norm={"a": rng.normal(0, .1, (n, 2))},
               off_diag_steps=np.arange(n), off_diag_operators=["a"], m_array=[1, 2])
    evaluate_observables(raw)
    for bad in (None, np.arange(n) + 0.5, np.arange(n)[::-1], np.arange(n - 1),
                np.arange(n) - 1, np.r_[np.arange(n - 1), n + 5],
                np.arange(n).reshape(2, -1), np.r_[0, 0, np.arange(2, n)]):
        with pytest.raises(ValueError, match="off_diag_steps"):
            evaluate_observables(dict(raw, off_diag_steps=bad))
