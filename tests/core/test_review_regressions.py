"""Regression coverage for numerical, checkpoint and API review findings."""
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygfmc import run_gfmc, write_to_HDF5, run_from_HDF5, evaluate_observables
from pygfmc.engine import evolution, driver
from pygfmc.engine.estimators import estimate_average_log_branch_weight
from pygfmc.engine.estimators import autocorrelation
from pygfmc.engine.io import resume_from_HDF5
from pygfmc.hamiltonian import SpinHalf, Spin, BoseHubbard, PlaquetteModel
from pygfmc.walkers import Walker
from pygfmc.wavefunction import JastrowSpinHalf, ClusterCPS
from pygfmc.wavefunction.generic_netket_training import train_with_netket


def walkers(n=8, sites=2):
    return Walker(jnp.ones((n, sites)), jnp.zeros(n), jnp.zeros(n))


@pytest.mark.parametrize("hopping", [0.0, 2e-9, 2e-12])
def test_tiny_hopping_does_not_overflow_geometric_hold(hopping):
    h = SpinHalf.create([], N=1, h_x=hopping)
    w = walkers(32, 1)
    out = evolution.evolve_walkers(w, h, 1, None, "multi_step",
                                  jax.random.PRNGKey(0), 0, lam=1)
    np.testing.assert_array_equal(out.config, w.config)
    assert np.all(np.isfinite(out.log_weight))


def small_run(diag=None):
    h = SpinHalf.create([(0, 1)], J_z=0.2, h_x=1.0)
    out = run_gfmc(walkers(), 12, h, 0.1, None, [2],
                   diag_operators=diag, rng_key=jax.random.PRNGKey(0),
                   avg_log_weight=estimate_average_log_branch_weight(walkers(), h, None),
                   verbosity=0)
    return h, out


def write(path, out, **kwargs):
    w, energy, obs, stats, raw = out
    write_to_HDF5(str(path), stats, w, energy, obs, raw, verbose=False, **kwargs)


def test_checkpoint_switches_source_and_honors_override(tmp_path):
    h, out = small_run()
    path, external = tmp_path / "chain.h5", tmp_path / "walkers.h5"
    write(path, out)
    walkers()._replace(config=jnp.zeros((8, 2))).save(str(external))
    with pytest.raises(ValueError, match="walker_hash"):
        resume_from_HDF5(str(path), hamiltonian=h, walker_filename=str(external))
    next_out = run_from_HDF5(str(path), h, n_steps=12, verbosity=0)
    write(path, next_out, walker_filename=str(external))
    resumed = resume_from_HDF5(str(path), hamiltonian=h)
    for got, expected in zip(resumed["walkers"], next_out[0]):
        np.testing.assert_array_equal(got, expected)
    # Explicit external override must not silently use the embedded group.
    with h5py.File(external, "a") as f:
        f["walkers/config"][...] = 0
    with pytest.raises(ValueError, match="walker_hash"):
        resume_from_HDF5(str(path), hamiltonian=h, walker_filename=str(external))
    # A later embedded checkpoint supersedes the stale external file hint.
    later = run_gfmc(next_out[0], 12, h, 0.1, None, [2],
                     avg_log_weight=next_out[3]["avg_log_weight"],
                     rng_key=next_out[3]["parameters/rng_key_next"], verbosity=0)
    write(path, later)
    resumed = resume_from_HDF5(str(path), hamiltonian=h, config_dtype=jnp.int8)
    np.testing.assert_array_equal(resumed["walkers"].config, later[0].config)


def test_resume_without_raw_handles_diagonal_lead_in(tmp_path):
    diag = {"z": SpinHalf.create([], N=2, h_z=1)}
    h, out = small_run(diag)
    path = tmp_path / "chain.h5"
    write(path, out)
    resumed = run_from_HDF5(str(path), h, diag_operators=diag, n_steps=12,
                           return_raw=False, verbosity=0)
    assert len(resumed) == 4
    assert np.isfinite(resumed[1][2])


@pytest.mark.parametrize("h", [Spin.create([], N=2, S=1, h_pm=-1),
                               BoseHubbard.create([(0, 1)], n_max=2)])
def test_nonergodic_default_training_sampler_rejected_before_netket(h):
    with pytest.raises(ValueError, match="explicit ergodic sampler"):
        train_with_netket(h, None)


@pytest.mark.parametrize("rho", [-0.8, 0.8])
def test_autocorrelation_handles_both_signs(rho):
    rng = np.random.default_rng(9)
    x = np.zeros(100000)
    for i in range(1, len(x)):
        x[i] = rho * x[i - 1] + rng.normal()
    tau, window = autocorrelation(x)
    assert window > 1
    assert tau == pytest.approx(0.5 * (1 + rho) / (1 - rho), rel=0.25)


class NestedGuide:
    incremental = True

    def init_state(self, config, cache):
        return {"config": (config,)}

    def log_diff_from_operator(self, config, op, cache, state=None):
        return jnp.zeros(op.M_total)

    def update_state(self, state, config, sites, deltas, cache):
        return {"config": (state["config"][0].at[sites].add(deltas),)}


jax.tree_util.register_pytree_node(NestedGuide, lambda x: ((), None),
                                  lambda aux, children: NestedGuide())


@pytest.mark.parametrize("method", ["single_step", "multi_step", "cont_time"])
def test_nested_incremental_state_tracks_configuration(method):
    h = Spin.create([], N=1, S=1, h_pm=-0.5)
    guide = NestedGuide()
    w = walkers(8, 1)
    state = jax.vmap(lambda x: guide.init_state(x, None))(w.config)
    out, _, state_out = evolution.evolve_walkers(
        w, h, 4, guide, method, jax.random.PRNGKey(8), 0,
        lam=2, state=state, return_rates=True)
    np.testing.assert_array_equal(state_out["config"][0], out.config)


def test_float64_constructor_precision_and_roundtrip(tmp_path, x64):
    with x64():
        value = 1 + 2**-40
        j = JastrowSpinHalf(2, [(0, 1)], np.array([value]),
                            h=np.array([value, value]), dtype=jnp.float64)
        c = ClusterCPS([dict(clusters=[[0, 1]], parity=False)],
                       amplitudes=[np.full(4, value)], dtype=jnp.float64)
        assert float(j.beta[0]) == value
        np.testing.assert_array_equal(j.h, [value, value])
        for i, guide in enumerate([j, c]):
            path = str(tmp_path / f"guide{i}.h5")
            guide.save(path)
            loaded = type(guide).load(path, dtype=jnp.float64)
            for a, b in zip(jax.tree_util.tree_leaves(guide),
                            jax.tree_util.tree_leaves(loaded)):
                np.testing.assert_array_equal(a, b)
        assert any(np.all(np.asarray(x) == value)
                   for x in jax.tree_util.tree_leaves(c)
                   if jnp.issubdtype(x.dtype, jnp.floating))


def test_geometry_rejects_self_edges_and_fractional_indices():
    with pytest.raises(ValueError, match="self-edges"):
        SpinHalf.create([(0, 0)], J_x=-1, J_y=-1)
    with pytest.raises(ValueError, match="integers"):
        PlaquetteModel.create([[0.9, 1.9]], g=1)


def test_unsupported_rng_checkpoint_does_not_overwrite(tmp_path):
    path = str(tmp_path / "walkers.h5")
    w = walkers()
    w.save(path, rng_key=jax.random.key(0))
    before = (tmp_path / "walkers.h5").read_bytes()
    with pytest.raises(ValueError, match="threefry2x32"):
        w.save(path, rng_key=jax.random.key(0, impl="rbg"))
    assert (tmp_path / "walkers.h5").read_bytes() == before
    _, key, _ = Walker.load(path)
    np.testing.assert_array_equal(jax.random.split(key),
                                 jax.random.split(jax.random.PRNGKey(0)))


def test_unsupported_rng_fails_before_running_or_writing(tmp_path):
    key = jax.random.key(0, impl="rbg")
    h = SpinHalf.create([], N=2, h_x=1)
    with pytest.raises(ValueError, match="threefry2x32"):
        run_gfmc(walkers(), 12, h, 0.1, None, [2], rng_key=key)
    path = tmp_path / "invalid.h5"
    with pytest.raises(ValueError, match="threefry2x32"):
        write_to_HDF5(str(path), {"parameters/rng_key_next": key}, walkers())
    assert not path.exists()


def test_shadow_rates_respect_small_positive_budget(monkeypatch):
    h = SpinHalf.create([], N=2, h_x=1)
    op = SpinHalf.create([], N=2, h_x=-1, sign_check=False)
    w = walkers()
    monkeypatch.setattr(driver, "_RATES_CARRY_MAX_BYTES", 1)
    assert driver._initial_carried(w, h, None, None)[0] is None
    original = evolution.initial_carried
    decisions = []

    def record(*args, **kwargs):
        decisions.append(kwargs["with_rates"])
        return original(*args, **kwargs)

    monkeypatch.setattr(evolution, "initial_carried", record)
    evolution.forward_walk_operator(w, op, h, 0.1, None, "cont_time",
                                    jax.random.PRNGKey(0), 0, [1])
    assert decisions == [False]
    needed = len(w.config) * h.M_total * w.log_weight.dtype.itemsize
    assert evolution.rates_fit_budget(len(w.config), h, w.log_weight.dtype, needed)
    assert not evolution.rates_fit_budget(len(w.config), h, w.log_weight.dtype, needed - 1)


def test_pmap_shadow_budget_counts_whole_population(monkeypatch):
    from pygfmc.engine import multi_gpu
    n_devices = jax.local_device_count()
    h = SpinHalf.create([], N=2, h_x=1)
    op = SpinHalf.create([], N=2, h_x=-1, sign_check=False)
    w = multi_gpu.shard_walkers(walkers(4 * n_devices), n_devices)
    needed = 4 * n_devices * h.M_total * w.log_weight.dtype.itemsize
    monkeypatch.setattr(driver, "_RATES_CARRY_MAX_BYTES", needed - 1)
    monkeypatch.setattr(multi_gpu, "_PMAP_CARRY_RATES", True)
    decisions = []
    original = multi_gpu.initial_carried

    def record(*args, **kwargs):
        decisions.append(kwargs["with_rates"])
        return original(*args, **kwargs)

    monkeypatch.setattr(multi_gpu, "initial_carried", record)
    out = jax.pmap(lambda shard: multi_gpu.forward_walk_operator_global(
        shard, op, h, 0.1, None, "cont_time", jax.random.PRNGKey(0), 0, [1]),
        axis_name="devices")(w)
    jax.block_until_ready(out)
    assert decisions == [False]


def test_jackknife_tail_truncates_all_observable_series():
    steps = np.arange(0, 41, 2)
    raw = {"energy": np.arange(41.), "log_weights": np.zeros(41),
           "p_array": [2], "m_array": [1],
           "diag": {"z": np.arange(41.)[:, None]},
           "off_diag": {"x": steps[:, None] * 1.0},
           "off_diag_norm": {"x": np.zeros((len(steps), 1))},
           "off_diag_steps": steps}
    trimmed = {**raw, "energy": raw["energy"][:40], "log_weights": np.zeros(40),
               "diag": {"z": raw["diag"]["z"][:40]},
               "off_diag": {"x": raw["off_diag"]["x"][:-1]},
               "off_diag_norm": {"x": raw["off_diag_norm"]["x"][:-1]},
               "off_diag_steps": steps[:-1]}
    with pytest.warns(UserWarning, match="Discarding 1 trailing"):
        result = evaluate_observables(raw, n_blocks=20)
    expected = evaluate_observables(trimmed, n_blocks=20)
    for got, want in zip(jax.tree_util.tree_leaves(result),
                         jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(got, want)
    integer_valued = evaluate_observables(
        {**trimmed, "off_diag_steps": steps[:-1].astype(float)}, n_blocks=20)
    for got, want in zip(jax.tree_util.tree_leaves(integer_valued),
                         jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(got, want)
