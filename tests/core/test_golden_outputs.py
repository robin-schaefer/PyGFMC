"""Golden-output regression: every propagator, for every Hamiltonian class and
every guide that accepts it, must reproduce stored results to float64 precision.

The statistical tests elsewhere accept anything within a statistical
tolerance, so a change that moves an energy or a reconfiguration decision
slightly passes them all.  This file pins the numbers of short float64 runs on
CPU over the full product

    {cont_time, multi_step, single_step}
    x {SpinHalf, Spin, BoseHubbard, ToricCode, PlaquetteModel}
    x {no guide, every guide class that accepts the Hamiltonian},

each run measuring that class's diagonal and off-diagonal operators (the
structure factors included), plus the gated off-diagonal path and
``statistics()``.

The runs use float64 (``jax_enable_x64``) and every Hamiltonian, operator and
guide is built with ``dtype=float64``.  Integer data (configurations, counts)
and strings must match exactly; floating-point data must agree within
``TOLERANCE`` (rtol 1e-10 for float64 data, about a million float64 rounding
steps but far below any real change; float32 for the few diagnostics the engine
accumulates in float32).  That absorbs the
last-bit differences between CPUs (different vector instructions, different
LAPACK kernels for the exact guides) while any change of the arithmetic, which
moves results by many orders of magnitude more, still fails.  The guide is
compared through its parameters, not through ``stats/parameters/guide_hash``:
the hash covers the exact bits of the parameters, which legitimately differ in
the last place between CPUs.  A failure means sampler or estimator arithmetic
changed; when the change is intended, regenerate the file::

    PYGFMC_WRITE_GOLDEN=1 pytest core/test_golden_outputs.py -q

XLA's fusion decisions can differ between JAX versions and between backends,
so the file records the JAX version and platform it was written with and the
comparison is skipped when either differs.
"""
import os
import platform as _platform
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygfmc import run_gfmc, statistics
from pygfmc.hamiltonian import (BoseHubbard, PlaquetteModel, Spin, SpinHalf,
                                SzSzStructureFactor, ToricCode, chain)
from pygfmc.walkers import (prepare_walkers_from_configs, prepare_walkers_random,
                            prepare_walkers_sz)
from pygfmc.wavefunction import (ClusterCPS, ClusterMLP, ExactGroundState,
                                 JastrowBoseHubbard, JastrowSpinHalf, JastrowSpin)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "reference_data", "golden_cpu.npz")
_SKIP_STATS = {"it_per_sec", "runtime", "total_wall_time"}
METHODS = ("cont_time", "multi_step", "single_step")
F64 = jnp.float64
# Production steps per run.  The autocorrelation times in ``stats`` need a few
# hundred measurements for the automatic window to exist; with fewer they are
# NaN for most observables.  The runs are dominated by compilation, so the
# length costs little.
N_STEPS = 300
# Agreement required for floating-point data: far above the rounding
# differences between CPUs, far below any change of the arithmetic.  A few
# branching diagnostics are accumulated in float32 inside the engine even in a
# float64 run, so float32 arrays get the float32 counterpart.
TOLERANCE = {np.dtype(np.float64): (1e-10, 1e-12),
             np.dtype(np.float32): (1e-5, 1e-6)}
# Recorded, but compared through the guide's parameters (see the module docstring).
_NOT_COMPARED = {"stats/parameters/guide_hash"}


def _key(name):
    """A PRNG key fixed by the case name, so adding a case never reseeds another."""
    return jax.random.PRNGKey(zlib.crc32(name.encode()))


def _flatten(prefix, obj, out):
    """Every array of a run result under a slash-separated key."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}", v, out)
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], str):
        out[prefix] = np.asarray(obj, dtype="S64")
    else:
        out[prefix] = np.asarray(obj)


def _record(case):
    result, guide = case
    walkers, energy, obs, stats = result[:4]
    out = {}
    if guide is not None:
        for i, leaf in enumerate(jax.tree_util.tree_leaves(guide)):
            out[f"guide/{i}"] = np.asarray(leaf)
    _flatten("config", walkers.config, out)
    _flatten("log_weight", walkers.log_weight, out)
    _flatten("e_loc", walkers.e_loc, out)
    _flatten("energy", energy, out)
    _flatten("obs", obs, out)
    _flatten("stats", {k: v for k, v in stats.items() if k not in _SKIP_STATS}, out)
    if len(result) == 5:
        _flatten("raw", result[4], out)
    return out


def _largest_sector(H):
    """The largest set of configurations connected by the moves of ``H``, with
    its ground state: ``(config, log_amp)`` for ``ExactGroundState``.

    The ring exchange of the plaquette model conserves more than S^z, so the
    Hilbert space splits into many disconnected sectors with the same ground
    energy; a full diagonalization returns an arbitrary vector of that
    degenerate space, generally without support on the walkers.  Within one
    connected sector the off-diagonal elements are negative and the
    Perron-Frobenius ground state is unique and positive.  Configurations
    outside the sector get the ``eps`` amplitude ``ExactGroundState`` uses.
    """
    values = np.asarray(H.local_hilbert_space, dtype=float).ravel()
    d, N = len(values), int(H.N)
    powers = d ** np.arange(N - 1, -1, -1)                  # site 0 most significant
    digits = np.array(np.unravel_index(np.arange(d ** N), (d,) * N)).T
    configs = jnp.asarray(values[digits], dtype=H.dtype)
    rates, succ = jax.vmap(lambda c: H.get_rates(c, return_configs=True))(configs)
    rates, succ = np.asarray(rates), np.asarray(succ)
    succ_idx = np.argmax(succ[..., None] == values, axis=-1) @ powers   # (d^N, M)

    parent = np.arange(d ** N)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(d ** N):
        for m in range(1, rates.shape[1]):
            if rates[i, m] != 0:
                parent[root(i)] = root(succ_idx[i, m])
    roots = np.array([root(i) for i in range(d ** N)])
    labels, counts = np.unique(roots, return_counts=True)
    members = np.flatnonzero(roots == labels[np.argmax(counts)])
    position = {int(i): k for k, i in enumerate(members)}
    mat = np.zeros((len(members), len(members)))
    for i in members:
        for m in range(rates.shape[1]):
            if m == 0 or rates[i, m] != 0:
                mat[position[int(i)], position[int(succ_idx[i, m])]] += rates[i, m]
    _, vecs = np.linalg.eigh(mat)
    log_amp = np.full(d ** N, np.log(1e-30))
    log_amp[members] = np.log(np.maximum(np.abs(vecs[:, 0]), 1e-30))
    return configs[members[0]], log_amp


def _exact_guide(H, log_amp=None):
    """``ExactGroundState`` with its amplitudes perturbed by a fixed pattern.

    With the exact ground state every walker's local energy equals E_0 up to
    rounding, so the energy autocorrelation time and the ancestor correlation
    are computed from rounding noise and differ between CPUs by O(1).  The
    perturbation (a deterministic 0.3 sin(1.7 n) on log psi) keeps the class's
    code path but gives the local energy a real spread, so every recorded
    statistic is well defined.
    """
    exact = ExactGroundState(H, log_amp=log_amp, dtype=F64)
    la = np.asarray(exact.log_amp, dtype=float)
    return ExactGroundState(H, log_amp=la + 0.3 * np.sin(1.7 * np.arange(la.size)),
                            dtype=F64)


# ── one system per Hamiltonian class ────────────────────────────────────────
# Each builder returns the Hamiltonian, the walkers, the diagonal and
# off-diagonal measurement operators (the latter with the measurement sign,
# i.e. rates >= 0), the cont_time step, the lam of the discrete propagators
# and one factory per guide that accepts the class.

def _nn_structure_factor(edges, k):
    """4 S^x_iS^x_j weighted by 1 + cos(k) on the nearest-neighbor bonds of a
    unit-spaced chain: the "SpSm+SpSp" structure-factor operator restricted to
    those bonds, built directly as StructureFactor's construction does."""
    W = 1.0 + np.cos(np.ones(len(edges)) * k)
    return SpinHalf.create(np.asarray(edges, dtype=np.int32),
                           J_x=jnp.asarray(4.0 * W, dtype=F64),
                           J_y=jnp.asarray(0.0 * W, dtype=F64),
                           J_z=0.0, N=8, dtype=F64, sign_check=False)


def _spinhalf():
    e = chain(8).edges
    H = SpinHalf.create(e, J_x=-1.0, J_y=-1.0, J_z=1.0, h_x=0.5, dtype=F64)
    pos = np.arange(8, dtype=float)[:, None]
    ks = (2 * np.pi * np.arange(3) / 8)[:, None]
    diag = {"zz": SpinHalf.create(e, J_z=1.0, dtype=F64),
            "sf": SzSzStructureFactor(pos, ks, dtype=F64)}
    off = {"spsm": SpinHalf.create(e, J_x=2.0, J_y=2.0, J_z=0.0, sign_check=False, dtype=F64),
           "sx": SpinHalf.create(e, h_x=-1.0, sign_check=False, dtype=F64),
           # k = pi/2: the weight 1 + cos(k) rounds to exactly 1, so without a
           # guide the series is exactly constant (not constant up to rounding)
           "sfx": _nn_structure_factor(e, np.pi / 2)}
    clusters = np.array([[i, (i + 1) % 8] for i in range(8)])
    guides = {"none": lambda: None,
              "jastrow": lambda: JastrowSpinHalf(8, e, beta=0.3, h=0.05, dtype=F64),
              "cps": lambda: ClusterCPS([{"clusters": clusters, "parity": True}],
                                        init="gaussian", seed=3, dtype=F64),
              "mlp": lambda: ClusterMLP([[clusters, [4]]], seed=4, dtype=F64),
              "exact": lambda: _exact_guide(H)}
    w = prepare_walkers_random(64, 8, _key("spinhalf walkers"))
    return dict(H=H, w=w, diag=diag, off=off, dt=0.2, lam=4.0, guides=guides)


def _spin1():
    e = chain(8).edges
    H = Spin.create(e, S=1.0, J_z=1.0, J_pm=-0.5, J_pmpm=-0.1, h_z=0.1, h_pm=-0.2, dtype=F64)
    diag = {"zz": Spin.create(e, S=1.0, J_z=1.0, dtype=F64)}
    off = {"pm": Spin.create(e, S=1.0, J_pm=0.5, sign_check=False, dtype=F64),
           "hpm": Spin.create(e, S=1.0, h_pm=0.2, sign_check=False, dtype=F64)}
    guides = {"none": lambda: None,
              "jastrow": lambda: JastrowSpin(8, e, beta=0.2, h=0.05, d=0.1, dtype=F64),
              "exact": lambda: _exact_guide(H)}
    w = prepare_walkers_sz(64, 8, _key("spin1 walkers"), sz=0, states=(-1.0, 0.0, 1.0))
    return dict(H=H, w=w, diag=diag, off=off, dt=0.1, lam=12.0, guides=guides)


def _bose():
    e = chain(6).edges
    H = BoseHubbard.create(e, t=1.0, eta=0.3, U=2.0, mu=0.5, h_pm=-0.2, n_max=2, dtype=F64)
    diag = {"nn": BoseHubbard.create(e, U=2.0, n_max=2, dtype=F64)}
    off = {"hop": BoseHubbard.create(e, t=-1.0, n_max=2, sign_check=False, dtype=F64),
           "pair": BoseHubbard.create(e, eta=-0.3, n_max=2, sign_check=False, dtype=F64)}
    guides = {"none": lambda: None,
              "jastrow": lambda: JastrowBoseHubbard(6, e, beta=0.05, h=0.1, d=0.3, dtype=F64),
              "exact": lambda: _exact_guide(H)}
    w = prepare_walkers_sz(64, 6, _key("bose walkers"), sz=6, states=(0.0, 1.0, 2.0))
    return dict(H=H, w=w, diag=diag, off=off, dt=0.1, lam=20.0, guides=guides)


def _toric():
    H = ToricCode.create(2, 2, h_x=0.3, h_z=0.1, dtype=F64)
    diag = {"bp": ToricCode.create(2, 2, A_s=0.0, B_p=1.0, dtype=F64)}
    off = {"star": ToricCode.create(2, 2, A_s=-1.0, B_p=0.0, sign_check=False, dtype=F64),
           "sx": ToricCode.create(2, 2, A_s=0.0, B_p=0.0, h_x=-1.0, sign_check=False, dtype=F64)}
    guides = {"none": lambda: None, "exact": lambda: _exact_guide(H)}
    w = prepare_walkers_random(64, H.N, _key("toric walkers"))
    return dict(H=H, w=w, diag=diag, off=off, dt=0.2, lam=6.0, guides=guides)


def _square_plaquettes(Lx, Ly):
    """The four-site plaquettes of a periodic ``Lx`` x ``Ly`` square lattice."""
    def idx(x, y):
        return (x % Lx) + Lx * (y % Ly)
    return np.array([[idx(x, y), idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)]
                     for x in range(Lx) for y in range(Ly)], dtype=np.int32)


def _plaquette():
    # 4 x 3 square lattice: its largest sector holds 15 configurations with
    # three distinct diagonal energies, where the 2 x 2 kagome sectors hold at
    # most four with one.
    plaquettes = _square_plaquettes(4, 3)
    H = PlaquetteModel.create(plaquettes, g=1.0, mu=-0.5, dtype=F64)
    diag = {"flip": PlaquetteModel.create(plaquettes, mu=1.0, dtype=F64)}
    off = {"ring": PlaquetteModel.create(plaquettes, g=-1.0, sign_check=False, dtype=F64)}
    config, log_amp = _largest_sector(H)          # all walkers in one sector
    w = prepare_walkers_from_configs(jnp.tile(config[None], (64, 1)))
    guides = {"none": lambda: None,
              "exact": lambda: _exact_guide(H, log_amp)}
    return dict(H=H, w=w, diag=diag, off=off, dt=0.3, lam=4.0, guides=guides)


SYSTEMS = {"spinhalf": _spinhalf, "spin1": _spin1, "bose": _bose, "toric": _toric,
           "plaquette": _plaquette}
GUIDES = {"spinhalf": ("none", "jastrow", "cps", "mlp", "exact"),
          "spin1": ("none", "jastrow", "exact"),
          "bose": ("none", "jastrow", "exact"),
          "toric": ("none", "exact"),
          "plaquette": ("none", "exact")}


def _run(system, guide, method, **extra):
    s = SYSTEMS[system]()
    name = f"{system}_{guide}_{method}" + ("_" + extra.pop("tag") if "tag" in extra else "")
    kw = dict(method=method, rng_key=_key(name), diag_operators=s["diag"],
              off_diag_operators=s["off"], warmup_steps=8, verbosity=0, **extra)
    step = s["dt"] if method == "cont_time" else 2
    if method != "cont_time":
        kw["lam"] = s["lam"]
    g = s["guides"][guide]()
    return run_gfmc(s["w"], N_STEPS, s["H"], step, g, [2, 4], **kw), g


def _statistics_case():
    s = SYSTEMS["spinhalf"]()
    g = s["guides"]["jastrow"]()
    _, st = statistics(s["w"], s["H"], 0.2, g, "cont_time",
                       _key("spinhalf_jastrow_statistics"), 0.5, M=4, quiet=True,
                       m_array=[1, 2, 3])
    return (s["w"], {}, {}, st), g


CASES = {f"{sy}_{g}_{m}": (lambda sy=sy, g=g, m=m: _run(sy, g, m))
         for sy, guides in GUIDES.items() for g in guides for m in METHODS}
# return_raw=False with a stride and a batch size: the equal-window gating path
CASES["spinhalf_jastrow_multi_step_gated"] = lambda: _run(
    "spinhalf", "jastrow", "multi_step", tag="gated", return_raw=False,
    off_diag_stride=2, off_diag_batch_size=16)
CASES["spinhalf_jastrow_statistics"] = _statistics_case


def _golden():
    if not os.path.exists(GOLDEN):
        return None
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def golden():
    if os.environ.get("PYGFMC_WRITE_GOLDEN") == "1":
        out = {"__jax_version__": np.asarray(jax.__version__, dtype="S32"),
               "__platform__": np.asarray(jax.default_backend(), dtype="S16"),
               "__machine__": np.asarray(_platform.machine(), dtype="S16")}
        with jax.enable_x64(True):
            for name, fn in CASES.items():
                for k, v in _record(fn()).items():
                    out[f"{name}/{k}"] = v
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        np.savez(GOLDEN, **out)
        pytest.skip(f"wrote {GOLDEN} with {len(out)} arrays")
    g = _golden()
    if g is None:
        pytest.skip(f"no golden file at {GOLDEN}; write it with PYGFMC_WRITE_GOLDEN=1")
    recorded_jax = g["__jax_version__"].item().decode()
    recorded_platform = g["__platform__"].item().decode()
    if jax.default_backend() != recorded_platform or jax.__version__ != recorded_jax:
        pytest.skip(f"golden file was written on {recorded_platform} with jax "
                    f"{recorded_jax}; this is {jax.default_backend()} with jax "
                    f"{jax.__version__}, where XLA may round differently")
    return g


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_output_is_reproduced(golden, name):
    """Configurations and counts match the stored float64 run exactly; weights,
    energies, observables, raw series, stats and guide parameters to TOLERANCE."""
    with jax.enable_x64(True):
        got = _record(CASES[name]())
    prefix = name + "/"
    stored = {k[len(prefix):]: golden[k] for k in golden.files if k.startswith(prefix)}
    assert stored, f"{name}: not in the golden file; regenerate it with PYGFMC_WRITE_GOLDEN=1"
    assert set(stored) == set(got), (
        f"{name}: recorded keys differ from the stored ones:\n"
        f"  missing now: {sorted(set(stored) - set(got))}\n"
        f"  new now:     {sorted(set(got) - set(stored))}")
    bad = []
    for k in sorted(set(stored) - _NOT_COMPARED):
        a, b = stored[k], np.asarray(got[k])
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append(f"{k}: shape/dtype {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
        elif a.dtype.kind in "fc":
            rtol, atol = TOLERANCE.get(a.dtype, TOLERANCE[np.dtype(np.float64)])
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                off = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
                d = np.abs(a.astype(complex) - b.astype(complex))
                bad.append(f"{k}: max |diff| {np.nanmax(d):.3e} in {int(off.sum())} entries "
                           f"(NaN in one run only: {int((np.isnan(a) != np.isnan(b)).sum())})")
        elif not np.array_equal(a, b):
            bad.append(f"{k}: differs")
    assert not bad, f"{name} no longer agrees with the golden run:\n  " + "\n  ".join(bad)
