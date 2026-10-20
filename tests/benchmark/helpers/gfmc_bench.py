"""Shared GFMC driver for the benchmark: walker setup, measurement operators, run.

The measurement operators reproduce, with PyGFMC's ``sign_check=False``
raw-coefficient idiom, exactly the operators the reference generators
evaluate (``quspin_ref.py``, ``generate_tenpy_reference.py``,
``generate_pyrochlore_reference.py``), so a GFMC run compares directly
against the stored values.  Tolerances live in the test files.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pygfmc import run_gfmc
from pygfmc.walkers import Walker, prepare_walkers_random, prepare_walkers_sz, prepare_walkers_from_configs
from pygfmc.hamiltonian import BoseHubbard, PlaquetteModel, Spin, SpinHalf, ToricCode


# ── walker initialization ───────────────────────────────────────────────

def _walkers(configs, dtype):
    zeros = jnp.zeros(configs.shape[0], dtype=dtype)
    return Walker(config=configs.astype(dtype), log_weight=zeros, e_loc=zeros)


def _random_levels(n_walkers, N, n_levels, offset, rng_key, dtype):
    """iid per-site draws from ``offset + {0, 1, ..., n_levels-1}``."""
    levels = jax.random.randint(rng_key, (n_walkers, N), 0, n_levels)
    return _walkers(levels.astype(dtype) + dtype(offset), dtype)


def _bose_hubbard_fixed_n(n_walkers, N, n_max, target_n, rng_key, dtype):
    """Random occupations in [0, n_max] with exact total particle number."""
    base_fill, remainder = divmod(target_n, N)
    assert base_fill + (remainder > 0) <= n_max, f"target_n={target_n} unreachable"
    base = jnp.array([base_fill + 1] * remainder + [base_fill] * (N - remainder),
                     dtype=dtype)
    keys = jax.random.split(rng_key, n_walkers)
    return _walkers(jax.vmap(lambda k: jax.random.permutation(k, base))(keys), dtype)


def _from_states(states, n_walkers, rng_key, dtype):
    """Draw the population from a stored ensemble (the pyrochlore ice states)."""
    states = np.asarray(states)
    idx = jax.random.choice(rng_key, states.shape[0], (n_walkers,),
                            replace=n_walkers > states.shape[0])
    return _walkers(jnp.asarray(states)[idx], dtype)


def prepare_walkers_for(cfg, n_walkers, rng_key, dtype=jnp.float32):
    """Initial walkers for one model entry.

    ``cfg["target_charge"]``, when present, is the exact sum of the stored
    config values every walker starts with (S^z_tot for Spin, 2 S^z_tot for
    SpinHalf's +-1 encoding, the particle number for BoseHubbard) -- see the
    note in ``models.py``.  Otherwise the draw is iid per site, which spreads
    the population over many conserved sectors.
    """
    family, N = cfg["family"], cfg["N"]
    target = cfg.get("target_charge")
    if cfg.get("init_states") is not None:
        return _from_states(cfg["init_states"](), n_walkers, rng_key, dtype)
    if family == "Spin":
        S = cfg["params"]["S"]
        if target is not None:
            states = tuple(float(m) for m in np.arange(-S, S + 0.5, 1.0))
            return prepare_walkers_sz(n_walkers, N, rng_key, sz=target, states=states)
        return _random_levels(n_walkers, N, int(round(2 * S)) + 1, -S, rng_key, dtype)
    if family == "SpinHalf":
        if target is not None:
            return prepare_walkers_sz(n_walkers, N, rng_key, sz=target, states=(-1.0, 1.0))
        return prepare_walkers_random(n_walkers, N, rng_key)
    if family == "BoseHubbard":
        n_max = cfg["params"]["n_max"]
        if target is not None:
            return _bose_hubbard_fixed_n(n_walkers, N, n_max, target, rng_key, dtype)
        return _random_levels(n_walkers, N, n_max + 1, 0, rng_key, dtype)
    if family == "ToricCode":
        # star moves conserve every B_p: start in the all-ones sector
        return prepare_walkers_from_configs(jnp.ones((n_walkers, N), dtype=dtype))
    raise ValueError(f"unknown family {family!r}")


# ── measurement operators (mirror quspin_ref.py) ─────────────────────────

class _SumSquareOccupation:
    """Diagonal operator Sum_i n_i^2 (BoseHubbard ``n2_tot``)."""

    def __init__(self, N, dtype=jnp.float32):
        self.N = N
        self.dtype = dtype

    def diagonal_weight(self, config):
        return jnp.sum(config.astype(self.dtype) ** 2)


jax.tree_util.register_pytree_node(
    _SumSquareOccupation,
    lambda o: ((), (o.N, o.dtype)),
    lambda aux, _leaves: _SumSquareOccupation(*aux),
)


def _onehot(n, i, value):
    arr = np.zeros(n)
    arr[i] = value
    return arr


def measurement_operators(cfg, dtype=jnp.float32):
    """Return ``(diag_operators, off_diag_operators)`` dicts for ``run_gfmc``.

    Bond observables act on the first edge only, but are built from the FULL
    edge list with a one-hot coupling array so the operator has the same N
    as the lattice.
    """
    family, params, N = cfg["family"], cfg["params"], cfg["N"]
    diag_ops, off_diag_ops = {}, {}

    if family in ("Spin", "SpinHalf", "BoseHubbard"):
        _, edges = cfg["build"]()
        n_edges = len(edges)

    if family == "SpinHalf":
        diag_ops["z_tot"] = SpinHalf.create(edges, h_z=-1.0, N=N, dtype=dtype)
        off_diag_ops["pm_bond"] = SpinHalf.create(
            edges, _onehot(n_edges, 0, 2.0), _onehot(n_edges, 0, 2.0), 0.0,
            N=N, dtype=dtype, sign_check=False)
        off_diag_ops["x_tot"] = SpinHalf.create(edges, 0.0, 0.0, 0.0, h_x=-2.0, N=N,
                                                dtype=dtype, sign_check=False)
    elif family == "Spin":
        S = params["S"]
        diag_ops["z_tot"] = Spin.create(edges, S=S, h_z=-1.0, N=N, dtype=dtype)
        off_diag_ops["pm_bond"] = Spin.create(
            edges, S=S, J_pm=_onehot(n_edges, 0, 1.0), N=N, dtype=dtype, sign_check=False)
        off_diag_ops["pm_tot"] = Spin.create(edges, S=S, h_pm=-1.0, N=N,
                                             dtype=dtype, sign_check=False)
    elif family == "BoseHubbard":
        n_max = params["n_max"]
        diag_ops["n_tot"] = BoseHubbard.create(edges, mu=-1.0, n_max=n_max, N=N, dtype=dtype)
        diag_ops["n2_tot"] = _SumSquareOccupation(N, dtype=dtype)
        # H = -t(...) - eta(...), so a +1 measurement needs t = eta = -1
        off_diag_ops["hop_bond"] = BoseHubbard.create(
            edges, t=_onehot(n_edges, 0, -1.0), n_max=n_max, N=N,
            dtype=dtype, sign_check=False)
        off_diag_ops["pair_bond"] = BoseHubbard.create(
            edges, eta=_onehot(n_edges, 0, -1.0), n_max=n_max, N=N,
            dtype=dtype, sign_check=False)
    elif family == "ToricCode":
        Lx, Ly = params["Lx"], params["Ly"]
        n_plaq = n_stars = Lx * Ly
        diag_ops["Bp_avg"] = ToricCode.create(Lx, Ly, A_s=0.0, B_p=-1.0 / n_plaq,
                                              h_z=0.0, h_x=0.0, dtype=dtype)
        # H = -A_s sum prod sx - h_x sum sx: negative couplings measure +1
        off_diag_ops["As_avg"] = ToricCode.create(
            Lx, Ly, A_s=-1.0 / n_stars, B_p=0.0, h_z=0.0, h_x=0.0,
            dtype=dtype, sign_check=False)
        off_diag_ops["x_tot"] = ToricCode.create(
            Lx, Ly, A_s=0.0, B_p=0.0, h_z=0.0, h_x=-1.0, dtype=dtype, sign_check=False)
    elif family == "PlaquetteModel":
        h, _ = cfg["build"]()
        plaquettes = np.asarray(h.plaquettes)
        diag_ops["flippable_frac"] = PlaquetteModel.create(
            plaquettes, g=0.0, mu=1.0 / plaquettes.shape[0], dtype=dtype)
        # H = -g sum flip: g = -1 measures +1 per flip (rates must be >= 0)
        off_diag_ops["ring_flip_tot"] = PlaquetteModel.create(
            plaquettes, g=-1.0, mu=0.0, dtype=dtype, sign_check=False)
    else:
        raise ValueError(f"unknown family {family!r}")

    # forward walking is expensive; models.py enables it on a few entries only
    if not cfg.get("measure_off_diag", True):
        off_diag_ops = {}
    return diag_ops, off_diag_ops


# ── driver ───────────────────────────────────────────────────────────────

def methods_for(cfg):
    """Propagation schemes to run: ``cfg["methods"]`` or the single ``gfmc.method``."""
    return list(cfg.get("methods") or [cfg["gfmc"]["method"]])


def step_size_for(cfg, method):
    """Step size at matched projection depth across schemes.

    The discrete schemes take an integer number of sub-steps, each advancing
    imaginary time by ~1/lam; ``cont_time`` takes the imaginary time directly,
    so it gets ``step_size / lam``.
    """
    g = cfg["gfmc"]
    if method == "cont_time":
        return float(g["step_size"]) / float(g["lam"])
    return g["step_size"]


def run_benchmark(cfg, rng_key, dtype=jnp.float32, method=None):
    """Run GFMC for one model entry; return ``(energy_results, obs_results)``."""
    h, _ = cfg["build"]()
    g = cfg["gfmc"]
    method = method or g["method"]
    diag_ops, off_diag_ops = measurement_operators(cfg, dtype=dtype)

    rng_key, init_key = jax.random.split(rng_key)
    walkers = prepare_walkers_for(cfg, g["n_walkers"], init_key, dtype=dtype)

    _, energy_results, obs_results, _, _ = run_gfmc(
        walkers,
        n_steps=g["n_steps"],
        hamiltonian=h,
        lam=g["lam"],
        step_size=step_size_for(cfg, method),
        guide=None,
        p_array=g["p_array"],
        method=method,
        rng_key=rng_key,
        diag_operators=diag_ops,
        off_diag_operators=off_diag_ops,
        verbosity=0,
        warmup_steps=10000,
    )
    return energy_results, obs_results


def best_p_value(results):
    """``(value, p)`` at the largest window p (the most projected estimate)."""
    p_max = max(results.keys())
    return results[p_max], p_max
