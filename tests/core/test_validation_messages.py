"""Argument validation: every bad input raises its own message, before any work.

One parametrized table of ``(callable, kwargs, exception, message)`` covers the
guide constructors (``ClusterCPS``, ``ClusterMLP``), the engine's input checks,
the Hamiltonian constructors, ``ExactGroundState``, the spin-1/2 guides'
operator-layout check and ``evaluate_observables``.  The engine cases run with
the warm-up stages, the positivity probe and the production scan replaced by a
function that fails the test, so a check that fires only after walking has
started is caught.  A protocol smoke test checks that the built-in classes
satisfy ``pygfmc.Operator`` / ``pygfmc.Guide`` and that the package version is
the one in ``codes/pyproject.toml``.
"""
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pygfmc
from pygfmc import evaluate_observables, run_gfmc
from pygfmc.engine import driver
from pygfmc.engine.evolution import require_guide_protocol, require_operator_protocol
from pygfmc.hamiltonian import (BoseHubbard, PlaquetteModel, Spin, SpinHalf, ToricCode,
                                chain, kagome)
from pygfmc.walkers import prepare_walkers_from_configs, prepare_walkers_sz
from pygfmc.wavefunction import (ClusterCPS, ClusterMLP, ExactGroundState, JastrowBoseHubbard,
                                 JastrowSpin, JastrowSpinHalf)

_N = 4
_H = SpinHalf.create(chain(_N).edges, J_x=-1.0, J_y=-1.0, J_z=1.0)
_XX = SpinHalf.create(chain(_N).edges, J_x=-1.0, J_y=-1.0, sign_check=False)   # M_total > 1
_W = prepare_walkers_sz(8, _N, jax.random.PRNGKey(0), sz=0)
_TORIC = ToricCode.create(2, 2, A_s=1.0, B_p=1.0, h_x=0.2)
_PAIRS = np.asarray([[0, 1], [1, 2], [2, 3]])          # (n_patches, 2) clusters
_PATCH = np.asarray([[0, 1]])


class _NoRatios:
    """A guide object exposing none of the ratio routines."""


def _run(**overrides):
    """run_gfmc on 8 walkers of a 4-site chain, with one argument changed."""
    kw = dict(walkers=_W, n_steps=20, hamiltonian=_H, step_size=0.1, guide=None,
              p_array=[2], rng_key=jax.random.PRNGKey(1), warmup_steps=8, verbosity=0)
    kw.update(overrides)
    return run_gfmc(**kw)


@pytest.fixture
def no_walking(monkeypatch):
    """Fail the test if the driver starts any walk."""
    def walked(*args, **kwargs):
        raise AssertionError("validation fired only after walking had started")
    for name in ("_warmup_stage", "_positivity_probe", "_production"):
        monkeypatch.setattr(driver, name, walked)


ENGINE = [
    *[(f"nonfinite step {value}", dict(step_size=value), ValueError,
       "step_size must be finite") for value in (np.nan, np.inf, -np.inf)],
    *[(f"nonfinite shift {method} {value}", dict(method=method, step_size=2, lam=value),
       ValueError, "lam must be finite")
      for method in ("single_step", "multi_step") for value in (np.nan, np.inf, -np.inf)],
    ("no rng_key", dict(rng_key=None), ValueError, "run_gfmc requires rng_key"),
    ("zero walkers", dict(walkers=prepare_walkers_from_configs(jnp.zeros((0, _N)))),
     ValueError, "At least one walker is required"),
    pytest.param("step_size zero", dict(step_size=0.0), ValueError, "step_size must be > 0"),
    pytest.param("step_size negative", dict(step_size=-1.0), ValueError,
                 "step_size must be > 0"),
    pytest.param("non-integer discrete step", dict(step_size=2.5, method="multi_step", lam=3.0),
                 ValueError, "method='multi_step' requires an integer step_size"),
    ("hamiltonian without get_rates", dict(hamiltonian=type("Bare", (), {"M_total": 2})()),
     TypeError, "hamiltonian 'Bare' is missing required attribute(s) get_rates"),
    ("off-diagonal operator without get_rates",
     dict(off_diag_operators={"bad": type("Bare", (), {"M_total": 1})()}), TypeError,
     "off-diagonal operator 'bad' 'Bare' is missing required attribute(s) get_rates"),
    ("guide without ratio routines", dict(guide=_NoRatios()), TypeError,
     "Guide '_NoRatios' must implement at least one of log_diff_from_operator"),
    ("batch size not dividing the walkers", dict(off_diag_batch_size=3), ValueError,
     "off_diag_batch_size=3 must divide N_walkers=8"),
    ("max(p) not below n_steps", dict(n_steps=8, p_array=[8]), ValueError,
     "every p value must be smaller than n_steps: got n_steps=8 and max(p)=8"),
    ("off-diagonal run too short", dict(n_steps=10, p_array=[8], off_diag_operators={"xx": _XX}),
     ValueError, "off-diagonal operators need n_steps > 1.5*max(p)"),
    ("equal windows hold no stride step",
     dict(n_steps=13, p_array=[8], off_diag_stride=4, return_raw=False,
          off_diag_operators={"xx": _XX}),
     ValueError, "measure only the steps 8..8 (n_steps=13"),
    ("odd p with an off-diagonal operator", dict(p_array=[3], off_diag_operators={"xx": _XX}),
     ValueError, "All p values must be divisible by 2 when off_diag_operators have "
                 "off-diagonal elements (M_total > 1); got [3]"),
]


@pytest.mark.parametrize("label,overrides,exc,message", ENGINE,
                         ids=[c.values[0] if hasattr(c, "marks") else c[0] for c in ENGINE])
def test_engine_rejects_bad_input_before_walking(no_walking, label, overrides, exc, message):
    """Each bad driver argument raises its message before any step is taken."""
    with pytest.raises(exc, match=re.escape(message)):
        _run(**overrides)


def _bad_mlp_params():
    """Weights for a [2, 3, 1] network passed with a [2, 4, 1] architecture."""
    return [[(np.zeros((2, 3)), np.zeros(3)), (np.zeros((3, 1)), np.zeros(1))]]


CONSTRUCTORS = [
    # ClusterCPS
    ("cps no groups", lambda: ClusterCPS([]), "groups must contain at least one cluster group"),
    ("cps unwrapped list spec", lambda: ClusterCPS([_PAIRS, None, True]),
     "groups looks like one [clusters, symmetries, parity] spec passed unwrapped"),
    ("cps amplitude count", lambda: ClusterCPS([[_PAIRS, None, True]], amplitudes=[0, 0]),
     "amplitudes must have one entry per patch group (1); got 2"),
    ("cps dict without clusters", lambda: ClusterCPS([dict(parity=True)]),
     "group spec dict must contain 'clusters'"),
    ("cps unknown dict key", lambda: ClusterCPS([dict(clusters=_PAIRS, amplitudes=None)]),
     "unknown key(s) in ClusterCPS group spec: ['amplitudes'] -- 'amplitudes' is a "
     "TOP-LEVEL ClusterCPS argument"),
    ("cps bare spec", lambda: ClusterCPS([42]),
     "a cluster group must be a [clusters, symmetries, parity] list or a dict; got a bare int"),
    ("cps two-field list", lambda: ClusterCPS([[_PAIRS, None]]),
     "a list group spec must give all three fields explicitly: "
     "[clusters, symmetries, parity]; got 2"),
    ("cps parity not a bool", lambda: ClusterCPS([[_PAIRS, None, 1]]),
     "the third field of a group spec is parity and must be a bool; got int"),
    ("cps one-dimensional clusters", lambda: ClusterCPS([[np.arange(3), None, False]]),
     "clusters must be 2-D; got (3,)"),
    ("cps cluster_type length", lambda: ClusterCPS([dict(clusters=_PAIRS, cluster_type=[0])]),
     "cluster_type shape (1,) must equal (3,)"),
    ("cps negative cluster_type",
     lambda: ClusterCPS([dict(clusters=_PAIRS, cluster_type=[0, -1, 0])]),
     "group 0: cluster_type contains negative ids"),
    ("cps symmetry not a permutation",
     lambda: ClusterCPS([dict(clusters=_PATCH, symmetries=[[0, 0]])]),
     "every row of automorphisms must be a permutation of 0..m-1"),
    ("cps unknown init", lambda: ClusterCPS(_PAIRS, init="uniform"),
     "init must be 'zero' or 'gaussian'; got 'uniform'"),
    ("cps amplitude shape", lambda: ClusterCPS([[_PATCH, None, False]], amplitudes=[np.zeros(3)]),
     "group 0: amplitudes must have shape (1, 4), got (1, 3)"),
    # ClusterMLP
    ("mlp dict groups", lambda: ClusterMLP({"clusters": _PAIRS}, seed=0),
     "groups must be a list of [clusters, hidden] specs; got dict"),
    ("mlp no groups", lambda: ClusterMLP([], seed=0),
     "groups must contain at least one cluster group"),
    ("mlp unwrapped spec", lambda: ClusterMLP([_PAIRS, [4]], seed=0),
     "groups looks like one [clusters, hidden] spec passed unwrapped"),
    ("mlp one-field spec", lambda: ClusterMLP([[_PAIRS]], seed=0),
     "group 0: a group spec must be a 2-field list [clusters, hidden]; got list of length 1"),
    ("mlp one-dimensional clusters", lambda: ClusterMLP([[np.arange(3), [4]]], seed=0),
     "group 0: clusters must be 2-D (n_clusters, L); got shape (3,)"),
    ("mlp bare hidden width", lambda: ClusterMLP([[_PAIRS, 4]], seed=0),
     "group 0: hidden must be a list of layer widths, not a bare integer; pass [4]"),
    ("mlp N too small", lambda: ClusterMLP([[_PAIRS, [4]]], seed=0, N=2),
     "N=2 is smaller than the largest site index in clusters (3)"),
    ("mlp params count", lambda: ClusterMLP([[_PAIRS, [4]]], params=[]),
     "clusters (1) and params (0) must have the same length"),
    ("mlp params architecture", lambda: ClusterMLP([[_PAIRS, [4]]], params=_bad_mlp_params()),
     "group 0: params describe layer dims [2, 3, 1], but clusters/hidden imply [2, 4, 1]"),
    # Hamiltonians and the exact guide
    ("spin S not half-integer", lambda: Spin.create(chain(_N).edges, S=0.3),
     "S must be a non-negative half-integer; got 0.3"),
    ("bose-hubbard n_max below one", lambda: BoseHubbard.create(chain(_N).edges, t=1.0, n_max=0),
     "n_max must be ≥ 1, got 0"),
    ("exact guide log_amp length", lambda: ExactGroundState(_H, log_amp=np.zeros(3)),
     "log_amp has 3 entries but the basis has 2**4 = 16"),
    # the spin-1/2 guides index by the SpinHalf transition layout
    ("jastrow on a toric code",
     lambda: JastrowSpinHalf(_TORIC.N, [[0, 1]], beta=0.1).log_diff_from_operator(
         jnp.ones(_TORIC.N), _TORIC),
     "JastrowSpinHalf needs an operator with the SpinHalf transition layout"),
    ("jastrow cache for a toric code",
     lambda: JastrowSpinHalf(_TORIC.N, [[0, 1]], beta=0.1).build_cache(_TORIC),
     "JastrowSpinHalf needs an operator with the SpinHalf transition layout"),
    ("cps on a toric code",
     lambda: ClusterCPS([[_PAIRS, None, True]]).log_diff_from_operator(
         jnp.ones(_TORIC.N), _TORIC),
     "ClusterCPS needs an operator with the SpinHalf transition layout"),
    ("cps cache for a toric code", lambda: ClusterCPS([[_PAIRS, None, True]]).build_cache(_TORIC),
     "ClusterCPS needs an operator with the SpinHalf transition layout"),
    ("mlp on a toric code",
     lambda: ClusterMLP([[_PAIRS, [4]]], seed=0, N=_TORIC.N).log_diff_from_operator(
         jnp.ones(_TORIC.N), _TORIC),
     "ClusterMLP needs an operator with the SpinHalf transition layout"),
    ("mlp cache for a toric code",
     lambda: ClusterMLP([[_PAIRS, [4]]], seed=0, N=_TORIC.N).build_cache(_TORIC),
     "ClusterMLP needs an operator with the SpinHalf transition layout"),
    # re-estimation needs the energy series
    ("evaluate_observables without energy",
     lambda: evaluate_observables(dict(log_weights=np.zeros(10), p_array=[2])),
     "raw holds no 'energy' series, so nothing can be re-estimated"),
]

_TYPE_ERRORS = {"on a toric code", "cache for a toric code"}


@pytest.mark.parametrize("label,build,message", CONSTRUCTORS, ids=[c[0] for c in CONSTRUCTORS])
def test_constructor_and_layout_messages(label, build, message):
    """Each malformed constructor argument or wrong operator raises its own message."""
    exc = TypeError if any(label.endswith(s) for s in _TYPE_ERRORS) else ValueError
    with pytest.raises(exc, match=re.escape(message)):
        build()


_HAMILTONIANS = {
    "SpinHalf": _H,
    "Spin": Spin.create(chain(_N).edges, S=1.0, J_pm=-0.5),
    "BoseHubbard": BoseHubbard.create(chain(_N).edges, t=1.0, U=2.0, n_max=2),
    "ToricCode": _TORIC,
    "PlaquetteModel": PlaquetteModel.create([[0, 1, 2, 3]], g=1.0, mu=0.0),
}
_GUIDES = {
    "JastrowSpinHalf": JastrowSpinHalf(_N, chain(_N).edges, beta=0.1),
    "JastrowSpin": JastrowSpin(_N, chain(_N).edges, beta=0.1),
    "JastrowBoseHubbard": JastrowBoseHubbard(_N, chain(_N).edges, beta=0.1),
    "ClusterCPS": ClusterCPS([[_PAIRS, None, True]]),
    "ClusterMLP": ClusterMLP([[_PAIRS, [4]]], seed=0),
    "ExactGroundState": ExactGroundState(_H),
}


def test_built_in_classes_satisfy_the_protocols():
    """Built-in Hamiltonians have every ``pygfmc.Operator`` member and pass the
    engine's check; built-in guides have a ratio routine; the version is the
    one in ``codes/pyproject.toml``; ``kagome(2, 2)`` builds with its hexagons."""
    members = [name for name in ("M_total", "get_rates", "generate_state", "move")
               if hasattr(pygfmc.Operator, name)
               or name in getattr(pygfmc.Operator, "__annotations__", {})]
    assert {"M_total", "get_rates", "generate_state"} <= set(members)
    for name, h in _HAMILTONIANS.items():
        for member in ("M_total", "get_rates", "generate_state"):
            assert hasattr(h, member), (name, member)
        require_operator_protocol(h)
        assert int(h.M_total) > 1, name
    ratio_routines = [name for name in ("log_diff_from_operator", "log_diff_from_configs",
                                        "log_psi") if hasattr(pygfmc.Guide, name)]
    assert len(ratio_routines) == 3
    for name, g in _GUIDES.items():
        assert any(hasattr(g, r) for r in ratio_routines), name
        require_guide_protocol(g)

    pyproject = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "codes", "pyproject.toml")
    with open(pyproject, encoding="utf-8") as f:
        version = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M).group(1)
    assert pygfmc.__version__ == version

    lattice = kagome(2, 2)
    assert lattice.N == 12 and len(lattice.edges) > 0
    assert "hexagons" in lattice.extras and len(lattice.extras["hexagons"]) > 0
