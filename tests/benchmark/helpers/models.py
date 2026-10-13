"""Model registry shared by the reference generators and the benchmark tests.

Three flat lists:

  - ``QUSPIN_MODELS``     small systems (random graphs for Spin/SpinHalf/
                          BoseHubbard, tori for ToricCode), checked against
                          QuSpin exact diagonalization;
  - ``TENPY_MODELS``      the L=50 two-leg ladder (Spin/SpinHalf/BoseHubbard),
                          checked against TeNPy DMRG;
  - ``PYROCHLORE_MODELS`` the N=128 pyrochlore ring exchange (PlaquetteModel),
                          checked against constrained Lanczos / RK analytics.

Each entry is a dict with

  id            unique string, also the reference-file stem;
  family        Hamiltonian class name;
  phase         phase-point label;
  N             number of sites;
  params        Hamiltonian couplings (stored as ``param_*`` attrs);
  build         zero-argument callable returning ``(h, edges)`` (edges None
                for ToricCode and PlaquetteModel);
  gfmc          run_gfmc settings (lam, step_size, n_steps, p_array,
                n_walkers, method);
  methods       optional list of propagation schemes to run (default: the
                one in ``gfmc``);
  measure_off_diag  whether the forward-walked observables are measured;
  target_charge optional exact conserved-sector start (see below);
  init_states   optional callable returning a stored walker ensemble.

lam is the class's analytic bound on the diagonal element plus ``MARGIN``.
The SpinHalf entries (TFIM and Heisenberg points) also pass ``h_x`` to
``_lam_spinhalf`` as ``extra_margin_field``: the SpinHalf diagonal holds no
``h_x`` term, so their lam carries ``|h_x| N / 2`` of margin on top of
``MARGIN``.  The tolerances and the reference contracts were set with this lam.
"""

from __future__ import annotations

from pygfmc.hamiltonian import BoseHubbard, PlaquetteModel, Spin, SpinHalf, ToricCode

from .graphs import RANDOM_GRAPHS
from .ladder import ladder_edges
from .native_lattices import TORIC_TORI
from .pyrochlore_n128 import N_SITES, load_lattice

MARGIN = 3.0
LADDER_L = 50
N_MAX_BH = 2

# The SpinHalf points run under all three propagation schemes (cont_time is
# the lam -> infinity limit of multi_step), checking that neither the batched
# holds nor the continuous-time limit introduces bias; everything else runs
# multi_step only.  gfmc_bench.step_size_for matches the projection depth.
ALL_METHODS = ["single_step", "multi_step", "cont_time"]


# ── lam bounds ─────────────────────────────────────────────────────────────

def _lam_spin(edges, S, J_z, h_z=0.0, margin=MARGIN):
    N = max(max(e) for e in edges) + 1
    return abs(J_z) * len(edges) * S**2 + abs(h_z) * N * S + margin


def _lam_spinhalf(edges, J_z, extra_margin_field=0.0, margin=MARGIN):
    """``|J_z| n_edges / 4 + |extra_margin_field| N / 2 + margin``.

    The middle term has the form of a longitudinal-field bound; the SpinHalf
    models here have no ``h_z`` and pass ``h_x``, so it is extra margin.
    """
    return _lam_spin(edges, 0.5, J_z, extra_margin_field, margin)


def _lam_bose_hubbard(N, U, mu, n_max=N_MAX_BH, margin=MARGIN):
    return N * max(0.5 * U * n_max * (n_max - 1) - mu * n_max, 0.0) + margin


def _lam_toric(Lx, Ly, B_p, h_z, margin=MARGIN):
    return abs(B_p) * Lx * Ly + abs(h_z) * (2 * Lx * Ly) + margin


def _lam_plaquette(max_flippable, mu, margin=MARGIN):
    """H_diag = mu * n_flippable, bounded by the largest count actually reached
    on the ice manifold rather than by the plaquette count (~4x too high)."""
    return max(mu, 0.0) * max_flippable + margin


# ── phase points ───────────────────────────────────────────────────────────

SPIN_PHASES = [
    ("fm_sign_isotropic",  dict(S=1.0, J_z=1.0, J_pm=-1.0)),
    # J_pmpm changes Sz_tot by +-2 but conserves its parity, and at L=50 the
    # two parity sectors are degenerate far below any tolerance.  h_pm = -0.5
    # (negative for sign-problem freedom) breaks the parity and makes the
    # ground state unique.
    ("field_polarized",    dict(S=1.0, J_z=1.0, J_pm=-1.0, J_pmpm=-0.5, h_z=-4.0,
                                h_pm=-0.5)),
    ("transverse_coverage", dict(S=1.0, J_z=1.0, J_pm=-1.0, h_pm=-2.0)),
]

SPINHALF_PHASES = [
    # h/J = 2 in Pauli units; on the ladder (coordination 3) this is still
    # well inside the ordered phase (DMRG bond correlator 0.221 vs 0.25).
    ("tfim_disordered",   dict(J_x=0.0, J_y=0.0, J_z=-4.0, h_x=4.0)),
    ("field_polarized",   dict(J_x=0.0, J_y=0.0, J_z=-4.0, h_x=8.0)),
    # J_z = +1 against J_x = J_y = -1 is the sublattice-rotated isotropic
    # antiferromagnetic Heisenberg model.
    ("fm_sign_isotropic", dict(J_x=-1.0, J_y=-1.0, J_z=1.0)),
]

BOSE_HUBBARD_PHASES = [
    ("superfluid",     dict(t=1.0, eta=0.0, U=2.0, mu=1.0)),
    ("eta_coverage",   dict(t=0.5, eta=0.5, U=4.0, mu=2.0)),
]

# pure_topological comes first because it is the family's off-diagonal
# representative: <A_s> = 1 and <sum_i X_i> = 0 exactly there.
TORIC_PHASES = [
    ("pure_topological",      dict(A_s=1.0, B_p=1.0, h_z=0.0, h_x=0.0)),
    ("weak_field_topological", dict(A_s=1.0, B_p=1.0, h_z=0.1, h_x=0.1)),
    ("repo_anchored_field",   dict(A_s=1.0, B_p=1.0, h_z=0.5, h_x=0.5)),
    ("deep_polarized",        dict(A_s=1.0, B_p=1.0, h_z=2.0, h_x=2.0)),
]

# g = +1: ring-flip matrix element -1, gauge-equivalent to the external ED's
# +1 (see pyrochlore_n128.py).
PYROCHLORE_PHASES = [
    ("ring_exchange", dict(g=1.0, mu=0.0)),
    ("rk_point",      dict(g=1.0, mu=1.0)),
]

# largest flippable-hexagon count over the stored ice ensemble (mean 23.4)
MAX_FLIPPABLE_N128 = 34

_SMALL = dict(step_size=1, n_steps=8000, p_array=[8, 16, 32, 64],
              n_walkers=2048, method="multi_step")
_LADDER = dict(step_size=1, n_steps=1000, p_array=[8, 16],
               n_walkers=512, method="multi_step")


# ── QuSpin models (random graphs + tori) ───────────────────────────────────

QUSPIN_MODELS = []

for N_spin in (4, 9):
    lattice, seed = RANDOM_GRAPHS[N_spin]
    edges = list(lattice.edges)
    for phase, params in SPIN_PHASES:
        QUSPIN_MODELS.append(dict(
            id=f"spin_graphN{N_spin}_{phase}",
            family="Spin", phase=phase, N=N_spin, graph_seed=seed, params=params,
            build=(lambda edges=edges, params=params: (Spin.create(edges, **params), edges)),
            gfmc=dict(_SMALL, lam=_lam_spin(edges, params["S"], params["J_z"],
                                            params.get("h_z", 0.0))),
        ))

for N_sh in (4, 9, 13):
    lattice, seed = RANDOM_GRAPHS[N_sh]
    edges = list(lattice.edges)
    for phase, params in SPINHALF_PHASES:
        QUSPIN_MODELS.append(dict(
            id=f"spinhalf_graphN{N_sh}_{phase}",
            family="SpinHalf", phase=phase, N=N_sh, graph_seed=seed, params=params,
            build=(lambda edges=edges, params=params: (SpinHalf.create(edges, **params), edges)),
            gfmc=dict(_SMALL, lam=_lam_spinhalf(
                edges, params["J_z"], extra_margin_field=params.get("h_x", 0.0))),
            methods=ALL_METHODS,
        ))

for N_bh in (4, 9):
    lattice, seed = RANDOM_GRAPHS[N_bh]
    edges = list(lattice.edges)
    for phase, params in BOSE_HUBBARD_PHASES:
        full_params = dict(params, n_max=N_MAX_BH)
        QUSPIN_MODELS.append(dict(
            id=f"bosehubbard_graphN{N_bh}_{phase}",
            family="BoseHubbard", phase=phase, N=N_bh, graph_seed=seed, params=full_params,
            build=(lambda edges=edges, p=full_params: (BoseHubbard.create(edges, **p), edges)),
            gfmc=dict(_SMALL, lam=_lam_bose_hubbard(N_bh, params["U"], params["mu"])),
        ))

for (Lx, Ly) in TORIC_TORI:
    N_tc = 2 * Lx * Ly
    for phase, params in TORIC_PHASES:
        QUSPIN_MODELS.append(dict(
            id=f"toriccode_Lx{Lx}Ly{Ly}_{phase}",
            family="ToricCode", phase=phase, N=N_tc, params=dict(params, Lx=Lx, Ly=Ly),
            build=(lambda Lx=Lx, Ly=Ly, params=params: (ToricCode.create(Lx, Ly, **params), None)),
            # <B_p> converges monotonically in p with the residual halving per
            # doubling, and diagonal observables are cheap, so the field
            # points go to p = 128.  At the unperturbed point <B_p> = 1 at
            # every p.
            gfmc=dict(_SMALL, lam=_lam_toric(Lx, Ly, params["B_p"], params["h_z"]),
                      p_array=([8, 16, 32, 64] if phase == "pure_topological"
                               else [8, 16, 32, 64, 96, 128])),
        ))


# ── pyrochlore models (N=128 ring exchange) ────────────────────────────────
#
# ring_exchange (mu=0): external constrained-ED Lanczos, E0 = -24.63446378.
# rk_point (mu=+1): H is a sum of projectors annihilating the equal-amplitude
# superposition over the sector, so E0 = 0 and observables are classical
# sector averages.  The local energy vanishes for EVERY configuration, so the
# walk never branches and any deviation from E = 0 is an estimator bug --
# which makes it the family's off-diagonal representative.

PYROCHLORE_MODELS = []

for phase, params in PYROCHLORE_PHASES:
    PYROCHLORE_MODELS.append(dict(
        id=f"pyrochlore_N128_{phase}",
        family="PlaquetteModel", phase=phase, N=N_SITES, params=params,
        build=(lambda params=params: (
            PlaquetteModel.create(load_lattice()[0], **params), None)),
        init_states=(lambda: load_lattice()[1]),
        gfmc=dict(lam=_lam_plaquette(MAX_FLIPPABLE_N128, params["mu"]),
                  step_size=1,
                  # no branching at the RK point: a few hundred steps suffice
                  n_steps=800 if params["mu"] else 3000,
                  p_array=[8, 16, 32, 64],
                  n_walkers=2048, method="multi_step"),
        measure_off_diag=(phase == "rk_point"),
        reference_unavailable=({"flippable_frac":
            "The constrained-ED export contains energy and ring flips, not flippability."}
            if phase == "ring_exchange" else {}),
    ))


# ── TeNPy models (L=50 two-leg ladder) ─────────────────────────────────────

TENPY_MODELS = []

_LADDER_EDGES, _LADDER_N = ladder_edges(LADDER_L, periodic_legs=False)

for phase, params in SPIN_PHASES:
    TENPY_MODELS.append(dict(
        id=f"spin_ladderL{LADDER_L}_{phase}",
        family="Spin", phase=phase, N=_LADDER_N, params=params,
        build=(lambda edges=_LADDER_EDGES, params=params:
               (Spin.create(edges, **params), edges)),
        gfmc=dict(_LADDER, lam=_lam_spin(_LADDER_EDGES, params["S"], params["J_z"],
                                         params.get("h_z", 0.0))),
    ))

for phase, params in SPINHALF_PHASES:
    TENPY_MODELS.append(dict(
        id=f"spinhalf_ladderL{LADDER_L}_{phase}",
        family="SpinHalf", phase=phase, N=_LADDER_N, params=params,
        build=(lambda edges=_LADDER_EDGES, params=params:
               (SpinHalf.create(edges, **params), edges)),
        gfmc=dict(_LADDER, lam=_lam_spinhalf(
            _LADDER_EDGES, params["J_z"], extra_margin_field=params.get("h_x", 0.0))),
        methods=ALL_METHODS,
    ))

for phase, params in BOSE_HUBBARD_PHASES:
    full_params = dict(params, n_max=N_MAX_BH)
    TENPY_MODELS.append(dict(
        id=f"bosehubbard_ladderL{LADDER_L}_{phase}",
        family="BoseHubbard", phase=phase, N=_LADDER_N, params=full_params,
        build=(lambda edges=_LADDER_EDGES, p=full_params: (BoseHubbard.create(edges, **p), edges)),
        gfmc=dict(_LADDER, lam=_lam_bose_hubbard(_LADDER_N, params["U"], params["mu"])),
    ))


# ── per-model overrides ────────────────────────────────────────────────────

_OVERRIDES = {
    # tau = (p_max / 2) * step_size / lam was the shallowest in the ladder set, so
    # step_size 1 -> 8.  target_charge = 100 starts every walker fully
    # polarized: J_pmpm breaks the U(1) sector, so this is not a pin but it
    # removes a slow ~4 sigma migration from <Sz> = 0 toward z_tot ~ 32.
    "spin_ladderL50_field_polarized": dict(
        target_charge=100, gfmc=dict(p_array=[8, 16, 32, 64, 128], step_size=8)),
    # under-projected at the shared settings, with a seed-to-seed spread
    # larger than the tolerance; p = 128 is excluded because its variance
    # reopens that spread.
    "bosehubbard_ladderL50_eta_coverage": dict(
        gfmc=dict(step_size=2, method="multi_step", n_steps=4000, p_array=[16, 32, 64])),
    # n_tot = 117 is the DMRG ground-state sector (eta = 0 conserves N)
    "bosehubbard_ladderL50_superfluid": dict(
        target_charge=117, gfmc=dict(n_steps=4000, p_array=[16, 32, 64, 128])),
}

for _m in TENPY_MODELS:
    ov = _OVERRIDES.get(_m["id"])
    if ov:
        if "target_charge" in ov:
            _m["target_charge"] = ov["target_charge"]
        _m["gfmc"] = dict(_m["gfmc"], **ov["gfmc"])

_QUSPIN_OVERRIDES = {
    # pm_tot = Sum (S+ + S-) leaves the Sz_tot = 0 sector (gap 1.31 to
    # Sz_tot = +-1), so its forward-walking estimate decays to the exact 0 only
    # as exp(-1.31 tau): at step_size 1 (tau = 3.6 at p = 64) it is still 1.33.
    # step_size 4 reaches tau = (p/2) step_size / lam = 14 at p = 64
    # (pm_tot ~ 0.006) with fewer branching steps; the small Sz_tot = 0
    # sector (19 states) needs far fewer than 2048 walkers.
    "spin_graphN4_fm_sign_isotropic": dict(
        gfmc=dict(step_size=4, n_walkers=512)),
}

for _m in QUSPIN_MODELS:
    ov = _QUSPIN_OVERRIDES.get(_m["id"])
    if ov:
        _m["gfmc"] = dict(_m["gfmc"], **ov["gfmc"])


# One off-diagonal representative per family on the small systems (forward
# walking costs ~p/2 extra steps per step); none on the N=100 ladder.
def _mark_off_diag_representative(models):
    seen = set()
    for m in models:
        m["measure_off_diag"] = m["family"] not in seen
        seen.add(m["family"])


_mark_off_diag_representative(QUSPIN_MODELS)
for _m in TENPY_MODELS:
    _m["measure_off_diag"] = False


# ── conserved-sector pinning ───────────────────────────────────────────────
#
# ``target_charge`` is the exact sum of the stored config values every walker
# starts with: S^z_tot for Spin, 2*S^z_tot for SpinHalf (+-1 configs), the
# particle number for BoseHubbard.  Where that sum is exactly conserved, a
# random per-site start spreads the walkers over ~sqrt(N) sectors, each of
# which relaxes to its own ground state, and the energy comes out biased up
# with nothing in the output to say so.  Each value is the reference's own
# sector; the odd-N SpinHalf graphs have a degenerate Sz = +-1/2 doublet and
# are pinned to the member the reference leans to.  In
# bosehubbard_graphN4_superfluid the iid start also creates frozen walkers
# (n_tot = 0 or 2N admit no hop, so p_d = 1 and the multi_step hold count
# diverges), which the pin removes.
#
# Not pinned: BoseHubbard points with eta != 0 (N not conserved), ToricCode
# (all-ones B_p start), the pyrochlore (stored zero-flux ensemble).
_TARGET_CHARGE = {
    "spin_ladderL50_fm_sign_isotropic":     0,
    "spin_graphN4_fm_sign_isotropic":       0,
    "spin_graphN9_fm_sign_isotropic":       0,
    "spinhalf_ladderL50_fm_sign_isotropic": 0,
    "spinhalf_graphN4_fm_sign_isotropic":   0,
    "spinhalf_graphN9_fm_sign_isotropic":  -1,
    "spinhalf_graphN13_fm_sign_isotropic": -1,
    "bosehubbard_graphN4_superfluid":       5,
    "bosehubbard_graphN9_superfluid":      10,
}

for _m in TENPY_MODELS + QUSPIN_MODELS:
    if _m["id"] in _TARGET_CHARGE:
        assert "target_charge" not in _m, f"{_m['id']} already pins a sector"
        _m["target_charge"] = _TARGET_CHARGE[_m["id"]]
