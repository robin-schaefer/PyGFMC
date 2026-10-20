"""QuSpin reference builders: ground energy + observable expectation values.

For every Hamiltonian family an independent QuSpin operator is built from the
PyGFMC model's couplings, diagonalized by sparse Lanczos
(``eigsh(k=1, which="SA")``), and a handful of named observables are
evaluated on that ground state.  Each observable name records the raw
operator measured (e.g. ``pm_tot`` = Sum (S+ + S-) = 2 Sx_tot for Spin), so
``gfmc_bench.py`` can build exactly the same operator on the GFMC side.
"""

from __future__ import annotations

import numpy as np


# ── generic sparse-Krylov helpers (shared across families) ─────────────────

def _qham(static, basis):
    from quspin.operators import hamiltonian as qham
    return qham(static, [], basis=basis, dtype=np.float64,
                check_symm=False, check_herm=False, check_pcon=False)


def _ground_state(static, basis):
    """Sparse Krylov (Lanczos) ground state: (E0, psi0)."""
    H = _qham(static, basis)
    E0, V0 = H.eigsh(k=1, which="SA", return_eigenvectors=True)
    return float(E0[0]), np.asarray(V0[:, 0])


def _expt(static, basis, psi0):
    """Expectation value <psi0| op |psi0> for a (static-list, basis) operator."""
    H = _qham(static, basis)
    val = H.expt_value(psi0)
    val = complex(val)
    if abs(val.imag) > 1e-6 * max(abs(val.real), 1.0):
        raise ValueError(f"unexpectedly large imaginary part in expectation value: {val}")
    return float(val.real)


# ── SpinHalf ─────────────────────────────────────────────────────────────

def _spinhalf_static(h, edges):
    """QuSpin static list equivalent to a SpinHalf Hamiltonian's own terms.

    PyGFMC convention (pygfmc/hamiltonian/SpinHalf.py):
        H = J_pm Sum (S+_i S-_j + S-_i S+_j) + J_pmpm Sum (S+_i S+_j + S-_i S-_j)
          + J_z Sum Sz_i Sz_j + h_z Sum Sz_i + h_x_half Sum (S+_i + S-_i)
    in terms of the STORED fields, which are the negated create() arguments
    (``h.h_z = -h_z``, ``h.h_x_half = -h_x/2``, since H contains
    ``-Sum (h_z Sz_i + h_x Sx_i)``).
    QuSpin spin_basis_general(S="1/2", pauli=False) uses the same physical
    S^a operators, so e.g. "+-" coupling g means g * S+_i S-_j directly.
    """
    from quspin.basis import spin_basis_general

    n_edges = len(edges)
    Jpm   = np.broadcast_to(np.asarray(h.J_pm),   (n_edges,)).astype(float)
    Jpmpm = np.broadcast_to(np.asarray(h.J_pmpm), (n_edges,)).astype(float)
    Jz    = np.broadcast_to(np.asarray(h.J_z),    (n_edges,)).astype(float)

    pm_list, mp_list, pp_list, mm_list, zz_list = [], [], [], [], []
    for (i, j), jpm, jpp, jz in zip(edges, Jpm, Jpmpm, Jz):
        if jpm:
            pm_list.append([float(jpm), int(i), int(j)])
            mp_list.append([float(jpm), int(i), int(j)])
        if jpp:
            pp_list.append([float(jpp), int(i), int(j)])
            mm_list.append([float(jpp), int(i), int(j)])
        if jz:
            zz_list.append([float(jz), int(i), int(j)])

    z_list, p_list, m_list = [], [], []
    if h.h_z is not None and np.any(np.asarray(h.h_z) != 0):
        hz = np.broadcast_to(np.asarray(h.h_z), (h.N,)).astype(float)
        for i, v in enumerate(hz):
            if v:
                z_list.append([float(v), i])
    if h.h_x_half is not None and np.any(np.asarray(h.h_x_half) != 0):
        hxh = np.broadcast_to(np.asarray(h.h_x_half), (h.N,)).astype(float)
        for i, v in enumerate(hxh):
            if v:
                p_list.append([float(v), i])
                m_list.append([float(v), i])

    static = []
    for op_str, lst in (("+-", pm_list), ("-+", mp_list), ("++", pp_list),
                         ("--", mm_list), ("zz", zz_list), ("z", z_list),
                         ("+", p_list), ("-", m_list)):
        if lst:
            static.append([op_str, lst])

    basis = spin_basis_general(h.N, S="1/2", pauli=False)
    return static, basis


def spinhalf_reference(h, edges):
    static, basis = _spinhalf_static(h, edges)
    E0, psi0 = _ground_state(static, basis)
    N = h.N
    i0, j0 = int(edges[0][0]), int(edges[0][1])

    obs = {
        # Sum Sz_i (physical spin-1/2 convention).
        "z_tot": _expt([["z", [[1.0, i] for i in range(N)]]], basis, psi0),
        # <S+_i0 S-_j0 + h.c.> on the first bond, coefficient 1 (physical).
        "pm_bond": _expt([["+-", [[1.0, i0, j0]]], ["-+", [[1.0, i0, j0]]]], basis, psi0),
        # Sum (S+_i + S-_i) = Sum X_i (Pauli convention, matches the h_x=-2 measurement operator).
        "x_tot": _expt([["+", [[1.0, i] for i in range(N)]],
                        ["-", [[1.0, i] for i in range(N)]]], basis, psi0),
    }
    return dict(E0=E0, **obs)


# ── Spin (general S) ────────────────────────────────────────────────────

def _spin_static(h, edges):
    """QuSpin static list equivalent to a Spin (general S) Hamiltonian."""
    from quspin.basis import spin_basis_general

    n_edges = len(edges)
    Jpm   = np.broadcast_to(np.asarray(h.J_pm   if h.has_jpm   else 0.0), (n_edges,)).astype(float)
    Jpmpm = np.broadcast_to(np.asarray(h.J_pmpm if h.has_jpmpm else 0.0), (n_edges,)).astype(float)
    Jz    = np.broadcast_to(np.asarray(h.J_z    if h.has_jz    else 0.0), (n_edges,)).astype(float)

    pm_list, mp_list, pp_list, mm_list, zz_list = [], [], [], [], []
    for (i, j), jpm, jpp, jz in zip(edges, Jpm, Jpmpm, Jz):
        if jpm:
            pm_list.append([float(jpm), int(i), int(j)])
            mp_list.append([float(jpm), int(i), int(j)])
        if jpp:
            pp_list.append([float(jpp), int(i), int(j)])
            mm_list.append([float(jpp), int(i), int(j)])
        if jz:
            zz_list.append([float(jz), int(i), int(j)])

    z_list, p_list, m_list = [], [], []
    if h.has_hz:
        hz = np.broadcast_to(np.asarray(h.h_z), (h.N,)).astype(float)
        for i, v in enumerate(hz):
            if v:
                z_list.append([float(v), i])
    # h_pm multiplies both S+ and S- (the flag is has_hpm on Spin, has_h_pm
    # on BoseHubbard)
    if h.has_hpm:
        hp = np.broadcast_to(np.asarray(h.h_pm), (h.N,)).astype(float)
        for i, v in enumerate(hp):
            if v:
                p_list.append([float(v), i])
    if h.has_hpm:
        hm = np.broadcast_to(np.asarray(h.h_pm), (h.N,)).astype(float)
        for i, v in enumerate(hm):
            if v:
                m_list.append([float(v), i])

    static = []
    for op_str, lst in (("+-", pm_list), ("-+", mp_list), ("++", pp_list),
                         ("--", mm_list), ("zz", zz_list), ("z", z_list),
                         ("+", p_list), ("-", m_list)):
        if lst:
            static.append([op_str, lst])

    if h.S == int(h.S):
        S_str = str(int(h.S))
    else:
        S_str = f"{int(2 * h.S)}/2"
    basis = spin_basis_general(h.N, S=S_str, pauli=False)
    return static, basis


def spin_reference(h, edges):
    static, basis = _spin_static(h, edges)
    E0, psi0 = _ground_state(static, basis)
    N = h.N
    i0, j0 = int(edges[0][0]), int(edges[0][1])

    obs = {
        "z_tot":   _expt([["z", [[1.0, i] for i in range(N)]]], basis, psi0),
        "pm_bond": _expt([["+-", [[1.0, i0, j0]]], ["-+", [[1.0, i0, j0]]]], basis, psi0),
        # Sum (S+_i + S-_i) = 2 * Sum Sx_i (raw ladder sum, see module docstring).
        "pm_tot": _expt([["+", [[1.0, i] for i in range(N)]],
                         ["-", [[1.0, i] for i in range(N)]]], basis, psi0),
    }
    return dict(E0=E0, **obs)


# ── BoseHubbard ──────────────────────────────────────────────────────────

def _bh_static(h, edges):
    """QuSpin static list equivalent to a BoseHubbard Hamiltonian's own terms.

    PyGFMC convention (pygfmc/hamiltonian/bose_hubbard.py):
        H = -t Sum (b+_i b_j + h.c.) - eta Sum (b+_i b+_j + h.c.)
            + (U/2) Sum n_i(n_i-1) - mu Sum n_i + h_pm Sum (b+_i + b_i)
    in terms of the STORED field ``h.h_pm``, the negated create() argument
    (H contains ``-h_pm Sum (b+_i + b_i)``).
    QuSpin "+-" coupling g -> g * b+_i b_j; "n" coupling g -> g * n_i;
    "nn" coupling g -> g * n_i n_j (i==j gives n_i^2, NOT n_i(n_i-1)).
    """
    from quspin.basis import boson_basis_general

    n_edges = len(edges)
    pm_list, mp_list, pp_list, mm_list, n_list, nn_list = [], [], [], [], [], []

    if h.has_t:
        t_arr = np.broadcast_to(np.asarray(h.t), (n_edges,)).astype(float)
        for (i, j), te in zip(edges, t_arr):
            if te:
                pm_list.append([-float(te), int(i), int(j)])
                mp_list.append([-float(te), int(i), int(j)])

    if h.has_eta:
        eta_arr = np.broadcast_to(np.asarray(h.eta), (n_edges,)).astype(float)
        for (i, j), ee in zip(edges, eta_arr):
            if ee:
                pp_list.append([-float(ee), int(i), int(j)])
                mm_list.append([-float(ee), int(i), int(j)])

    if h.has_U:
        U_arr = np.broadcast_to(np.asarray(h.U), (h.N,)).astype(float)
        for i, Ui in enumerate(U_arr):
            if Ui:
                nn_list.append([0.5 * float(Ui), i, i])
                n_list.append([-0.5 * float(Ui), i])

    if h.has_mu:
        mu_arr = np.broadcast_to(np.asarray(h.mu), (h.N,)).astype(float)
        for i, mi in enumerate(mu_arr):
            if mi:
                n_list.append([-float(mi), i])

    p_list, m_list = [], []
    if h.has_h_pm:
        hp = np.broadcast_to(np.asarray(h.h_pm), (h.N,)).astype(float)
        for i, v in enumerate(hp):
            if v:
                p_list.append([float(v), i])
    if h.has_h_pm:
        hm = np.broadcast_to(np.asarray(h.h_pm), (h.N,)).astype(float)
        for i, v in enumerate(hm):
            if v:
                m_list.append([float(v), i])

    static = []
    for op_str, lst in (("+-", pm_list), ("-+", mp_list), ("++", pp_list),
                         ("--", mm_list), ("n", n_list), ("nn", nn_list),
                         ("+", p_list), ("-", m_list)):
        if lst:
            static.append([op_str, lst])

    basis = boson_basis_general(h.N, sps=h.n_max + 1)
    return static, basis


def bose_hubbard_reference(h, edges):
    static, basis = _bh_static(h, edges)
    E0, psi0 = _ground_state(static, basis)
    N = h.N
    i0, j0 = int(edges[0][0]), int(edges[0][1])

    obs = {
        "n_tot":  _expt([["n", [[1.0, i] for i in range(N)]]], basis, psi0),
        # Sum n_i^2 (NOT n(n-1)); combine with n_tot downstream for
        # per-site number variance <n^2>-<n>^2 (the Mott/SF discriminator).
        "n2_tot": _expt([["nn", [[1.0, i, i] for i in range(N)]]], basis, psi0),
        # <b+_i0 b_j0 + h.c.> on one representative bond, coefficient 1.
        "hop_bond": _expt([["+-", [[1.0, i0, j0]]], ["-+", [[1.0, i0, j0]]]], basis, psi0),
        # <b+_i0 b+_j0 + h.c.> on the same bond, coefficient 1 (eta cross-check).
        "pair_bond": _expt([["++", [[1.0, i0, j0]]], ["--", [[1.0, i0, j0]]]], basis, psi0),
    }
    return dict(E0=E0, **obs)


# ── ToricCode ────────────────────────────────────────────────────────────

def _toric_static(h):
    """QuSpin static list equivalent to a ToricCode Hamiltonian's own terms.

    PyGFMC convention (pygfmc/hamiltonian/toric.py):
        H = -A_s Sum_v prod_{e in v} sz_e - B_p Sum_p prod_{e in p} sz_e
            - h_z Sum_e sz_e - h_x Sum_e sx_e
    QuSpin spin_basis_general(pauli=True) uses Pauli matrices, so n-body
    strings like "xxxx"/"zzzz" map directly to products of sigma^x / sigma^z.
    """
    from quspin.basis import spin_basis_general

    # ToricCode's couplings may be uniform scalars or per-star /
    # per-plaquette / per-site arrays, so broadcast before indexing.
    stars = np.asarray(h.stars)
    plaqs = np.asarray(h.plaquettes)
    As = np.broadcast_to(np.asarray(h.A_s), (stars.shape[0],)).astype(float)
    Bp = np.broadcast_to(np.asarray(h.B_p), (plaqs.shape[0],)).astype(float)
    hz = np.broadcast_to(np.asarray(h.h_z), (h.N,)).astype(float)
    hx = np.broadcast_to(np.asarray(h.h_x), (h.N,)).astype(float)

    static = []
    if h.has_As:
        static.append(["xxxx", [[-float(c), *map(int, sites)]
                                 for sites, c in zip(stars, As)]])
    if h.has_Bp:
        static.append(["zzzz", [[-float(c), *map(int, sites)]
                                 for sites, c in zip(plaqs, Bp)]])
    if h.has_hz:
        static.append(["z", [[-float(hz[i]), i] for i in range(h.N)]])
    if h.has_hx:
        static.append(["x", [[-float(hx[i]), i] for i in range(h.N)]])

    basis = spin_basis_general(h.N, S="1/2", pauli=True)
    return static, basis


def toric_reference(h):
    static, basis = _toric_static(h)
    E0, psi0 = _ground_state(static, basis)
    n_plaq  = int(np.asarray(h.plaquettes).shape[0])
    n_stars = int(np.asarray(h.stars).shape[0])

    obs = {
        # <B_p> averaged over all plaquettes (sign-flipped from H's own -B_p).
        "Bp_avg": _expt([["zzzz", [[1.0 / n_plaq, *map(int, sites)]
                                    for sites in np.asarray(h.plaquettes)]]], basis, psi0),
        # <A_s> averaged over all stars (sign-flipped, sign_check bypass needed
        # on the GFMC side since A_s>=0 is otherwise enforced).
        "As_avg": _expt([["xxxx", [[1.0 / n_stars, *map(int, sites)]
                                    for sites in np.asarray(h.stars)]]], basis, psi0),
        # Sum X_i (Pauli convention, sign-flipped from H's own -h_x).
        "x_tot": _expt([["x", [[1.0, i] for i in range(h.N)]]], basis, psi0),
    }
    return dict(E0=E0, **obs)


# ── family dispatch ──────────────────────────────────────────────────────

def compute_reference(family, h, edges=None):
    """Dispatch to the right per-family reference builder.

    Returns a dict with at least ``E0`` plus named observables; the caller
    (``generate_quspin_reference.py``) is responsible for HDF5 serialization.
    """
    if family == "SpinHalf":
        return spinhalf_reference(h, edges)
    if family == "Spin":
        return spin_reference(h, edges)
    if family == "BoseHubbard":
        return bose_hubbard_reference(h, edges)
    if family == "ToricCode":
        return toric_reference(h)
    raise ValueError(f"unknown family {family!r}")
