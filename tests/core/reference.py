"""Independent reference constructions for the Hamiltonian tests.

Each model's matrix is built here from Kronecker products of single-site
operators (or, for PlaquetteModel, an explicit loop over plaquettes), sharing
no code with the ``get_rates`` path the tests check.  These were the models'
own ``build_sparse_matrix`` / ``config_to_basis_index`` methods before they
were removed from the library.

Basis conventions: site 0 is the most significant digit, with digit
``(1 - sigma)/2`` (SpinHalf, ToricCode), ``m + S`` (Spin) or ``n``
(BoseHubbard).  PlaquetteModel is LSB-first with ``+1 -> bit set``.
"""
import numpy as np
from scipy import sparse


# ── matrices ────────────────────────────────────────────────────────────────

def _spinhalf_matrix(h):
    sz = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]), dtype=float) / 2
    sp = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]), dtype=float)
    sm = sparse.csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))

    L = h.N
    Z, Sp, Sm = [], [], []
    for i in range(L):
        li, ri = 2 ** i, 2 ** (L - i - 1)
        Z.append(sparse.kron(sparse.kron(sparse.eye(li), sz), sparse.eye(ri), "csr"))
        Sp.append(sparse.kron(sparse.kron(sparse.eye(li), sp), sparse.eye(ri), "csr"))
        Sm.append(sparse.kron(sparse.kron(sparse.eye(li), sm), sparse.eye(ri), "csr"))

    H = sparse.csr_matrix((2 ** L, 2 ** L), dtype=np.double)
    n_edges = len(h.edges)
    J_pm = np.broadcast_to(np.asarray(h.J_pm), (n_edges,))
    J_pmpm = np.broadcast_to(np.asarray(h.J_pmpm), (n_edges,))
    J_z = np.broadcast_to(np.asarray(h.J_z), (n_edges,))
    for (e0, e1), jpm, jpmpm, jz in zip(h.edges, J_pm, J_pmpm, J_z):
        e0, e1 = int(e0), int(e1)
        H += float(jpm) * (Sp[e0] * Sm[e1] + Sm[e0] * Sp[e1])
        H += float(jpmpm) * (Sp[e0] * Sp[e1] + Sm[e0] * Sm[e1])
        H += float(jz) * Z[e0] * Z[e1]

    # SpinHalf STORES its fields as the coefficients of +Sum S^z and
    # +Sum (S^+ + S^-), i.e. h.h_z = -h_z and h.h_x_half = -h_x/2 of the
    # create() arguments (H contains -Sum (h_z S^z + h_x S^x)); the public
    # sign is checked separately in test_sign_problem_free.py.  Spin and
    # BoseHubbard store their fields negated in the same way.
    if h.h_z is not None and np.linalg.norm(np.asarray(h.h_z)) > 0:
        for i, hz in enumerate(np.broadcast_to(np.asarray(h.h_z), (L,))):
            H += float(hz) * Z[i]
    if h.n_hx > 0:
        for i, hx in enumerate(np.broadcast_to(np.asarray(h.h_x_half), (L,))):
            H += float(hx) * (Sp[i] + Sm[i])
    return H


def _site_op(local, k, d, N):
    return sparse.kron(sparse.kron(sparse.eye(d ** k), local),
                       sparse.eye(d ** (N - k - 1)), "csr")


def _spin_matrix(h):
    d, S, N = int(h.d), h.S, h.N
    m_vals = np.arange(d, dtype=float) - S
    sz_local = sparse.diags(m_vals)
    # <m+1|S+|m> = sqrt(S(S+1) - m(m+1)) at row m+1, col m
    sp_local = sparse.diags(np.sqrt(S * (S + 1) - m_vals[:-1] * (m_vals[:-1] + 1)),
                            offsets=-1)
    sm_local = sp_local.T.tocsr()
    Sz = [_site_op(sz_local, k, d, N) for k in range(N)]
    Sp = [_site_op(sp_local, k, d, N) for k in range(N)]
    Sm = [_site_op(sm_local, k, d, N) for k in range(N)]

    H = sparse.csr_matrix((d ** N, d ** N), dtype=float)
    if h.has_jz or h.has_jpm or h.has_jpmpm:
        Jz = np.broadcast_to(np.asarray(h.J_z), (h.n_edges,)).astype(float)
        Jpm = np.broadcast_to(np.asarray(h.J_pm), (h.n_edges,)).astype(float)
        Jpp = np.broadcast_to(np.asarray(h.J_pmpm), (h.n_edges,)).astype(float)
        for (i, j), jz, jpm, jpp in zip(h.edges, Jz, Jpm, Jpp):
            i, j = int(i), int(j)
            if jz:
                H = H + float(jz) * (Sz[i] @ Sz[j])
            if jpm:
                H = H + float(jpm) * (Sp[i] @ Sm[j] + Sm[i] @ Sp[j])
            if jpp:
                H = H + float(jpp) * (Sp[i] @ Sp[j] + Sm[i] @ Sm[j])
    if h.has_hz:
        for k, v in enumerate(np.broadcast_to(np.asarray(h.h_z), (N,)).astype(float)):
            if v:
                H = H + float(v) * Sz[k]
    if h.has_hpm:
        for k, v in enumerate(np.broadcast_to(np.asarray(h.h_pm), (N,)).astype(float)):
            if v:
                H = H + float(v) * (Sp[k] + Sm[k])
    return H


def _bose_hubbard_matrix(h):
    d, N = h.n_max + 1, h.N
    n_local = sparse.diags(np.arange(d, dtype=float))
    b_local = sparse.diags(np.sqrt(np.arange(1, d, dtype=float)), offsets=1)
    bdag_local = b_local.T.tocsr()
    N_op = [_site_op(n_local, k, d, N) for k in range(N)]
    B = [_site_op(b_local, k, d, N) for k in range(N)]
    Bdag = [_site_op(bdag_local, k, d, N) for k in range(N)]

    H = sparse.csr_matrix((d ** N, d ** N), dtype=float)
    if h.has_t:
        t_arr = np.broadcast_to(np.asarray(h.t), (h.n_edges,)).astype(float)
        for (i, j), te in zip(h.edges, t_arr):
            i, j = int(i), int(j)
            H = H - float(te) * (Bdag[i] @ B[j] + Bdag[j] @ B[i])
    if h.has_eta:
        eta_arr = np.broadcast_to(np.asarray(h.eta), (h.n_edges,)).astype(float)
        for (i, j), ee in zip(h.edges, eta_arr):
            i, j = int(i), int(j)
            if ee:
                H = H - float(ee) * (Bdag[i] @ Bdag[j] + B[i] @ B[j])
    if h.has_U:
        for i, Ui in enumerate(np.broadcast_to(np.asarray(h.U), (N,)).astype(float)):
            if Ui:
                H = H + 0.5 * float(Ui) * (N_op[i] @ (N_op[i] - sparse.eye(d ** N)))
    if h.has_mu:
        for i, mi in enumerate(np.broadcast_to(np.asarray(h.mu), (N,)).astype(float)):
            if mi:
                H = H - float(mi) * N_op[i]
    if h.has_h_pm:
        for i, hi in enumerate(np.broadcast_to(np.asarray(h.h_pm), (N,)).astype(float)):
            if hi:
                H = H + float(hi) * (Bdag[i] + B[i])
    return H


def _plaquette_matrix(h):
    L = h.N
    dim = 2 ** L
    plaq = np.asarray(h.plaquettes)
    g = np.broadcast_to(np.asarray(h.g), (h.n_plaquette,))
    mu = np.broadcast_to(np.asarray(h.mu), (h.n_plaquette,))
    signs = np.asarray(h.signs)

    data, rows, cols = [], [], []
    for x in range(dim):
        config = np.array([1.0 if ((x >> i) & 1) else -1.0 for i in range(L)])
        for k in range(h.n_plaquette):
            if abs(float(np.sum(signs * config[plaq[k]]))) > h.plaquette_size - 0.5:
                if h.has_mu:
                    data.append(float(mu[k]))
                    rows.append(x)
                    cols.append(x)
                if h.has_g:
                    x_flip = x
                    for site in plaq[k]:
                        x_flip ^= (1 << int(site))
                    data.append(-float(g[k]))
                    rows.append(x_flip)
                    cols.append(x)
    return sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim),
                             dtype=float).tocsr()


def _toric_matrix(h):
    N, dim = h.N, 2 ** h.N
    sz = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
    sx = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))

    def kron_site(mat, i):
        return sparse.kron(sparse.kron(sparse.eye(2 ** i, format="csr"), mat,
                                       format="csr"),
                           sparse.eye(2 ** (N - i - 1), format="csr"), format="csr")

    H = sparse.csr_matrix((dim, dim), dtype=float)
    Bp = np.broadcast_to(np.asarray(h.B_p), (h.n_plaq,)).astype(float)
    As = np.broadcast_to(np.asarray(h.A_s), (h.n_stars,)).astype(float)
    hz = np.broadcast_to(np.asarray(h.h_z), (N,)).astype(float)
    hx = np.broadcast_to(np.asarray(h.h_x), (N,)).astype(float)
    if h.has_Bp:
        for p_sites, c in zip(np.asarray(h.plaquettes), Bp):
            if c:
                term = sparse.eye(dim, format="csr")
                for site in p_sites:
                    term = term @ kron_site(sz, int(site))
                H = H - float(c) * term
    if h.has_hz:
        for i in range(N):
            if hz[i]:
                H = H - float(hz[i]) * kron_site(sz, i)
    if h.has_As:
        for s_sites, c in zip(np.asarray(h.stars), As):
            if c:
                term = sparse.eye(dim, format="csr")
                for site in s_sites:
                    term = term @ kron_site(sx, int(site))
                H = H - float(c) * term
    if h.has_hx:
        for i in range(N):
            if hx[i]:
                H = H - float(hx[i]) * kron_site(sx, i)
    return H


_MATRIX = {"SpinHalf": _spinhalf_matrix, "Spin": _spin_matrix,
           "BoseHubbard": _bose_hubbard_matrix, "PlaquetteModel": _plaquette_matrix,
           "ToricCode": _toric_matrix}


def reference_matrix(h):
    """The full sparse Hamiltonian of ``h`` from Kronecker products."""
    return _MATRIX[type(h).__name__](h)


def reference_dense(h):
    """``reference_matrix(h)`` as a dense float64 array."""
    return np.asarray(reference_matrix(h).todense(), dtype=np.float64)


def reference_ground_state(h):
    """(E0, psi0) of the dense reference matrix, psi0 in the reference basis."""
    e, v = np.linalg.eigh(reference_dense(h))
    return float(e[0]), v[:, 0]


# ── basis index ─────────────────────────────────────────────────────────────

def reference_basis_index(h, config):
    """Index of ``config`` (shape ``(..., N)``) in ``reference_matrix(h)``."""
    c = np.asarray(config, dtype=np.float64)
    N = c.shape[-1]
    name = type(h).__name__
    if name == "PlaquetteModel":                       # LSB-first, +1 -> bit set
        return (c > 0).astype(np.int64) @ (1 << np.arange(N, dtype=np.int64))
    if name in ("SpinHalf", "ToricCode"):
        d, digits = 2, (1 - c) / 2
    elif name == "Spin":
        d, digits = int(h.d), c + h.S
    elif name == "BoseHubbard":
        d, digits = h.n_max + 1, c
    else:
        raise TypeError(f"no reference basis for {name}")
    powers = d ** np.arange(N - 1, -1, -1, dtype=np.int64)
    return np.rint(digits).astype(np.int64) @ powers


# ── geometry ────────────────────────────────────────────────────────────────

def pyrochlore_cubic_geometry(Lx, Ly, Lz):
    """Positions and supercell (rows) of the 16-site-cell cubic pyrochlore."""
    sub_cell_origins = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                                 [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    tetra_vertices = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.0],
                               [0.25, 0.0, 0.25], [0.0, 0.25, 0.25]])
    offsets = (sub_cell_origins[:, None, :] + tetra_vertices[None]).reshape(16, 3)
    positions = np.zeros((16 * Lx * Ly * Lz, 3))
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                ci = z + y * Lz + x * Ly * Lz
                positions[ci * 16:(ci + 1) * 16] = np.array([x, y, z], float) + offsets
    return positions, np.diag([float(Lx), float(Ly), float(Lz)])
