"""Generic spin-S Hamiltonian with arbitrary single-site and edge couplings.

    H = J_z    Σ_<ij> Sz_i Sz_j
      + J_pm   Σ_<ij> (S+_i S-_j + S-_i S+_j)
      + J_pmpm Σ_<ij> (S+_i S+_j + S-_i S-_j)
      - h_z    Σ_i Sz_i
      - h_pm   Σ_i (S+_i + S-_i)          (= -2 h_pm Σ_i Sx_i)

The fields enter with a minus sign, as in SpinHalf and ToricCode; ``create``
stores their negatives (``self.h_z = -h_z``, ``self.h_pm = -h_pm``), so the
rate code below works with ``+ stored value``.

Configurations hold the magnetic quantum numbers ``config[i] ∈ {-S, -S+1, …, S}``,
stored as floats.  ``S`` is the spin per site (integer or half-integer,
e.g. 0.5, 1, 1.5, 2); ``2S`` must be a non-negative integer.

Sign-problem-free convention: ``J_pm ≤ 0``, ``J_pmpm ≤ 0``, ``h_pm ≥ 0``;
``J_z`` and ``h_z`` have no sign restriction.

Off-diagonal layout (each block dropped if its coupling is zero)
────────────────────────────────────────────────────────────────
``rates[0]``                                     diagonal H_{σσ}
followed by these blocks in order:
  J_pm    forward    S+_i S-_j    n_edges slots  (m_i+1, m_j-1)
  J_pm    backward   S-_i S+_j    n_edges slots  (m_i-1, m_j+1)
  J_pmpm  forward    S+_i S+_j    n_edges slots  (m_i+1, m_j+1)
  J_pmpm  backward   S-_i S-_j    n_edges slots  (m_i-1, m_j-1)
  h_pm               S+_i          N      slots  (m_i+1)
  h_pm               S-_i          N      slots  (m_i-1)

Each off-diagonal rate is the coupling times the matrix elements
``⟨m±1|S±|m⟩ = √(S(S+1) - m(m±1))`` of the operators involved.  These
vanish at the extremes m = ±S, so forbidden moves carry zero rate without an
extra mask (their successor configurations lie outside the physical basis
and are never sampled).

Recommended ``lam``:
``lam ≥ |J_z|·n_edges·S² + |h_z|·N·S`` is sufficient under the SPF
convention (since off-diagonal rates ≤ 0 only enlarge b_x).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from ._common import edges_array, two_site_update, two_site_move
import numpy as np


def _raise_factor(m, S):
    """⟨m+1|S+|m⟩ = √(S(S+1) - m(m+1)).  Vanishes at m = S."""
    return jnp.sqrt(jnp.maximum(S * (S + 1) - m * (m + 1), 0.0))


def _lower_factor(m, S):
    """⟨m-1|S-|m⟩ = √(S(S+1) - m(m-1)).  Vanishes at m = -S."""
    return jnp.sqrt(jnp.maximum(S * (S + 1) - m * (m - 1), 0.0))


class Spin(NamedTuple):
    """Spin-S Hamiltonian; see the module docstring for H and the rate layout.

    Build instances with :meth:`create`.
    """

    dtype: object
    S: float                # spin per site (must be half-integer ≥ 0)
    N: int
    edge_i: jnp.ndarray
    edge_j: jnp.ndarray
    # Couplings: shape (1,) for uniform or (n_edges,)/(N,) for non-uniform.
    # None when the term is absent.
    J_z:    object
    J_pm:   object
    J_pmpm: object
    h_z:    object
    h_pm:   object   # one amplitude for BOTH S+ and S- (keeps H Hermitian)
    # Static structure flags.
    n_jpm_off:    int = 0
    n_jpmpm_off:  int = 0
    n_hpm_off:    int = 0   # per direction: n_hpm_off slots for S+, the same for S-
    has_jz:    bool = False
    has_jpm:   bool = False
    has_jpmpm: bool = False
    has_hz:    bool = False
    has_hpm:   bool = False
    uniform_jz:    bool = True
    uniform_jpm:   bool = True
    uniform_jpmpm: bool = True
    uniform_hz:    bool = True
    uniform_hpm:   bool = True

    # ── derived properties ─────────────────────────────────────────────

    @property
    def edges(self):
        """(n_edges, 2) numpy array of site pairs."""
        return np.stack([np.asarray(self.edge_i), np.asarray(self.edge_j)], axis=1)

    @property
    def n_edges(self):
        """Number of edges (static Python int)."""
        return int(self.edge_i.shape[0])

    @property
    def n_off(self):
        """Number of off-diagonal transitions (static Python int)."""
        return (self.n_jpm_off + self.n_jpmpm_off
                + 2 * self.n_hpm_off)

    @property
    def M_total(self):
        """Total number of transitions including the diagonal: n_off + 1."""
        return 1 + self.n_off

    @property
    def d(self):
        """Local Hilbert-space dimension = 2S + 1."""
        return int(round(2 * self.S)) + 1

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def create(cls, edges, S=0.5,
               J_z=0.0, J_pm=0.0, J_pmpm=0.0,
               h_z=None, h_pm=None, N=None, *,
               dtype=jnp.float32, sign_check=True):
        """Build a spin-S Hamiltonian.

        Args:
            edges:     (n_edges, 2) site index pairs.
            S:         spin per site (half-integer ≥ 0); 2S must be a non-negative int.
            J_z:       scalar or (n_edges,) Sz Sz coupling.       Any sign (diagonal).
            J_pm:      scalar or (n_edges,) S+S- + h.c. coupling. ≤ 0 (SPF).
            J_pmpm:    scalar or (n_edges,) S+S+ + h.c. coupling. ≤ 0 (SPF).
            h_z:       scalar, (N,), or None. Any sign (diagonal); enters
                       H as -h_z * sum_i S^z_i.
            h_pm:      scalar, (N,), or None. ≥ 0 (SPF).  ONE amplitude
                       multiplying both S^+ and S^-, i.e. the transverse
                       field -h_pm * sum_i (S^+_i + S^-_i) = -2 h_pm * sum_i
                       S^x_i.  A single shared amplitude keeps H Hermitian.
            N:         number of sites.  Inferred from ``edges`` and any
                       per-site array when None; give it explicitly for an
                       operator without two-body terms (``edges=[]``).
            dtype:     JAX dtype for amplitudes.
            sign_check: enforce SPF conventions if True.
        """
        # validate spin
        twoS = 2 * float(S)
        if abs(round(twoS) - twoS) > 1e-9 or twoS < 0:
            raise ValueError(f"S must be a non-negative half-integer; got {S}")
        S = float(S)
        # the fields enter H with a minus sign; store their negatives
        h_z = None if h_z is None else -np.asarray(h_z) if isinstance(h_z, (list, tuple)) else -h_z
        h_pm = None if h_pm is None else -np.asarray(h_pm) if isinstance(h_pm, (list, tuple)) else -h_pm

        edges_arr, n_edges, N_min = edges_array(edges)

        # Allow site-only Hamiltonians (no edges) by inferring N from a site array.
        for site_arr in (h_z, h_pm):
            if np.ndim(site_arr) > 0:
                N_min = max(N_min, np.shape(site_arr)[0])
        # ``N`` may be given explicitly so that an operator with no two-body
        # terms -- a pure field observable, say -- can be built with
        # ``edges=[]``.  When omitted it is inferred from the edge list and any
        # per-site arrays.
        if N is None:
            N = N_min
        else:
            N = int(N)
            if N < N_min:
                raise ValueError(
                    f"N={N} is smaller than required by the given edges/site "
                    f"arrays (need at least {N_min})")

        def _edge_arr(x, name):
            if np.ndim(x) > 0:
                arr = jnp.asarray(x, dtype=dtype)
                if arr.shape[0] != n_edges:
                    raise ValueError(
                        f"non-uniform {name} must have length n_edges={n_edges}, "
                        f"got {arr.shape[0]}")
                return arr, False
            return jnp.array([float(x)], dtype=dtype), True

        def _site_arr(x, name):
            if x is None:
                return None, True
            if np.ndim(x) > 0:
                arr = jnp.asarray(x, dtype=dtype)
                if arr.shape[0] != N:
                    raise ValueError(
                        f"non-uniform {name} must have length N={N}, "
                        f"got {arr.shape[0]}")
                return arr, False
            return jnp.array([float(x)], dtype=dtype), True

        Jz_a,    uniform_jz    = _edge_arr(J_z,    "J_z")
        Jpm_a,   uniform_jpm   = _edge_arr(J_pm,   "J_pm")
        Jpmpm_a, uniform_jpmpm = _edge_arr(J_pmpm, "J_pmpm")
        hz_a,    uniform_hz    = _site_arr(h_z,    "h_z")
        hpm_a,   uniform_hpm   = _site_arr(h_pm, "h_pm")

        if sign_check:
            if jnp.any(Jpm_a   > 0):
                raise ValueError("J_pm must be ≤ 0 (sign-problem-free convention).")
            if jnp.any(Jpmpm_a > 0):
                raise ValueError("J_pmpm must be ≤ 0 (sign-problem-free convention).")
            if hpm_a is not None and jnp.any(hpm_a > 0):
                raise ValueError("h_pm must be ≥ 0 (sign-problem-free convention).")
            # No Hermiticity check is needed: S^+ and S^- share the single
            # amplitude h_pm by construction, so h_pm (S^+ + S^-) is
            # Hermitian for real h_pm.

        def _is_active(arr):
            if arr is None:
                return False
            return bool(float(jnp.linalg.norm(arr)) > 0)

        has_jz    = _is_active(Jz_a)
        has_jpm   = _is_active(Jpm_a)
        has_jpmpm = _is_active(Jpmpm_a)
        has_hz    = _is_active(hz_a)
        has_hpm   = _is_active(hpm_a)

        n_jpm_off    = 2 * n_edges if has_jpm   else 0
        n_jpmpm_off  = 2 * n_edges if has_jpmpm else 0
        n_hpm_off    = N           if has_hpm   else 0

        # drop unused arrays for compactness
        if not has_hz:
            hz_a = None
        if not has_hpm:
            hpm_a = None

        idx_dtype = jnp.int16 if N <= 32767 else jnp.int32
        edge_i = jnp.array(edges_arr[:, 0] if n_edges else [], dtype=idx_dtype)
        edge_j = jnp.array(edges_arr[:, 1] if n_edges else [], dtype=idx_dtype)

        return cls(
            dtype=dtype, S=S, N=N,
            edge_i=edge_i, edge_j=edge_j,
            J_z=Jz_a, J_pm=Jpm_a, J_pmpm=Jpmpm_a,
            h_z=hz_a, h_pm=hpm_a,
            n_jpm_off=n_jpm_off, n_jpmpm_off=n_jpmpm_off,
            n_hpm_off=n_hpm_off,
            has_jz=has_jz, has_jpm=has_jpm, has_jpmpm=has_jpmpm,
            has_hz=has_hz, has_hpm=has_hpm,
            uniform_jz=uniform_jz, uniform_jpm=uniform_jpm, uniform_jpmpm=uniform_jpmpm,
            uniform_hz=uniform_hz, uniform_hpm=uniform_hpm,
        )

    # ── diagonal weight ────────────────────────────────────────────────

    def diagonal_weight(self, config):
        """Scalar H_{σσ}: J_z Σ Sz_i Sz_j + h_z Σ Sz_i."""
        m   = config.astype(self.dtype)
        out = jnp.array(0.0, dtype=self.dtype)
        if self.has_jz:
            Jz = self.J_z[0] if self.uniform_jz else self.J_z
            mi = m[self.edge_i]
            mj = m[self.edge_j]
            out = out + jnp.sum(Jz * mi * mj)
        if self.has_hz:
            hz = self.h_z[0] if self.uniform_hz else self.h_z
            out = out + jnp.sum(hz * m)
        return out

    # ── transition weights ─────────────────────────────────────────────

    def get_rates(self, config, return_configs=False):
        """Return ``(rates, new_configs|None)``.

        ``rates`` has shape ``(M_total,)``.  See module docstring for layout.
        """
        m  = config.astype(self.dtype)
        S  = self.S
        M  = self.M_total
        rates = jnp.zeros(M, dtype=self.dtype).at[0].set(self.diagonal_weight(config))

        idx = 1
        if self.n_jpm_off > 0:
            mi = m[self.edge_i]
            mj = m[self.edge_j]
            Jpm = self.J_pm[0] if self.uniform_jpm else self.J_pm
            n_e = self.n_edges
            # forward S+_i S-_j: m_i+1, m_j-1
            r_fwd = Jpm * _raise_factor(mi, S) * _lower_factor(mj, S)
            # backward S-_i S+_j: m_i-1, m_j+1
            r_bwd = Jpm * _lower_factor(mi, S) * _raise_factor(mj, S)
            rates = rates.at[idx        : idx +   n_e].set(r_fwd)
            rates = rates.at[idx + n_e  : idx + 2*n_e].set(r_bwd)
            idx += 2 * n_e

        if self.n_jpmpm_off > 0:
            mi = m[self.edge_i]
            mj = m[self.edge_j]
            Jpp = self.J_pmpm[0] if self.uniform_jpmpm else self.J_pmpm
            n_e = self.n_edges
            # forward S+_i S+_j: m_i+1, m_j+1
            r_fwd = Jpp * _raise_factor(mi, S) * _raise_factor(mj, S)
            # backward S-_i S-_j: m_i-1, m_j-1
            r_bwd = Jpp * _lower_factor(mi, S) * _lower_factor(mj, S)
            rates = rates.at[idx        : idx +   n_e].set(r_fwd)
            rates = rates.at[idx + n_e  : idx + 2*n_e].set(r_bwd)
            idx += 2 * n_e

        if self.n_hpm_off > 0:
            # one amplitude, two transition blocks: S^+ then S^-
            hpm = self.h_pm[0] if self.uniform_hpm else self.h_pm
            rates = rates.at[idx : idx + self.N].set(hpm * _raise_factor(m, S))
            idx  += self.N
            rates = rates.at[idx : idx + self.N].set(hpm * _lower_factor(m, S))
            idx  += self.N

        if not return_configs:
            return rates, None

        # Build full new_configs tensor (used only for tests / sparse build).
        new_configs = jnp.zeros((M, self.N), dtype=config.dtype).at[0].set(config)
        idx = 1
        n_e = self.n_edges
        ei, ej = self.edge_i, self.edge_j

        def _edge_block(forward_signs):
            """forward_signs = (di, dj) added to (m_i, m_j)."""
            di, dj = forward_signs
            base = jnp.broadcast_to(config[None, :], (n_e, self.N))
            cfg = (base
                   .at[jnp.arange(n_e), ei].add(di)
                   .at[jnp.arange(n_e), ej].add(dj))
            return cfg

        if self.n_jpm_off > 0:
            new_configs = new_configs.at[idx          : idx +   n_e].set(_edge_block((+1, -1)))
            new_configs = new_configs.at[idx + n_e    : idx + 2*n_e].set(_edge_block((-1, +1)))
            idx += 2 * n_e
        if self.n_jpmpm_off > 0:
            new_configs = new_configs.at[idx          : idx +   n_e].set(_edge_block((+1, +1)))
            new_configs = new_configs.at[idx + n_e    : idx + 2*n_e].set(_edge_block((-1, -1)))
            idx += 2 * n_e
        if self.n_hpm_off > 0:
            base = jnp.broadcast_to(config[None, :], (self.N, self.N))
            diag = jnp.arange(self.N)
            new_configs = new_configs.at[idx : idx + self.N].set(
                base.at[diag, diag].add(+1))
            idx += self.N
            new_configs = new_configs.at[idx : idx + self.N].set(
                base.at[diag, diag].add(-1))
            idx += self.N

        return rates, new_configs

    # ── single-state generation ────────────────────────────────────────

    def generate_state(self, config, idx):
        """Return the configuration after applying transition ``idx``.

        Every transition changes at most two sites, so the target sites and
        their increments are selected as scalars and the configuration is
        updated once, instead of building one full copy per transition block.
        """
        edge_blocks, site_blocks = self._move_blocks()
        return two_site_update(config, idx, self.edge_i, self.edge_j, self.N,
                               edge_blocks, site_blocks)

    def _move_blocks(self):
        """The edge and site blocks of the transition layout (see
        :func:`pygfmc.hamiltonian._common.two_site_move`)."""
        n_jpm = self.n_jpm_off
        n_jpp = self.n_jpmpm_off
        n_h   = self.n_hpm_off      # size of each of the S^+ and S^- blocks
        edge_blocks = []
        if n_jpm > 0:
            edge_blocks.append((0, +1, -1))          # S+_i S-_j / S-_i S+_j
        if n_jpp > 0:
            edge_blocks.append((n_jpm, +1, +1))      # S+_i S+_j / S-_i S-_j
        site_blocks = [(n_jpm + n_jpp + k * n_h, delta)   # S^+ block, then S^- block
                       for k, delta in ((0, +1), (1, -1)) if n_h > 0]
        return edge_blocks, site_blocks

    def move(self, config, idx):
        """The sites changed by transition ``idx`` and their increments,
        ``(sites, deltas)`` of shape ``(2,)`` each (zero increments for the
        diagonal ``idx == 0``); ``generate_state`` is ``config`` with
        ``deltas`` added at ``sites``.  Lets a guide update per-walker state
        after a move (see :class:`pygfmc.Guide`)."""
        edge_blocks, site_blocks = self._move_blocks()
        return two_site_move(idx, self.edge_i, self.edge_j, self.N,
                             edge_blocks, site_blocks, config.dtype)

    @property
    def local_hilbert_space(self):
        """Per-site values, ordered so that position i is basis digit i.

        Matches this class's own basis convention: digit = m + S, so m = -S is digit 0.
        """
        return tuple(float(k) - float(self.S) for k in range(int(self.d)))


# ── pytree registration ─────────────────────────────────────────────────

def _spin_flatten(h):
    leaves = [h.edge_i, h.edge_j, h.J_z, h.J_pm, h.J_pmpm]
    has_hz  = h.h_z  is not None
    has_hpm = h.h_pm is not None
    if has_hz:
        leaves.append(h.h_z)
    if has_hpm:
        leaves.append(h.h_pm)
    aux = (h.dtype, h.S, h.N,
           h.n_jpm_off, h.n_jpmpm_off, h.n_hpm_off,
           h.has_jz, h.has_jpm, h.has_jpmpm, h.has_hz, h.has_hpm,
           h.uniform_jz, h.uniform_jpm, h.uniform_jpmpm,
           h.uniform_hz, h.uniform_hpm,
           has_hz, has_hpm)
    return leaves, aux


def _spin_unflatten(aux, leaves):
    (dtype, S, N,
     n_jpm, n_jpmpm, n_hpm,
     has_jz, has_jpm, has_jpmpm, has_hz, has_hpm,
     uniform_jz, uniform_jpm, uniform_jpmpm,
     uniform_hz, uniform_hpm,
     has_hz_field, has_hpm_field) = aux
    edge_i, edge_j, Jz, Jpm, Jpmpm = leaves[:5]
    idx = 5
    hz = leaves[idx] if has_hz_field else None
    if has_hz_field:
        idx += 1
    hpm = leaves[idx] if has_hpm_field else None
    return Spin(
        dtype=dtype, S=S, N=N, edge_i=edge_i, edge_j=edge_j,
        J_z=Jz, J_pm=Jpm, J_pmpm=Jpmpm,
        h_z=hz, h_pm=hpm,
        n_jpm_off=n_jpm, n_jpmpm_off=n_jpmpm,
        n_hpm_off=n_hpm,
        has_jz=has_jz, has_jpm=has_jpm, has_jpmpm=has_jpmpm,
        has_hz=has_hz, has_hpm=has_hpm,
        uniform_jz=uniform_jz, uniform_jpm=uniform_jpm, uniform_jpmpm=uniform_jpmpm,
        uniform_hz=uniform_hz, uniform_hpm=uniform_hpm,
    )


jax.tree_util.register_pytree_node(Spin, _spin_flatten, _spin_unflatten)
