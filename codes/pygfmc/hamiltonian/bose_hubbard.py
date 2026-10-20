"""Bose-Hubbard Hamiltonian on a finite-occupation Hilbert space.

    H = -t Σ_<ij> (b†_i b_j + b†_j b_i)         # nearest-neighbour hopping
        - η Σ_<ij> (b†_i b†_j + b_i b_j)        # nearest-neighbour pairing
        + (U/2) Σ_i n_i (n_i - 1)               # onsite repulsion
        - μ Σ_i n_i                             # chemical potential
        - h_pm Σ_i (b†_i + b_i)                 # single-site source / sink

Configurations are occupation numbers ``config[i] ∈ {0, …, n_max}``, stored
as floats (non-negative integers, not ±1 spins).

Sign-problem-free convention: ``t ≥ 0``, ``η ≥ 0``, ``h_pm ≥ 0``
so that every off-diagonal matrix element of H is non-positive.  (``η``
carries the same explicit minus-sign convention as ``t``.)

Off-diagonal layout used by :meth:`get_rates`
─────────────────────────────────────────────────────
``rates[0]``                                      diagonal H_{σσ}
``rates[1                       : 1+n_edges]``    forward  hopping  i→j  on each
                                                  edge e=(i,j) with rate
                                                  ``-t_e √(n_i (n_j+1))``
                                                  (target: n_i-1, n_j+1)
``rates[1+n_edges               : 1+2 n_edges]``  backward hopping  j→i  with
                                                  rate ``-t_e √((n_i+1) n_j)``
                                                  (target: n_i+1, n_j-1)
``rates[1+2 n_edges             : 1+3 n_edges]``  pair creation on each edge
                                                  e=(i,j) with rate
                                                  ``-η_e √((n_i+1)(n_j+1))``
                                                  (target: n_i+1, n_j+1)
``rates[1+3 n_edges             : 1+4 n_edges]``  pair annihilation on each
                                                  edge e=(i,j) with rate
                                                  ``-η_e √(n_i n_j)``
                                                  (target: n_i-1, n_j-1)
``rates[1+4 n_edges             : 1+4 n_edges+N]``  creation b†_i with rate
                                                    ``h_pm_i √(n_i+1)``
``rates[1+4 n_edges+N           : 1+4 n_edges+2N]``  annihilation b_i with
                                                     rate ``h_pm_i √(n_i)``

Each off-diagonal block is dropped from the layout (slot count goes to 0)
when its coupling vector is identically zero; the offsets above assume every
block is present.  The two hopping and the two pairing blocks share one flag
each, and a nonzero ``h_pm`` always contributes both the b† and the b block.

Hard cutoff
───────────
``n_max`` truncates the local Hilbert space.  Transitions that would push
n_i beyond ``n_max`` (or below 0) get rate 0 (via a multiplicative mask
or a vanishing √ factor); ``generate_state`` performs the unmasked update,
which is harmless because zero-rate slots are never sampled.

Recommended ``lam``
───────────────────
``lam ≥ N · max_{0 ≤ n ≤ n_max} [U/2 · n(n-1) - μ n]`` is sufficient when
all off-diagonal rates are ≤ 0 (which holds under the SPF convention): the
diagonal is a sum of one per-site term, and the maximum over the occupation
``n`` covers repulsive and attractive ``U`` alike (``n = 0`` gives 0).  The
pairing term is purely off-diagonal and does not affect this bound.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from ._common import edges_array, two_site_update, two_site_move
import numpy as np


class BoseHubbard(NamedTuple):
    """Bose-Hubbard model; see the module docstring for H and the rate layout.

    Build instances with :meth:`create`.
    """

    dtype: object
    edge_i: jnp.ndarray
    edge_j: jnp.ndarray
    N: int
    n_max: int
    # Couplings: each is shape (1,) for uniform or (n_edges,) / (N,) for non-uniform.
    # Stored as None when the term is absent.
    t: object        # (n_edges,) or (1,) hopping amplitude (sign-problem-free: ≥ 0)
    eta: object      # (n_edges,) or (1,) pairing b†b†+bb (sign-problem-free: ≥ 0)
    U: object        # (N,)       or (1,) onsite repulsion
    mu: object       # (N,)       or (1,) chemical potential
    h_pm: object     # (N,) or (1,): ONE amplitude for BOTH b^dagger and b
                     # (sign-problem-free: ≤ 0); None if absent.  Sharing one
                     # amplitude keeps H Hermitian, since (h b^dagger)^dagger
                     # = h b for real h.
    # Static structure flags
    n_hop_off:    int = 0    # 2 * n_edges if has_t   else 0
    n_eta_off:    int = 0    # 2 * n_edges if has_eta else 0
    n_hpm_off:    int = 0    # N if has_h_pm else 0; the b^dagger and b blocks
                             # are separate TRANSITIONS but share h_pm
    has_t:        bool = False
    has_eta:      bool = False
    has_U:        bool = False
    has_mu:       bool = False
    has_h_pm:     bool = False
    uniform_t:    bool = True
    uniform_eta:  bool = True
    uniform_U:    bool = True
    uniform_mu:   bool = True
    uniform_hpm:  bool = True

    # ── basic shape / property helpers ──────────────────────────────────

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
        return self.n_hop_off + self.n_eta_off + 2 * self.n_hpm_off

    @property
    def M_total(self):
        """Total number of transitions including the diagonal: n_off + 1."""
        return 1 + self.n_off

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def create(cls, edges, t=0.0, eta=0.0, U=0.0, mu=0.0,
               h_pm=None, N=None,
               n_max=4, *, dtype=jnp.float32, sign_check=True):
        """Build a Bose-Hubbard Hamiltonian.

        Args:
            edges:      (n_edges, 2) array of (i, j) site index pairs.
            t:          scalar or (n_edges,) hopping amplitude.  Must be ≥ 0
                        for sign-problem-free GFMC (the Hamiltonian carries
                        the explicit minus sign in front of t).
            eta:        scalar or (n_edges,) nearest-neighbour pairing amplitude
                        η, coupling the term -η (b†_i b†_j + b_i b_j) on each
                        edge (same explicit minus-sign convention as ``t``).
                        Must be ≥ 0 for sign-problem-free GFMC.
            U:          scalar or (N,) onsite repulsion.  Any sign (diagonal).
            mu:         scalar or (N,) chemical potential.  Any sign (diagonal).
            h_pm:       scalar or (N,) amplitude multiplying BOTH b^dagger_i
                        and b_i, i.e. -h_pm * sum_i (b^dagger_i + b_i) (minus
                        sign as for the fields of SpinHalf and Spin; stored
                        negated).  ``None`` to disable.  One shared amplitude
                        keeps H Hermitian.  Must be ≥ 0 for sign-problem-free
                        GFMC.
            N:          number of sites.  Inferred from ``edges`` and any
                        per-site array when None; give it explicitly for an
                        operator without two-body terms (``edges=[]``).
            n_max:      hard cutoff on per-site occupation (≥ 1).
            dtype:      JAX dtype for amplitudes.
            sign_check: if True, enforce the sign-problem-free convention.
        """
        # the field enters H with a minus sign; store its negative
        h_pm = None if h_pm is None else -np.asarray(h_pm) if isinstance(h_pm, (list, tuple)) else -h_pm
        edges_arr, n_edges, N_min = edges_array(edges)

        # Need to know N up front to broadcast site-indexed couplings.
        # If edges is empty but the user supplied a per-site coupling, take its size.
        for site_arr in (U, mu, h_pm):
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

        if n_max < 1:
            raise ValueError(f"n_max must be ≥ 1, got {n_max}")

        # ── broadcast couplings ─────────────────────────────────────────
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

        t_arr,   uniform_t   = _edge_arr(t,   "t")
        eta_arr, uniform_eta = _edge_arr(eta, "eta")
        U_arr,  uniform_U  = _site_arr(U,  "U")
        mu_arr, uniform_mu = _site_arr(mu, "mu")
        hpm_arr, uniform_hpm = _site_arr(h_pm, "h_pm")

        # ── sign-problem-free checks ────────────────────────────────────
        if sign_check:
            if jnp.any(t_arr < 0):
                raise ValueError("t must be ≥ 0 (sign-problem-free convention).")
            if jnp.any(eta_arr < 0):
                raise ValueError("eta must be ≥ 0 (sign-problem-free convention).")
            if hpm_arr is not None and jnp.any(hpm_arr > 0):
                raise ValueError("h_pm must be ≥ 0 (sign-problem-free convention).")
            # No Hermiticity check needed: b^dagger and b share the single
            # amplitude h_pm, so h_pm (b^dagger + b) is Hermitian for real
            # h_pm.

        def _is_active(arr):
            if arr is None:
                return False
            return bool(float(jnp.linalg.norm(arr)) > 0)

        has_t       = _is_active(t_arr)
        has_eta     = _is_active(eta_arr)
        has_U       = _is_active(U_arr)
        has_mu      = _is_active(mu_arr)
        has_h_pm  = _is_active(hpm_arr)

        n_hop_off = 2 * n_edges if has_t   else 0
        n_eta_off = 2 * n_edges if has_eta else 0
        n_hpm_off = N           if has_h_pm   else 0

        # Drop unused arrays for compactness
        if not has_h_pm:
            hpm_arr = None
        if not has_U:
            U_arr = jnp.zeros((1,), dtype=dtype)
            uniform_U = True
        if not has_mu:
            mu_arr = jnp.zeros((1,), dtype=dtype)
            uniform_mu = True

        idx_dtype = jnp.int16 if N <= 32767 else jnp.int32
        edge_i = jnp.array(edges_arr[:, 0] if n_edges else [], dtype=idx_dtype)
        edge_j = jnp.array(edges_arr[:, 1] if n_edges else [], dtype=idx_dtype)

        return cls(
            dtype=dtype,
            edge_i=edge_i, edge_j=edge_j,
            N=N, n_max=int(n_max),
            t=t_arr, eta=eta_arr, U=U_arr, mu=mu_arr,
            h_pm=hpm_arr,
            n_hop_off=n_hop_off, n_eta_off=n_eta_off,
            n_hpm_off=n_hpm_off,
            has_t=has_t, has_eta=has_eta, has_U=has_U, has_mu=has_mu,
            has_h_pm=has_h_pm,
            uniform_t=uniform_t, uniform_eta=uniform_eta,
            uniform_U=uniform_U, uniform_mu=uniform_mu,
            uniform_hpm=uniform_hpm,
        )

    # ── diagonal weight ────────────────────────────────────────────────

    def diagonal_weight(self, config):
        """Scalar H_{σσ}: (U/2) Σ n(n-1) − μ Σ n."""
        n   = config.astype(self.dtype)
        out = jnp.array(0.0, dtype=self.dtype)
        if self.has_U:
            U = self.U[0] if self.uniform_U else self.U
            out = out + 0.5 * jnp.sum(U * n * (n - 1.0))
        if self.has_mu:
            mu = self.mu[0] if self.uniform_mu else self.mu
            out = out - jnp.sum(mu * n)
        return out

    # ── transition weights ─────────────────────────────────────────────

    def get_rates(self, config, return_configs=False):
        """Return ``(rates, new_configs|None)``.

        ``rates`` has shape ``(M_total,)`` with ``rates[0]`` the diagonal and
        the off-diagonal layout described in the module docstring.  Rates
        for moves that would exceed n_max (or go negative) are masked to 0.

        ``new_configs`` (when ``return_configs=True``) has shape
        ``(M_total, N)`` with ``new_configs[0] == config``.
        """
        n = config.astype(self.dtype)
        n_max = float(self.n_max)
        M     = self.M_total
        rates = jnp.zeros(M, dtype=self.dtype).at[0].set(self.diagonal_weight(config))

        idx = 1
        # ── hopping ──────────────────────────────────────────────────────
        if self.n_hop_off > 0:
            n_i = n[self.edge_i]
            n_j = n[self.edge_j]
            t   = self.t[0] if self.uniform_t else self.t
            # forward i → j: target n_j' = n_j+1 (must be ≤ n_max); n_i' = n_i-1 (n_i ≥ 1).
            # Note  sqrt(n_i (n_j+1))  is automatically 0 when n_i = 0.
            mask_fwd = (n_j < n_max).astype(self.dtype)
            rate_fwd = -t * jnp.sqrt(n_i * (n_j + 1.0)) * mask_fwd
            # backward j → i: target n_i' = n_i+1, n_j' = n_j-1.
            mask_bwd = (n_i < n_max).astype(self.dtype)
            rate_bwd = -t * jnp.sqrt((n_i + 1.0) * n_j) * mask_bwd
            n_e = self.n_edges
            rates = rates.at[idx          : idx +   n_e].set(rate_fwd)
            rates = rates.at[idx + n_e    : idx + 2*n_e].set(rate_bwd)
            idx  += 2 * n_e

        # ── pairing b†_i b†_j + b_i b_j ─────────────────────────────────
        if self.n_eta_off > 0:
            n_i = n[self.edge_i]
            n_j = n[self.edge_j]
            eta = self.eta[0] if self.uniform_eta else self.eta
            # pair creation: target n_i+1, n_j+1 (both ≤ n_max).
            mask_up = ((n_i < n_max) & (n_j < n_max)).astype(self.dtype)
            rate_up = -eta * jnp.sqrt((n_i + 1.0) * (n_j + 1.0)) * mask_up
            # pair annihilation: target n_i-1, n_j-1.
            # sqrt(n_i * n_j) is automatically 0 when n_i = 0 or n_j = 0.
            rate_down = -eta * jnp.sqrt(n_i * n_j)
            n_e = self.n_edges
            rates = rates.at[idx          : idx +   n_e].set(rate_up)
            rates = rates.at[idx + n_e    : idx + 2*n_e].set(rate_down)
            idx  += 2 * n_e

        # ── creation b†_i ────────────────────────────────────────────────
        if self.n_hpm_off > 0:
            hpm  = self.h_pm[0] if self.uniform_hpm else self.h_pm
            mask = (n < n_max).astype(self.dtype)
            rates = rates.at[idx : idx + self.N].set(
                hpm * jnp.sqrt(n + 1.0) * mask)
            idx  += self.N

            # ── annihilation b_i (same amplitude h_pm) ──────────────────
            # sqrt(n) is 0 when n = 0, so no explicit mask needed
            rates = rates.at[idx : idx + self.N].set(hpm * jnp.sqrt(n))
            idx  += self.N

        if not return_configs:
            return rates, None

        # ── full new_configs tensor (only used for tests / sparse build) ─
        new_configs = jnp.zeros((M, self.N), dtype=config.dtype).at[0].set(config)
        idx = 1
        if self.n_hop_off > 0:
            n_e = self.n_edges
            ei, ej = self.edge_i, self.edge_j
            base = jnp.broadcast_to(config[None, :], (n_e, self.N))
            # forward: -1 at i, +1 at j
            cfg_fwd = (base
                       .at[jnp.arange(n_e), ei].add(-1)
                       .at[jnp.arange(n_e), ej].add(+1))
            # backward: +1 at i, -1 at j
            cfg_bwd = (base
                       .at[jnp.arange(n_e), ei].add(+1)
                       .at[jnp.arange(n_e), ej].add(-1))
            new_configs = new_configs.at[idx          : idx +   n_e].set(cfg_fwd)
            new_configs = new_configs.at[idx + n_e    : idx + 2*n_e].set(cfg_bwd)
            idx += 2 * n_e
        if self.n_eta_off > 0:
            n_e = self.n_edges
            ei, ej = self.edge_i, self.edge_j
            base = jnp.broadcast_to(config[None, :], (n_e, self.N))
            # pair creation: +1 at i, +1 at j
            cfg_up = (base
                      .at[jnp.arange(n_e), ei].add(+1)
                      .at[jnp.arange(n_e), ej].add(+1))
            # pair annihilation: -1 at i, -1 at j
            cfg_down = (base
                        .at[jnp.arange(n_e), ei].add(-1)
                        .at[jnp.arange(n_e), ej].add(-1))
            new_configs = new_configs.at[idx          : idx +   n_e].set(cfg_up)
            new_configs = new_configs.at[idx + n_e    : idx + 2*n_e].set(cfg_down)
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

    # ── single-state generation (hot path) ─────────────────────────────

    def generate_state(self, config, idx):
        """Return the configuration after applying transition ``idx``.

        ``idx == 0`` returns ``config`` unchanged.  Off-diagonal indices
        index into the layout documented at the module level.

        Every transition changes at most two sites, so the target sites and
        their increments are selected as scalars and the configuration is
        updated once (the same scheme as :meth:`Spin.generate_state`),
        instead of building one full candidate configuration per block.
        """
        edge_blocks, site_blocks = self._move_blocks()
        return two_site_update(config, idx, self.edge_i, self.edge_j, self.N,
                               edge_blocks, site_blocks)

    def _move_blocks(self):
        """The edge and site blocks of the transition layout (see
        :func:`pygfmc.hamiltonian._common.two_site_move`)."""
        n_hop = self.n_hop_off
        n_eta = self.n_eta_off
        n_h   = self.n_hpm_off      # size of each of the b^dagger and b blocks
        edge_blocks = []
        if n_hop > 0:
            edge_blocks.append((0, -1, +1))          # hop i -> j (n_i-1, n_j+1), then j -> i
        if n_eta > 0:
            edge_blocks.append((n_hop, +1, +1))      # pair creation, then annihilation
        site_blocks = [(n_hop + n_eta + k * n_h, delta)   # b^dagger block, then b block
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

        Matches this class's own basis convention: the occupation is itself the digit.
        """
        return tuple(float(n) for n in range(int(self.n_max) + 1))


# ── pytree registration ─────────────────────────────────────────────────

def _bh_flatten(h):
    leaves = [h.edge_i, h.edge_j, h.t, h.eta, h.U, h.mu]
    has_hpm = h.h_pm is not None
    if has_hpm:
        leaves.append(h.h_pm)
    aux = (h.dtype, h.N, h.n_max,
           h.n_hop_off, h.n_eta_off, h.n_hpm_off,
           h.has_t, h.has_eta, h.has_U, h.has_mu, h.has_h_pm,
           h.uniform_t, h.uniform_eta, h.uniform_U, h.uniform_mu,
           h.uniform_hpm,
           has_hpm)
    return leaves, aux


def _bh_unflatten(aux, leaves):
    (dtype, N, n_max,
     n_hop, n_eta, n_hpm,
     has_t, has_eta, has_U, has_mu, has_h_pm,
     uniform_t, uniform_eta, uniform_U, uniform_mu, uniform_hpm,
     has_hpm_field) = aux
    edge_i, edge_j, t_arr, eta_arr, U_arr, mu_arr = leaves[:6]
    h_pm = leaves[6] if has_hpm_field else None
    return BoseHubbard(
        dtype=dtype, edge_i=edge_i, edge_j=edge_j,
        N=N, n_max=n_max,
        t=t_arr, eta=eta_arr, U=U_arr, mu=mu_arr, h_pm=h_pm,
        n_hop_off=n_hop, n_eta_off=n_eta, n_hpm_off=n_hpm,
        has_t=has_t, has_eta=has_eta, has_U=has_U, has_mu=has_mu,
        has_h_pm=has_h_pm,
        uniform_t=uniform_t, uniform_eta=uniform_eta,
        uniform_U=uniform_U, uniform_mu=uniform_mu,
        uniform_hpm=uniform_hpm,
    )


jax.tree_util.register_pytree_node(BoseHubbard, _bh_flatten, _bh_unflatten)
