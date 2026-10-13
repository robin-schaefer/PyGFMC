"""Kitaev's toric code in longitudinal and transverse fields on a torus.

    H = −A_s Σ_v Π_{i∈star(v)} σ^x_i − B_p Σ_p Π_{i∈plaq(p)} σ^z_i
        − h_z Σ_i σ^z_i − h_x Σ_i σ^x_i

Sites are the 2·Lx·Ly edges of the square lattice; configurations are ±1
floats (σ^z eigenvalues).  See :class:`ToricCode` for the site indexing, the
rate layout and the sign convention.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ToricCode(NamedTuple):
    """Toric code Hamiltonian on an Lx × Ly torus.

        H = −A_s Σ_v ∏_{i∈star(v)} σ^x_i
            − B_p Σ_p ∏_{i∈plaq(p)} σ^z_i
            − h_z Σ_i σ^z_i
            − h_x Σ_i σ^x_i

    Sites are the 2·Lx·Ly *edges* of the underlying square lattice:
        horizontal edge h(x,y): index x + Lx·y          (x∈[0,Lx), y∈[0,Ly))
        vertical   edge v(x,y): index Lx·Ly + x + Lx·y  (x∈[0,Lx), y∈[0,Ly))

    Spin values: config[i] ∈ {+1, −1} = σ^z eigenvalue.

    Star (vertex) at (x,y) acts on edges: h(x,y), h(x−1,y), v(x,y), v(x,y−1).
    Plaquette (face) at (x,y) acts on edges: h(x,y), h(x,y+1), v(x,y), v(x+1,y).

    Sign-problem-free convention: A_s ≥ 0, h_x ≥ 0 (off-diagonal rates −A_s, −h_x ≤ 0).
    No restriction on B_p or h_z (both are diagonal).

    Choosing lam: it must exceed the largest diagonal element H_{σσ}
    visited.  ``lam = |B_p|·Lx·Ly + |h_z|·N``, the worst case, is always safe.
    Equilibrated walkers sit near the bottom of the spectrum, so a smaller
    value often suffices; check the diagonal statistics.

    Couplings may be scalars or per-object arrays (see :meth:`create`).

    Layout of rates returned by get_rates:
        rates[0]                        : diagonal H_{σσ}
        rates[1 : 1+n_stars_off]        : star moves (−A_s each; active for all σ)
        rates[1+n_stars_off : M_total]  : single-site X moves (−h_x each)
    """

    dtype: object
    stars: jnp.ndarray       # (n_stars, 4) site indices per star vertex
    plaquettes: jnp.ndarray  # (n_plaq,  4) site indices per plaquette face
    N: int                   # 2 * Lx * Ly
    Lx: int
    Ly: int
    # Couplings: shape (1,) uniform or per-object arrays --
    #   A_s (n_stars,), B_p (n_plaq,), h_z (N,), h_x (N,).
    A_s: object              # star coupling, per star   (≥ 0 sign-problem-free)
    B_p: object              # plaquette coupling, per plaquette (any sign, diagonal)
    h_z: object              # z-field, per site (any sign, diagonal)
    h_x: object              # x-field, per site (≥ 0 sign-problem-free)
    n_stars: int             # Lx * Ly
    n_plaq: int              # Lx * Ly
    n_stars_off: int         # n_stars if A_s != 0 else 0
    n_hx_off: int            # N if h_x != 0 else 0
    has_As: bool
    has_Bp: bool
    has_hz: bool
    has_hx: bool
    uniform_As: bool = True
    uniform_Bp: bool = True
    uniform_hz: bool = True
    uniform_hx: bool = True

    @property
    def n_off(self):
        """Number of off-diagonal transitions (static Python int).

        Required for a ToricCode instance to be usable as an
        ``off_diag_operator`` in ``run_gfmc`` (which gates on ``op.n_off > 0``);
        the star flips and the single-site X moves are the off-diagonal ones.
        """
        return self.n_stars_off + self.n_hx_off

    @property
    def M_total(self):
        """Total transitions including diagonal (static Python int)."""
        return 1 + self.n_stars_off + self.n_hx_off

    @classmethod
    def create(cls, Lx, Ly, A_s=1.0, B_p=1.0, h_z=0.0, h_x=0.0, *,
               dtype=jnp.float32, sign_check=True):
        """Build the toric code Hamiltonian on an Lx × Ly torus.

        Args:
            Lx, Ly:      lattice dimensions
            A_s:         star (vertex) coefficient; scalar or (n_stars,) =
                         (Lx*Ly,) per star.  ≥ 0 for sign-problem-free GFMC.
            B_p:         plaquette coefficient; scalar or (n_plaq,) = (Lx*Ly,)
                         per plaquette (diagonal, any sign).
            h_z:         σ^z field; scalar or (N,) per site (diagonal, any sign).
            h_x:         σ^x field; scalar or (N,) per site.  ≥ 0 for
                         sign-problem-free GFMC.
            dtype:       JAX dtype (default float32)
            sign_check:  raise ValueError if any A_s < 0 or h_x < 0 (default True)

        Every coupling may be uniform (a scalar) or spatially varying (an
        array of the matching length), so disorder and boundary-modified
        stabilizers can be expressed without subclassing.
        """
        N   = 2 * Lx * Ly
        n_v = Lx * Ly  # number of vertices = stars = faces = plaquettes

        def _arr(v, n, name):
            """(array of shape (1,) or (n,), uniform_flag) for one coupling."""
            a = np.atleast_1d(np.asarray(v, dtype=float))
            if a.size == 1:
                return jnp.array(a.reshape(1), dtype=dtype), True
            if a.shape != (n,):
                raise ValueError(
                    f"{name} must be a scalar or have shape ({n},), got {a.shape}.")
            return jnp.array(a, dtype=dtype), False

        A_s_a, uniform_As = _arr(A_s, n_v, "A_s")
        B_p_a, uniform_Bp = _arr(B_p, n_v, "B_p")
        h_z_a, uniform_hz = _arr(h_z, N,   "h_z")
        h_x_a, uniform_hx = _arr(h_x, N,   "h_x")

        if sign_check:
            if np.any(np.asarray(A_s_a) < 0):
                raise ValueError(
                    "A_s must be ≥ 0 everywhere (sign-problem-free convention); "
                    f"most negative entry {float(np.min(np.asarray(A_s_a)))}.")
            if np.any(np.asarray(h_x_a) < 0):
                raise ValueError(
                    "h_x must be ≥ 0 everywhere (sign-problem-free convention); "
                    f"most negative entry {float(np.min(np.asarray(h_x_a)))}.")

        # ── build star and plaquette index arrays ─────────────────────────────
        stars_list = []
        plaq_list  = []
        for y in range(Ly):
            for x in range(Lx):
                # Star at vertex (x, y): 4 adjacent edges
                h_right = x          + Lx * y
                h_left  = (x - 1) % Lx + Lx * y
                v_up    = Lx * Ly + x          + Lx * y
                v_down  = Lx * Ly + x          + Lx * ((y - 1) % Ly)
                stars_list.append([h_right, h_left, v_up, v_down])

                # Plaquette at face (x, y): 4 bounding edges
                p_bottom = x          + Lx * y
                p_top    = x          + Lx * ((y + 1) % Ly)
                p_left   = Lx * Ly + x          + Lx * y
                p_right  = Lx * Ly + (x + 1) % Lx + Lx * y
                plaq_list.append([p_bottom, p_top, p_left, p_right])

        idx_dtype = jnp.int16 if N <= 32767 else jnp.int32
        stars_arr = jnp.array(stars_list, dtype=idx_dtype)
        plaq_arr  = jnp.array(plaq_list,  dtype=idx_dtype)

        def _active(a):
            return bool(float(np.linalg.norm(np.asarray(a))) > 0)

        has_As = _active(A_s_a)
        has_Bp = _active(B_p_a)
        has_hz = _active(h_z_a)
        has_hx = _active(h_x_a)

        return cls(
            dtype=dtype,
            stars=stars_arr,
            plaquettes=plaq_arr,
            N=N, Lx=Lx, Ly=Ly,
            A_s=A_s_a, B_p=B_p_a, h_z=h_z_a, h_x=h_x_a,
            n_stars=n_v, n_plaq=n_v,
            n_stars_off=n_v if has_As else 0,
            n_hx_off=N if has_hx else 0,
            has_As=has_As, has_Bp=has_Bp, has_hz=has_hz, has_hx=has_hx,
            uniform_As=uniform_As, uniform_Bp=uniform_Bp,
            uniform_hz=uniform_hz, uniform_hx=uniform_hx,
        )

    def diagonal_weight(self, config):
        """Scalar H_{σσ}: −B_p Σ_p ∏_{i∈p} σ^z_i − h_z Σ_i σ^z_i."""
        diag = jnp.array(0.0, dtype=self.dtype)
        if self.has_Bp:
            plaq = jnp.prod(config[self.plaquettes].astype(self.dtype), axis=-1)
            Bp = self.B_p[0] if self.uniform_Bp else self.B_p
            diag = diag - jnp.sum(Bp * plaq)
        if self.has_hz:
            hz = self.h_z[0] if self.uniform_hz else self.h_z
            diag = diag - jnp.sum(hz * config.astype(self.dtype))
        return diag

    def get_rates(self, config, return_configs=False):
        """Rates (M_total,) and optionally the (M_total, N) connected-configs tensor.

        Off-diagonal rates are (−A_s, −h_x) independent of config, because
        σ^x is always active (no flippability constraint).  With per-star /
        per-site couplings they are config-independent but differ from slot
        to slot.
        """
        M = self.M_total
        rates = jnp.zeros(M, dtype=self.dtype).at[0].set(self.diagonal_weight(config))

        if self.has_As:
            As = (jnp.full(self.n_stars_off, self.A_s[0], dtype=self.dtype)
                  if self.uniform_As else self.A_s)
            rates = rates.at[1 : 1 + self.n_stars_off].set(-As)
        if self.has_hx:
            hx = (jnp.full(self.n_hx_off, self.h_x[0], dtype=self.dtype)
                  if self.uniform_hx else self.h_x)
            rates = rates.at[1 + self.n_stars_off :].set(-hx)

        if not return_configs:
            return rates, None

        new_configs = jnp.zeros((M, self.N), dtype=config.dtype).at[0].set(config)

        if self.has_As:
            def _flip_star(cfg, sites):
                return cfg.at[sites].set(-cfg[sites])
            flipped = jax.vmap(_flip_star, in_axes=(None, 0))(config, self.stars)
            new_configs = new_configs.at[1 : 1 + self.n_stars].set(flipped)

        if self.has_hx:
            flip_masks = jnp.eye(self.N, dtype=bool)
            new_configs = new_configs.at[1 + self.n_stars_off :].set(
                jnp.where(flip_masks, -config, config))

        return rates, new_configs

    def generate_state(self, config, idx):
        """Apply off-diagonal transition idx to config.

        idx == 0                  : return config unchanged (diagonal)
        1 <= idx <= n_stars_off   : flip the 4 sites of star (idx−1)
        n_stars_off < idx         : flip site (idx − 1 − n_stars_off)
        """
        off_idx = idx - 1
        n_s = self.n_stars_off

        if self.has_As and self.has_hx:
            is_star   = off_idx < n_s
            star_idx  = jnp.clip(off_idx, 0, self.n_stars - 1)
            star_sites = self.stars[star_idx]
            star_flip = config.at[star_sites].set(-config[star_sites])
            site_idx  = jnp.clip(off_idx - n_s, 0, self.N - 1)
            site_flip = config.at[site_idx].set(-config[site_idx])
            flipped   = jnp.where(is_star, star_flip, site_flip)
        elif self.has_As:
            star_sites = self.stars[off_idx]
            flipped    = config.at[star_sites].set(-config[star_sites])
        else:
            # only h_x site flips; off_idx is the site index
            flipped = config.at[off_idx].set(-config[off_idx])

        return jnp.where(idx == 0, config, flipped)

    @property
    def local_hilbert_space(self):
        """Per-site values, ordered so that position i is basis digit i.

        Matches this class's own basis convention: digit = (1 - sigma)/2, so sigma = +1 is digit 0.
        """
        return (1.0, -1.0)




def _toric_flatten(h):
    # A_s/B_p/h_z/h_x are LEAVES, not aux: they are arrays (possibly
    # per-star / per-plaquette / per-site), and JAX requires aux data to be
    # hashable.  The static
    # structure -- geometry sizes and which terms are active -- stays in aux so
    # it can drive Python-level branching under jit.
    leaves = [h.stars, h.plaquettes, h.A_s, h.B_p, h.h_z, h.h_x]
    aux = (h.dtype, h.N, h.Lx, h.Ly,
           h.n_stars, h.n_plaq, h.n_stars_off, h.n_hx_off,
           h.has_As, h.has_Bp, h.has_hz, h.has_hx,
           h.uniform_As, h.uniform_Bp, h.uniform_hz, h.uniform_hx)
    return leaves, aux


def _toric_unflatten(aux, leaves):
    (dtype, N, Lx, Ly,
     n_stars, n_plaq, n_stars_off, n_hx_off,
     has_As, has_Bp, has_hz, has_hx,
     uniform_As, uniform_Bp, uniform_hz, uniform_hx) = aux
    stars, plaquettes, A_s, B_p, h_z, h_x = leaves
    return ToricCode(
        dtype=dtype, stars=stars, plaquettes=plaquettes,
        N=N, Lx=Lx, Ly=Ly,
        A_s=A_s, B_p=B_p, h_z=h_z, h_x=h_x,
        n_stars=n_stars, n_plaq=n_plaq,
        n_stars_off=n_stars_off, n_hx_off=n_hx_off,
        has_As=has_As, has_Bp=has_Bp, has_hz=has_hz, has_hx=has_hx,
        uniform_As=uniform_As, uniform_Bp=uniform_Bp,
        uniform_hz=uniform_hz, uniform_hx=uniform_hx,
    )


jax.tree_util.register_pytree_node(ToricCode, _toric_flatten, _toric_unflatten)
