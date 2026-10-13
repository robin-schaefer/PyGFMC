"""Plaquette ring-exchange / Rokhsar-Kivelson model on arbitrary even loops.

    H = −g Σ_p flip_p + mu Σ_p |flippable_p⟩⟨flippable_p|

Configurations are ±1 floats.  See :class:`PlaquetteModel` for the
flippability rule, the rate layout and the sign convention.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class PlaquetteModel(NamedTuple):
    """Plaquette ring-exchange / Rokhsar–Kivelson Hamiltonian.

        ``H = −g · Σ_p (flip term on plaquette p) + mu · Σ_p |flippable_p⟩⟨flippable_p|``

    Each plaquette contributes

      - one off-diagonal element (the flip S⁺S⁻S⁺S⁻ … around the loop), active
        only when the plaquette is *flippable* — i.e. its spins strictly
        alternate as (+,−,+,−,…) or (−,+,−,+,…) in the order supplied to
        :meth:`create`.
      - one diagonal element (the RK counting potential μ_p) added when the
        plaquette is flippable.

    Layout of the rates returned by ``get_rates``:
      rates[0]               : diagonal,  Σ_p μ_p · 1[flippable_p]
      rates[1 : N_p + 1]     : one entry per plaquette, −g_p if flippable else 0

    The site count ``N`` is inferred from the plaquettes (``max index + 1``);
    configurations are ±1 floats.  Plaquettes must have EVEN length
    (:meth:`create` raises otherwise): an odd loop cannot alternate all the
    way round, so its "flip" would change ``Sz_tot`` and is no ring exchange.

    Fields:
        plaquettes       int array (N_plaquette, plaquette_size) — site indices
                         per plaquette, listed consecutively around the loop.
        g                flip coupling, shape (1,) uniform or (N_plaquette,).
                         Must be ≥ 0 (sign-problem-free convention): the
                         off-diagonal matrix element is −g ≤ 0.  The check can
                         be disabled with ``sign_check=False`` when
                         constructing measurement operators — note that an
                         off-diagonal *measurement* operator needs rates ≥ 0,
                         i.e. g ≤ 0, the opposite restriction.
        mu               RK counting coupling (diagonal, any sign), shape
                         (1,) uniform or (N_plaquette,).
    """

    dtype: object
    plaquettes: jnp.ndarray
    N: int
    g: jnp.ndarray
    mu: jnp.ndarray
    signs: jnp.ndarray          # cache [+1, -1, +1, -1, ...] of length plaquette_size
    n_plaquette: int
    plaquette_size: int
    has_g: bool = False
    has_mu: bool = False
    uniform_g: bool = True
    uniform_mu: bool = True

    @property
    def M_total(self):
        """Total transitions incl. diagonal (static Python int): N_plaquette + 1."""
        return self.n_plaquette + 1

    @property
    def n_off(self):
        """Number of off-diagonal elements (static Python int)."""
        return self.n_plaquette

    @classmethod
    def create(cls, plaquettes, g=None, mu=None, *, dtype=jnp.float32, sign_check=True):
        """Build a plaquette ring-exchange / RK Hamiltonian.

        Args:
            plaquettes: (N_plaquette, plaquette_size) site indices, listed
                        consecutively around each loop.  ``N = max index + 1``,
                        and every site from 0 to N-1 must appear in at least
                        one plaquette (a gap raises ``ValueError``).  The
                        plaquette length must be EVEN (odd raises, whatever
                        ``sign_check``).
            g:          scalar, (N_plaquette,) or None; flip amplitude in
                        ``-g Σ flip``.  Must be ≥ 0 when ``sign_check``.
            mu:         scalar, (N_plaquette,) or None; RK potential on
                        flippable plaquettes (any sign).
            dtype:      JAX dtype of the couplings.
            sign_check: enforce ``g ≥ 0``.  Pass False for an off-diagonal
                        measurement operator, which instead needs ``g ≤ 0``.
        """
        plaq_arr = np.asarray(plaquettes)
        if (plaq_arr.dtype.kind not in "iuf"
                or not np.all(np.isfinite(plaq_arr))
                or np.any(plaq_arr < 0) or np.any(plaq_arr >= 2**63)
                or np.any(plaq_arr != np.floor(plaq_arr))):
            raise ValueError("plaquette indices must be nonnegative integers fitting in int64.")
        plaq_arr = plaq_arr.astype(np.int64)
        if plaq_arr.ndim != 2:
            raise ValueError(
                f"plaquettes must have shape (N_plaquette, plaquette_size); got {plaq_arr.shape}"
            )
        N_plaq, plaq_size = plaq_arr.shape
        # An odd loop cannot strictly alternate all the way round: the
        # flippability test |sum signs*sigma| > l - 0.5 is then satisfied by
        # configs whose alternation breaks at the closure (l=3: (+,-,+)), and
        # flipping them changes Sz_tot by one -- not a ring exchange.  Raised
        # regardless of sign_check: the diagonal RK projector and an
        # off-diagonal measurement built on odd loops are equally meaningless.
        if N_plaq and plaq_size % 2:
            raise ValueError(
                f"plaquettes must have EVEN length, got {plaq_size}: an odd "
                "loop cannot alternate around its closure, so its 'flip' "
                "changes Sz_tot and is not a ring exchange.")
        # The site count comes from the plaquettes alone -- there is no N
        # argument.  Every site of this model belongs to at least one ring by
        # construction (a site no plaquette touches has no term acting on it),
        # so `max index + 1` IS the site count, and a gap in the indices means
        # the caller's site numbering does not match the plaquettes it passed.
        # Left to inference, such a gap would silently shorten N and misalign
        # every walker configuration against the flip masks; it is therefore
        # rejected here, where the numbering is still in view.
        if N_plaq == 0:
            N = 0
        else:
            used = np.unique(plaq_arr)
            if used[0] < 0:
                raise ValueError(
                    f"plaquettes contain negative site indices, smallest "
                    f"{int(used[0])}.")
            N = int(used[-1]) + 1
            if used.size != N:
                missing = np.setdiff1d(np.arange(N), used)
                shown = ", ".join(str(int(m)) for m in missing[:10])
                more = "" if missing.size <= 10 else f", ... ({missing.size} total)"
                raise ValueError(
                    f"every site from 0 to {N - 1} must appear in at least one "
                    f"plaquette, but {shown}{more} appear in none. The site "
                    "count is inferred from the plaquettes, so a gap would "
                    "silently give N={} instead of the {} sites the numbering "
                    "implies -- renumber the plaquettes, or pass the sites "
                    "that are actually covered.".format(N, N))

        idx_dtype = jnp.int16 if N <= 32767 else jnp.int32
        plaquettes_jax = jnp.array(plaq_arr, dtype=idx_dtype)

        # g — off-diagonal flip coupling
        if g is None:
            g_arr, has_g, uniform_g = jnp.zeros(1, dtype=dtype), False, True
        elif np.ndim(g) > 0:
            g_arr = jnp.asarray(g, dtype=dtype)
            if g_arr.shape[0] != N_plaq:
                raise ValueError(
                    f"Per-plaquette g must have length N_plaquette={N_plaq}, got {g_arr.shape[0]}."
                )
            if sign_check and np.any(np.asarray(g_arr) < 0):
                raise ValueError("g must be >= 0 for all plaquettes (sign-problem-free convention).")
            has_g = float(np.linalg.norm(np.asarray(g_arr))) > 0
            uniform_g = False
        else:
            if sign_check and g < 0:
                raise ValueError("g must be >= 0 (sign-problem-free convention).")
            g_arr = jnp.array([g], dtype=dtype)
            has_g = bool(g != 0)
            uniform_g = True

        # mu — diagonal counting coupling (no sign restriction)
        if mu is None:
            mu_arr, has_mu, uniform_mu = jnp.zeros(1, dtype=dtype), False, True
        elif np.ndim(mu) > 0:
            mu_arr = jnp.asarray(mu, dtype=dtype)
            if mu_arr.shape[0] != N_plaq:
                raise ValueError(
                    f"Per-plaquette mu must have length N_plaquette={N_plaq}, got {mu_arr.shape[0]}."
                )
            has_mu = float(np.linalg.norm(np.asarray(mu_arr))) > 0
            uniform_mu = False
        else:
            mu_arr = jnp.array([mu], dtype=dtype)
            has_mu = bool(mu != 0)
            uniform_mu = True

        signs = jnp.array([(-1.0) ** i for i in range(plaq_size)], dtype=dtype)

        return cls(
            dtype=dtype,
            plaquettes=plaquettes_jax,
            N=N,
            g=g_arr,
            mu=mu_arr,
            signs=signs,
            n_plaquette=N_plaq,
            plaquette_size=plaq_size,
            has_g=has_g,
            has_mu=has_mu,
            uniform_g=uniform_g,
            uniform_mu=uniform_mu,
        )

    def _flippable_mask(self, config):
        """Bool array (N_plaquette,) — True where the plaquette spins alternate ±1."""
        plaq_spins = config[self.plaquettes]                    # (N_plaquette, plaquette_size)
        alt_sum = jnp.sum(self.signs * plaq_spins, axis=-1)     # (N_plaquette,)
        return jnp.abs(alt_sum) > (self.plaquette_size - 0.5)

    @property
    def local_hilbert_space(self):
        """Per-site values, ordered so that position i is basis digit i.

        Matches this class's own basis convention: spin-1/2 degrees of freedom encoded as +-1.
        """
        return (1.0, -1.0)

    def _diagonal_from_mask(self, flippable):
        """Σ_p μ_p · 1[flippable_p] from the flippability mask (0 without mu)."""
        if not self.has_mu:
            return jnp.array(0.0, dtype=self.dtype)
        mu_bc = jnp.broadcast_to(self.mu, (self.n_plaquette,))
        return jnp.sum(jnp.where(flippable, mu_bc, jnp.zeros((), dtype=self.dtype)))

    def diagonal_weight(self, config):
        """Scalar diagonal: Σ_p μ_p · 1[flippable_p]. Matches get_rates(c)[0][0]."""
        return self._diagonal_from_mask(self._flippable_mask(config))

    def get_rates(self, config, return_configs=False):
        """Return rates (M_total,) and optionally the (M_total, N) connected-configs tensor."""
        flippable = self._flippable_mask(config)
        diag_weight = self._diagonal_from_mask(flippable)

        M = self.M_total
        rates = jnp.zeros(M, dtype=self.dtype).at[0].set(diag_weight)

        if self.has_g:
            g_bc = jnp.broadcast_to(self.g, (self.n_plaquette,))
            rates = rates.at[1:].set(jnp.where(flippable, -g_bc, jnp.zeros((), dtype=self.dtype)))

        if not return_configs:
            return rates, None

        new_configs = jnp.zeros((M, self.N), dtype=config.dtype).at[0].set(config)
        rows = jnp.broadcast_to(jnp.arange(self.n_plaquette)[:, None], self.plaquettes.shape)
        flip_mask = (jnp.zeros((self.n_plaquette, self.N), dtype=bool)
                     .at[rows, self.plaquettes].set(True))
        base = jnp.broadcast_to(config[None, :], (self.n_plaquette, self.N))
        plaq_configs = jnp.where(flip_mask, -base, base)
        new_configs = new_configs.at[1:].set(plaq_configs)
        return rates, new_configs

    def generate_state(self, config, idx):
        """Apply off-diagonal transition.

        idx == 0 : return config unchanged (diagonal).
        idx >= 1 : flip all sites of plaquette (idx - 1).
        """
        plaq_idx = idx - 1
        sites = self.plaquettes[plaq_idx]
        flipped = config.at[sites].set(-config[sites])
        return jnp.where(idx == 0, config, flipped)




def _plaquette_flatten(p):
    leaves = [p.plaquettes, p.g, p.mu, p.signs]
    aux = (p.N, p.dtype, p.n_plaquette, p.plaquette_size,
           p.has_g, p.has_mu, p.uniform_g, p.uniform_mu)
    return leaves, aux


def _plaquette_unflatten(aux, leaves):
    N, dtype, n_plaq, plaq_size, has_g, has_mu, uniform_g, uniform_mu = aux
    plaquettes, g, mu, signs = leaves
    return PlaquetteModel(
        dtype=dtype, plaquettes=plaquettes, N=N, g=g, mu=mu, signs=signs,
        n_plaquette=n_plaq, plaquette_size=plaq_size,
        has_g=has_g, has_mu=has_mu, uniform_g=uniform_g, uniform_mu=uniform_mu,
    )


jax.tree_util.register_pytree_node(PlaquetteModel, _plaquette_flatten, _plaquette_unflatten)
