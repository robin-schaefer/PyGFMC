#!/usr/bin/env python
"""Generic NetKet training: any guide with ``log_psi``, any Hamiltonian with ``get_rates``.

Training needs two pieces: a Flax module so NetKet can own the parameters,
and code to unpack the trained parameters back into the guide's class.  Both
follow from the guide protocol alone, so :func:`train_with_netket` is one
function for any guide exposing ``log_psi`` and any Hamiltonian exposing
``get_rates``.

Three design points:

**The Hilbert space is built from the local Hilbert space directly.**
``HomogeneousHilbert(StaticRange(start, step, length), N)`` has exactly the
per-site values PyGFMC uses, and its samples come out in those values, so
NetKet configurations *are* PyGFMC configurations and no re-encoding happens
anywhere.  Going through ``nk.hilbert.Spin`` instead would need one: NetKet's
``Spin(s=1)`` has local states ``[2, 0, -2]``, i.e. 2m rather than m, so a
bridge that passed configurations straight through would be silently wrong for
every S > 1/2.  Building the space from the ladder also covers ladders that
have no ``Spin``/``Fock`` equivalent.

**Only the floating-point pytree leaves are trained.**  "A guide's pytree
children are its trainable parameters" is not true of the real guides:
``JastrowSpinHalf`` flattens to ``(site_other, J, site_valid_mask, h, _pair_site,
_pair_slot, _site_pair)``, where ``site_other`` and the three ``_pair_*`` /
``_site_pair`` arrays are integer index arrays and ``site_valid_mask`` is a
boolean mask.  ``J`` and ``h`` keep the form
the guide was built with (a single tied value, one per pair or site, or no
``h`` at all), so training mirrors the parametrisation it was given.  Casting every leaf to the parameter dtype turns the index array into
floats and the guide dies with "Indexer must have integer or boolean type".
Integer and boolean leaves are therefore held fixed.  This is a heuristic, not
a guarantee -- a guide with a deliberately integer-valued parameter would not
be trained, and every float leaf IS trained (e.g. ``JastrowSpinHalf``'s field ``h``)
-- but it matches every guide in the package.

**The operator is a pass-through.**  NetKet's ``DiscreteJaxOperator`` contract
maps one-to-one onto the rate protocol,

    get_conn_padded(x) -> (x_primes, mels)   <->   get_rates(c, return_configs=True)

with no factors to reconcile.
This relies on ``rates`` being the true matrix elements of H (``rates[0]`` the
diagonal, ``rates[k]`` the element to successor ``k``), which the Hamiltonian
protocol requires.

**Training runs in float32.**  ``import netket`` would otherwise set
``jax_enable_x64`` True process-wide (``NETKET_ENABLE_X64`` defaults to True;
see netket/utils/config_flags.py), and leaking it hands float64 walkers to a
float32 Hamiltonian, so the GFMC scan fails with a carry-type mismatch nowhere
near the cause.  The trainer therefore sets ``NETKET_ENABLE_X64=0`` before the
first ``import netket``, switches ``jax_enable_x64`` off while it trains, and
restores the caller's setting on the way out.  The NetKet parameters are
float32, and the trained parameters are cast back to the dtypes the input
guide used.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

import jax
import jax.numpy as jnp
from .._precision import float_dtype

__all__ = ["train_with_netket", "make_hilbert", "make_netket_operator",
           "make_guide_model", "rebuild_guide"]


@contextlib.contextmanager
def _netket_float32():
    """Train in float32, and put the caller's x64 flag back afterwards.

    Sets NETKET_ENABLE_X64=0 (unless already set) before the first
    ``import netket``, which would otherwise switch on x64 process-wide.

    Also sets NETKET_EXPERIMENTAL_SHARDING=0 (unless already set) before the
    first ``import netket``: its sharding path leaks a named mesh into JAX's
    thread-local config and breaks later ``jax.pmap`` calls.  Doing it here
    rather than at module scope keeps ``import pygfmc.wavefunction`` free of side
    effects.
    """
    os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "0")
    os.environ.setdefault("NETKET_ENABLE_X64", "0")
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# --------------------------------------------------------------------------
# Hilbert space straight from the local Hilbert space
# --------------------------------------------------------------------------
def make_hilbert(local_hilbert_space, N, total=None):
    """A NetKet Hilbert space whose local states are exactly these values.

    Returns ``nk.hilbert.HomogeneousHilbert(StaticRange(...), N)``.  Requires
    an evenly spaced ladder, which every convention in use is (``(-1, 1)``,
    ``(-S, ..., S)``, ``(0, ..., n_max)``).  With ``total``, the space is
    restricted to the configurations whose site values sum to ``total``.
    """
    from ._optional import import_netket
    nk = import_netket()
    from netket.utils.static_range import StaticRange

    values = np.asarray(local_hilbert_space, dtype=float).ravel()
    if values.size < 2:
        raise ValueError(
            f"local_hilbert_space needs at least two values, got "
            f"{values.tolist()}.")
    step = values[1] - values[0]
    if not np.allclose(np.diff(values), step):
        raise ValueError(
            f"local_hilbert_space must be evenly spaced to become a NetKet "
            f"StaticRange, got {values.tolist()}.")
    if total is not None and float(total).is_integer():
        total = int(round(float(total)))   # NetKet's random states need an integer sum
    constraint = (None if total is None
                  else nk.hilbert.constraint.SumConstraint(total))
    # An integer ladder (spins +-1, occupations 0..n_max) stays integer:
    # NetKet's constrained sampler divides the sum by start and step and needs
    # an integer excitation count.  Guides and Hamiltonians cast configurations
    # to their own dtype.  Any other ladder is float32.
    if float(values[0]).is_integer() and float(step).is_integer():
        local = StaticRange(start=int(round(values[0])), step=int(round(step)),
                            length=values.size, dtype=np.int32)
    else:
        local = StaticRange(start=float(values[0]), step=float(step),
                            length=values.size, dtype=np.float32)
    return nk.hilbert.HomogeneousHilbert(local, N=int(N), constraint=constraint)


def _default_sector(local_hilbert_space, N, initial_states):
    """The fixed sum of site values that the default exchange sampler keeps.

    The sum of ``initial_states`` when given (all rows must share it), else
    the balanced value ``N * mean(local values)``: ``S^z_tot = 0`` for spins,
    one particle per site for bosons.
    """
    values = np.asarray(local_hilbert_space, dtype=float).ravel()
    if initial_states is not None:
        sums = np.unique(np.round(np.asarray(initial_states, dtype=float).sum(axis=1), 6))
        if sums.size != 1:
            raise ValueError(
                f"initial_states lie in several sectors (site-value sums "
                f"{sums.tolist()}); the default MetropolisExchange sampler keeps "
                f"one sector, so pass states of one sector or a sampler.")
        return float(sums[0])
    total = N * values.mean()
    step = values[1] - values[0]
    if not np.isclose((total - N * values[0]) / step,
                      np.round((total - N * values[0]) / step)):
        raise ValueError(
            f"the default MetropolisExchange sampler keeps the balanced sector "
            f"(site values summing to {total:g}), which {N} sites with local "
            f"values {values.tolist()} cannot reach.  Pass initial_states in "
            f"the sector you want, or a sampler, e.g. "
            f"nk.sampler.MetropolisLocal(hilbert) for no constraint.")
    return float(total)


def _exchange_pairs(hamiltonian, N):
    """Site pairs for MetropolisExchange: the Hamiltonian edges, else all pairs."""
    edges = getattr(hamiltonian, "edges", None)
    if edges is not None and len(np.asarray(edges)):
        return np.asarray(edges, dtype=np.int64).reshape(-1, 2).tolist()
    return [[i, j] for i in range(N) for j in range(i + 1, N)]


def _require_sum_conserved(hamiltonian, local_hilbert_space, N, seed,
                           n_configs=8):
    """Raise unless ``hamiltonian`` conserves the sum of the site values.

    The default ``MetropolisExchange`` sampler keeps that sum fixed, so for a
    Hamiltonian that changes it (a transverse field, a pair term,
    ``J_x != J_y``) it would train in one sector of a Hamiltonian that mixes
    them.  Checked on ``n_configs`` random configurations: every successor
    with a nonzero rate must have the same sum as its parent.
    """
    values = np.asarray(local_hilbert_space, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    dtype = getattr(hamiltonian, "dtype", jnp.float32)
    for cfg in values[rng.integers(0, values.size, size=(n_configs, N))]:
        rates, successors = hamiltonian.get_rates(jnp.asarray(cfg, dtype=dtype),
                                                  return_configs=True)
        moved = np.asarray(rates)[1:] != 0
        sums = np.asarray(successors, dtype=float)[1:].sum(axis=1)
        if np.any(moved & ~np.isclose(sums, cfg.sum())):
            raise ValueError(
                f"{type(hamiltonian).__name__} changes the sum of the site "
                "values (e.g. S^z_tot or the particle number), but the "
                "default MetropolisExchange sampler conserves it and would "
                "train in a single sector. Pass a sampler that changes single "
                "sites, e.g. sampler=nk.sampler.MetropolisLocal(hilbert) with "
                "hilbert = make_hilbert(local_hilbert_space, N).")


# --------------------------------------------------------------------------
# Hamiltonian -> NetKet operator
# --------------------------------------------------------------------------
def make_netket_operator(hamiltonian, hilbert, is_hermitian=True):
    """Wrap any ``get_rates`` Hamiltonian as a NetKet ``DiscreteJaxOperator``.

    ``is_hermitian`` has to be declared: ``AbstractOperator`` assumes False, and
    NetKet then routes the gradient through its experimental non-Hermitian path
    and refuses to run.  True is right for every Hamiltonian in the package,
    where Hermiticity is structural, but a measurement operator need not be,
    so it stays a flag rather than a hard-coded True.

    Zero-rate successors (which may lie outside the physical basis) are mapped
    back to the source configuration so NetKet never evaluates the ansatz
    outside the Hilbert space.
    """
    from ._optional import import_netket
    import_netket()
    from netket.operator import DiscreteJaxOperator

    @jax.tree_util.register_pytree_node_class
    class RateProtocolOperator(DiscreteJaxOperator):
        """``get_conn_padded`` straight from ``get_rates``."""

        def __init__(self, hilbert, op, hermitian=True):
            super().__init__(hilbert)
            self._op = op
            self._hermitian = bool(hermitian)

        @property
        def is_hermitian(self):
            return self._hermitian

        @property
        def dtype(self):
            return float_dtype()

        @property
        def max_conn_size(self):
            return int(self._op.M_total)

        def get_conn_padded(self, x):
            lead, n_sites = x.shape[:-1], x.shape[-1]
            flat = x.reshape(-1, n_sites)
            rates, new_cfg = jax.vmap(
                lambda c: self._op.get_rates(c, return_configs=True))(flat)
            # Zero-rate transitions may point outside the physical basis
            # (S+ at m = +S, a boson past n_max).  They contribute nothing, but
            # NetKet would still evaluate the ansatz there, so send them back to
            # the source configuration and keep every state inside the space.
            keep = rates != 0.0
            new_cfg = jnp.where(keep[..., None], new_cfg, flat[:, None, :])
            m = self.max_conn_size
            return (new_cfg.reshape(lead + (m, n_sites)).astype(x.dtype),
                    rates.reshape(lead + (m,)).astype(self.dtype))

        def tree_flatten(self):
            return (self._op,), (self.hilbert, self._hermitian)

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(aux[0], children[0], aux[1])

    return RateProtocolOperator(hilbert, hamiltonian, is_hermitian)


# --------------------------------------------------------------------------
# guide -> Flax model
# --------------------------------------------------------------------------
def _split_leaves(guide):
    """(treedef, leaves, trainable indices) -- floats train, ints/bools do not."""
    leaves, treedef = jax.tree_util.tree_flatten(guide)
    if not leaves:
        raise ValueError(
            f"{type(guide).__name__} flattens to no pytree children, so it has "
            "no parameters to train.")
    trainable = [i for i, v in enumerate(leaves)
                 if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)]
    if not trainable:
        raise ValueError(
            f"{type(guide).__name__} has no floating-point pytree children, so "
            "there is nothing to train. Its leaves are "
            f"{[str(jnp.asarray(v).dtype) for v in leaves]}.")
    return treedef, [np.asarray(v) for v in leaves], trainable


def make_guide_model(guide, param_dtype=jnp.float32):
    """A Flax module wrapping ``guide.log_psi``, parametrised by its float leaves.

    Returns ``(model, treedef, leaves, trainable)`` for :func:`rebuild_guide`.
    """
    import flax.linen as nn

    treedef, leaves, trainable = _split_leaves(guide)

    class GuideModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            # jnp, not the numpy copies held for dtype bookkeeping: a static
            # integer leaf indexed by a traced value must be a jax array, or
            # numpy calls __array__ on the tracer and raises
            # TracerArrayConversionError (ClusterCPS indexes idx_to_param this
            # way).
            rebuilt = [jnp.asarray(v) for v in leaves]
            for i in trainable:
                rebuilt[i] = self.param(
                    f"leaf_{i}",
                    lambda _key, v=leaves[i]: jnp.asarray(v, param_dtype))
            g = jax.tree_util.tree_unflatten(treedef, rebuilt)
            flat = x.reshape(-1, x.shape[-1])
            # vmap rather than trusting the batch path: the protocol allows a
            # guide to implement log_psi for a single configuration only
            out = jax.vmap(lambda c: g.log_psi(c))(flat)
            return out.reshape(x.shape[:-1])

    return GuideModel(), treedef, leaves, trainable


def rebuild_guide(treedef, leaves, trainable, params):
    """Put trained parameters back into a guide of the original class."""
    rebuilt = list(leaves)
    for i in trainable:
        rebuilt[i] = np.asarray(params[f"leaf_{i}"]).astype(leaves[i].dtype)
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(v)
                                                  for v in rebuilt])


# --------------------------------------------------------------------------
# the standalone trainer
# --------------------------------------------------------------------------
def train_with_netket(hamiltonian, guide, local_hilbert_space=None, N=None,
                      sampler=None, initial_states=None, n_iter=100,
                      learning_rate=0.02, n_samples=1024, n_chains=16,
                      n_discard_per_chain=8, sr=False, diag_shift=1e-3,
                      qgt=None, seed=0, show_progress=False,
                      return_history=False):
    """Train ``guide`` on ``hamiltonian`` with NetKet VMC and return a new guide.

    Needs only ``guide.log_psi`` plus the guide's pytree registration, and
    ``hamiltonian.get_rates``.  Nothing is mutated: a new guide of the same
    class comes back, with parameters cast to their original dtypes.

    Args:
        hamiltonian:         Any object exposing ``M_total`` and ``get_rates``.
        guide:               Any registered-pytree guide exposing ``log_psi``.
        local_hilbert_space: Per-site values.  Defaults to
                             ``hamiltonian.local_hilbert_space``.
        N:                   Number of sites.  Defaults to ``hamiltonian.N``.
        sampler:             A ready-made NetKet sampler.  It carries its own
                             Hilbert space, so ``local_hilbert_space``/``N``
                             are then ignored.  The default is
                             ``MetropolisExchange``, which swaps the values of
                             two sites and so conserves their sum (``S^z_tot``
                             for spins, the particle number for bosons); it is
                             refused, with a ``ValueError``, for a Hamiltonian
                             that changes this sum.  It
                             samples one sector: that of ``initial_states``
                             when given, else the balanced one (``S^z_tot = 0``,
                             or one particle per site), and swaps along the
                             Hamiltonian's ``edges`` (all site pairs if it has
                             none).  For a Hamiltonian that does not conserve
                             this sum, e.g. a transverse field or a pair term,
                             pass ``nk.sampler.MetropolisLocal(hilbert)``, which
                             changes single sites. For local spaces with more
                             than two values, an explicit ergodic sampler is
                             required: swaps preserve the full value histogram,
                             not just the conserved sum. Use suitable transfer
                             proposals for a constrained higher-spin/boson space.
        initial_states:      ``(n, N)`` configurations seeding the Metropolis
                             chains, resampled to the chain count.  This is the
                             argument that changes results rather than just
                             cost: chains started from a random product state
                             can take very long to random-walk into a
                             constrained manifold such as the spin-ice
                             2-in-2-out states, and they must lie in the sector
                             of interest to begin with.
        n_iter, learning_rate, n_samples, n_chains, n_discard_per_chain:
                             VMC settings.
        sr, diag_shift, qgt: Stochastic reconfiguration, its regularisation and
                             the quantum-geometric-tensor implementation.
                             ``sr`` defaults to False because NetKet's default
                             on-the-fly QGT can fail with recent JAX versions
                             (JAX registers ``Tracer`` as a ``jax.Array``, so
                             plum cannot disambiguate NetKet's two
                             ``is_scalar`` overloads and SR dies with
                             ``AmbiguousLookupError`` inside
                             ``qgt_onthefly_logic``).  With NetKet 3.21,
                             ``qgt=nk.optimizer.qgt.QGTJacobianDense`` works,
                             while ``QGTJacobianPyTree`` and the default fail.
                             The optimiser is ``optax.sgd(learning_rate)``.
        seed:                Seeds parameter init and the sampler.
        show_progress:       Show NetKet's progress bar.
        return_history:      Also return the per-iteration energy array.

    Returns:
        The trained guide, or ``(guide, history)`` when ``return_history``.
    """
    if sampler is None:
        if local_hilbert_space is None:
            local_hilbert_space = getattr(hamiltonian, "local_hilbert_space",
                                          None)
            if local_hilbert_space is None:
                raise ValueError(
                    f"local_hilbert_space was not given and "
                    f"{type(hamiltonian).__name__} exposes none.")
        if N is None:
            N = getattr(hamiltonian, "N", None)
            if N is None:
                raise ValueError(
                    f"N was not given and {type(hamiltonian).__name__} "
                    "exposes none.")

    if sampler is None and np.unique(local_hilbert_space).size > 2:
        raise ValueError(
            "An explicit ergodic sampler is required for local spaces with more "
            "than two values: MetropolisExchange preserves the value histogram, "
            "not just total charge. Supply a sampler with suitable transfer proposals.")
    if sampler is None:
        _require_sum_conserved(hamiltonian, local_hilbert_space, N, seed)

    with _netket_float32():
        from ._optional import import_netket
        nk = import_netket()
        import optax

        if sampler is None:
            total = _default_sector(local_hilbert_space, N, initial_states)
            sampler = nk.sampler.MetropolisExchange(
                make_hilbert(local_hilbert_space, N, total=total),
                clusters=_exchange_pairs(hamiltonian, N), n_chains=n_chains)
        # a sampler carries its own space, so it is the single source of truth
        hi = sampler.hilbert
        operator = make_netket_operator(hamiltonian, hi)
        model, treedef, leaves, trainable = make_guide_model(guide)

        vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples,
                                n_discard_per_chain=n_discard_per_chain,
                                seed=seed, sampler_seed=seed)
        if initial_states is not None:
            sigma = np.asarray(initial_states)
            n_sites = int(hi.size)
            if sigma.ndim != 2 or sigma.shape[1] != n_sites:
                raise ValueError(
                    f"initial_states must have shape (n, N={n_sites}); got "
                    f"{sigma.shape}")
            n_have = sigma.shape[0]
            n_want = int(vstate.sampler_state.σ.shape[0])
            if n_have != n_want:
                # draw with replacement only when short, so a stored pool of
                # any size can seed any number of chains
                rng = np.random.default_rng(seed)
                sigma = sigma[rng.choice(n_have, size=n_want,
                                         replace=(n_have < n_want))]
            vstate.sampler_state = vstate.sampler_state.replace(
                σ=jnp.asarray(sigma, dtype=vstate.sampler_state.σ.dtype))
        driver = nk.driver.VMC(
            operator, optax.sgd(learning_rate=learning_rate),
            variational_state=vstate,
            preconditioner=(nk.optimizer.SR(
                                qgt=qgt, diag_shift=diag_shift,
                                holomorphic=False)
                            if sr else None))

        history = []
        driver.run(n_iter=n_iter, show_progress=show_progress,
                   callback=lambda step, log, dr: (
                       history.append(float(np.real(dr.energy.mean))) or True))

        trained = rebuild_guide(treedef, leaves, trainable,
                               jax.tree_util.tree_map(np.asarray,
                                                      vstate.parameters))

    return (trained, np.asarray(history)) if return_history else trained
