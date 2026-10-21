"""Walker propagation: the three GFMC propagators and the forward-walking
operator estimator.

``evolve_walkers`` advances a walker batch by one branching step with one of
``"single_step"``, ``"multi_step"`` or ``"cont_time"``;
``forward_walk_operator`` applies an off-diagonal operator to the ensemble and
propagates the result along a shadow chain to form the pure estimator.
"""
import jax
import jax.numpy as jnp
import math
from functools import partial
from typing import Any, Optional, Union

from ..walkers import Walker, reconfiguration
from ..protocols import Operator, Guide

# The operator protocol (:class:`pygfmc.Operator`).  ``generate_state`` is
# also required (for any operator with M_total > 1): the propagators call it
# whenever ``get_rates`` was not asked to materialise the (M_total, N)
# successor tensor, which is the normal path -- both for a guide-free run and
# for any guide exposing ``log_diff_from_operator``.  Checking up front turns
# what would be an AttributeError deep inside a jitted scan into a clear
# TypeError.
_REQUIRED_OPERATOR_ATTRS = ("M_total", "get_rates")
# The guide protocol (:class:`pygfmc.Guide`): one of these, tried in order.
_GUIDE_RATIO_METHODS = ("log_diff_from_operator", "log_diff_from_configs", "log_psi")


def sample_index(cdf, weights, key):
    """Draw a slot from ``cdf = cumsum(weights)`` (non-negative ``weights``).

    Requires a positive total weight.  Only slots with ``weights > 0`` are
    returned for a cumulative sum of nonnegative weights.  A plain
    ``searchsorted(cdf, uniform * cdf[-1])`` does not guarantee that, for two
    float reasons:

    * XLA evaluates ``cumsum`` as a parallel prefix scan, so ``cdf`` can rise
      by an ulp ACROSS a zero-weight slot, and ``u`` may land in that sliver.
    * ``uniform * total`` can round up to ``total`` itself, and
      ``searchsorted`` then returns ``len(cdf)``, which JAX clamps to the last
      slot whatever its weight.

    Either way a move with zero amplitude is taken.  For a spin model that is
    a silent, rare error; for a model whose zero-weight slots lead outside the
    physical basis (a Bose-Hubbard pair annihilation on an empty site) it
    gives an occupation of -1 and NaN in the next step.  Here ``u`` stays one
    ulp below the total, and a draw that still lands on a zero-weight slot is
    moved to the nearest positive slot before it -- an ulp-sized shift of
    probability, against a ~5 % cost for making the cdf exactly flat.
    """
    M = cdf.shape[0]
    total = cdf[-1]
    u = jnp.minimum(jax.random.uniform(key) * total,
                    jnp.nextafter(total, jnp.zeros_like(total)))
    idx = jnp.minimum(jnp.searchsorted(cdf, u, side="right"), M - 1)
    slots = jnp.arange(M)
    prev = jnp.max(jnp.where((weights > 0) & (slots <= idx), slots, -1))
    return jnp.where(prev >= 0, prev, idx)


def require_operator_protocol(op: Operator, role: str = "hamiltonian") -> None:
    """Raise ``TypeError`` unless ``op`` implements :class:`pygfmc.Operator`.

    ``generate_state`` is required for any operator that can actually produce a
    transition (``M_total > 1``).  It is NOT required at ``M_total == 1``: such
    an operator is purely diagonal, the propagators short-circuit it, and there
    is no successor state to generate -- a measurement-only identity is the
    canonical example.

    Args:
        op:   Hamiltonian or off-diagonal observable to check.
        role: How to name it in the error message.
    """
    missing = [name for name in _REQUIRED_OPERATOR_ATTRS
               if not hasattr(op, name)]
    if missing:
        raise TypeError(
            f"{role} {type(op).__name__!r} is missing required attribute(s) "
            f"{', '.join(missing)}; the operator protocol (pygfmc.Operator) "
            f"requires {', '.join(_REQUIRED_OPERATOR_ATTRS)}."
        )
    if int(op.M_total) > 1 and not hasattr(op, "generate_state"):
        raise TypeError(
            f"{role} {type(op).__name__!r} has M_total={int(op.M_total)} but no "
            "generate_state; it is required for any operator with an "
            "off-diagonal transition, even when "
            "get_rates(return_configs=True) is available, because the "
            "propagators avoid materialising the (M_total, N) successor tensor "
            "whenever they can."
        )


def require_guide_protocol(guide: Optional[Guide]) -> None:
    """Raise ``TypeError`` unless ``guide`` is None or implements
    :class:`pygfmc.Guide`, i.e. at least one of ``log_diff_from_operator``,
    ``log_diff_from_configs`` and ``log_psi``."""
    if guide is not None and not any(hasattr(guide, name)
                                     for name in _GUIDE_RATIO_METHODS):
        raise TypeError(
            f"Guide {type(guide).__name__!r} must implement at least one of "
            "log_diff_from_operator, log_diff_from_configs, or log_psi "
            "(the guide protocol, pygfmc.Guide)."
        )


def uses_guide_state(guide) -> bool:
    """Whether ``guide`` runs in the incremental mode (its ``incremental``
    attribute is set; see :class:`pygfmc.Guide`)."""
    return guide is not None and bool(getattr(guide, "incremental", False))


def require_incremental_support(guide, hamiltonian) -> None:
    """Raise ``TypeError`` when an incremental guide lacks ``init_state`` /
    ``update_state`` or ``hamiltonian`` lacks ``move`` (see
    :class:`pygfmc.Guide`, :class:`pygfmc.Operator`)."""
    if not uses_guide_state(guide):
        return
    missing = [n for n in ("init_state", "update_state") if not hasattr(guide, n)]
    if missing:
        raise TypeError(
            f"Guide {type(guide).__name__!r} has incremental=True but no "
            f"{', '.join(missing)}; the incremental mode needs both.")
    if not hasattr(hamiltonian, "move"):
        raise TypeError(
            f"hamiltonian {type(hamiltonian).__name__!r} has no move(config, idx), "
            f"which the incremental mode of guide {type(guide).__name__!r} needs "
            "(pass incremental=False, or add the method).")


def initial_carried(walkers, hamiltonian, guide, cache, with_rates=True):
    """The carried per-walker state ``(rates, state)`` of ``walkers``: their
    guide-transformed rates (None unless ``with_rates``) and, for an
    incremental guide, its state, both evaluated here once."""
    state = (jax.vmap(lambda config: guide.init_state(config, cache))(walkers.config)
             if uses_guide_state(guide) else None)
    if not with_rates:
        return None, state
    if state is None:
        rates = jax.vmap(lambda config: transition_rates(
            config, hamiltonian, guide, cache=cache)[0])(walkers.config)
    else:                                   # the fresh state, not a second rebuild
        rates = jax.vmap(lambda config, st: transition_rates(
            config, hamiltonian, guide, cache=cache, state=st)[0])(walkers.config, state)
    return rates, state


def _carry_rates_bytes():
    """The size limit of the carried rate array (0: the rates are never
    carried), ``PYGFMC_RATES_CARRY_MAX_BYTES``; see the driver."""
    from .driver import _RATES_CARRY_MAX_BYTES
    return _RATES_CARRY_MAX_BYTES


def rates_fit_budget(n_walkers, hamiltonian, dtype, budget):
    """Whether the global population's carried rates fit the opt-in budget."""
    return (budget > 0 and int(n_walkers) * int(hamiltonian.M_total)
            * jnp.dtype(dtype).itemsize <= budget)


# Every this many steps of a chain, counted from the start of the chain (a
# continuation passes its absolute offset), the carried pair of an
# incremental guide is rebuilt from the configurations: the guide state from
# scratch and the rates from that state, so the rounding of the incremental
# updates cannot accumulate and the schedule does not depend on how the run
# is chunked or split.
_STATE_REFRESH_STEPS = 100


def refresh_carried(carried, walkers, hamiltonian, guide, cache):
    """``carried`` rebuilt from the configurations when a guide state is
    carried (both the state and the rates, so the pair stays consistent);
    unchanged otherwise."""
    rates, state = carried
    if state is None:
        return carried
    return initial_carried(walkers, hamiltonian, guide, cache,
                           with_rates=rates is not None)


def refresh_carried_at(step, carried, walkers, hamiltonian, guide, cache):
    """:func:`refresh_carried` when the chain step ``step`` (a traced
    scalar) is a multiple of ``_STATE_REFRESH_STEPS``, else ``carried``;
    a no-op for a guide without state."""
    if not uses_guide_state(guide):
        return carried
    return jax.lax.cond(step % _STATE_REFRESH_STEPS == 0,
                        lambda c: refresh_carried(c, walkers, hamiltonian, guide, cache),
                        lambda c: c, carried)


def guide_cache(guide: Optional[Guide], op: Operator) -> Any:
    """The guide's optional per-operator cache, or None.

    Calls ``guide.build_cache(op)``; None when there is no guide or it
    defines no ``build_cache``.  Every engine call site builds its caches
    through this one helper.
    """
    if guide is None:
        return None
    build = getattr(guide, "build_cache", None)
    return None if build is None else build(op)


def transition_rates(config: jax.Array, op: Operator, guide: Optional[Guide],
                     return_configs: bool = False, cache: Any = None,
                     state: Any = None) -> tuple[jax.Array, Optional[jax.Array]]:
    """Return ``(rates, new_configs)`` with guide importance ratios folded in:
    ``rates[k] = w_k · ψ_T(σ′_k) / ψ_T(σ)``.  ``guide=None`` ⇒ bare rates.

    ``rates[0]`` is the diagonal element H_xx (unchanged by the guide) and
    ``rates[1:]`` the guide-transformed off-diagonal elements, so
    ``sum(rates)`` is the local energy e_L(x).  ``new_configs`` is the
    ``(M_total, N)`` successor tensor when it was materialised, else None.

    Dispatch:
      - If the guide exposes ``log_diff_from_operator``, use it.  ``cache``
        (a guide-specific cache from ``guide.build_cache(op)``, e.g.
        ``JastrowOpCache`` / ``ClusterCPSCache`` / ``ClusterMLPCache``) selects the
        cached fast path; ``cache=None`` falls through to the on-the-fly
        path inside ``log_diff_from_operator``.  Operator-specific fast paths
        avoid ``(M_total, N)`` materialisation; a guide may still materialise
        successors internally for an unsupported operator.  ``state`` is the
        walker's guide state of the incremental mode (see
        :class:`pygfmc.Guide`), passed on when given.
      - If the guide exposes ``log_diff_from_configs``, force
        ``return_configs=True`` so ``op.get_rates`` materialises the successor
        tensor, then call ``guide.log_diff_from_configs(config, new_configs)``.
      - Otherwise, if the guide exposes ``log_psi``, materialise configs and
        compute the ratio as ``vmap(guide.log_psi)(new_configs) - guide.log_psi(config)``.
    """
    require_guide_protocol(guide)
    has_from_operator = guide is not None and hasattr(guide, 'log_diff_from_operator')
    has_from_configs  = guide is not None and hasattr(guide, 'log_diff_from_configs')
    # With no guide there is no successor-ratio calculation, so retain the
    # Hamiltonian's lightweight generate_state path.  Materialising all
    # successors is especially prohibitive for a many-plaquette ring model:
    # it has shape (N_plaquette + 1, N_sites) per walker.
    if guide is not None and not has_from_operator:
        return_configs = True

    rates, new_configs = op.get_rates(config, return_configs=return_configs)
    if guide is not None:
        if has_from_operator:
            if state is None:
                log_diff = guide.log_diff_from_operator(config, op, cache)
            else:
                log_diff = guide.log_diff_from_operator(config, op, cache, state=state)
        elif has_from_configs:
            log_diff = guide.log_diff_from_configs(config, new_configs)
        else:
            log_diff = jax.vmap(guide.log_psi)(new_configs) - guide.log_psi(config)
        # A forbidden transition has a bare rate of exactly 0 and a ratio the
        # guide evaluates on a successor outside the physical space, which
        # can overflow: 0 * inf would be NaN, so its log-ratio is replaced by
        # 0 before the exponential (0 * 1 = 0).  Allowed transitions see
        # their ratio unmasked, overflow included.  Masking the log-ratio
        # rather than the product keeps XLA's fusion of the product and the
        # following sum as it was, so e_loc stays bit-identical.
        rates = rates * jnp.exp(jnp.where(rates == 0, 0.0, log_diff))
    return rates, new_configs


def gather_carried(carried, indices):
    """Reorder every array of the carried per-walker state (rates, guide
    state; None entries are skipped) with the resampling ``indices``."""
    return jax.tree_util.tree_map(lambda a: a[indices], carried)


def evolve_carried(walkers, carried, hamiltonian, step_size, guide, method,
                   rng_key, avg_log_weight, lam=None, cache=None):
    """:func:`evolve_walkers` with the carried pair ``(rates, state)`` in and
    out: ``(walkers, carried)``.  With nothing carried the plain propagator
    runs and the final rates are not materialised (see
    :func:`_current_rates`); an entry that was not carried in stays None, so
    a scan carry keeps its pytree structure.  The caller gathers the result
    with the resampling indices (:func:`gather_carried`)."""
    rates, state = carried
    if rates is None and state is None:
        return evolve_walkers(walkers, hamiltonian, step_size, guide, method,
                              rng_key, avg_log_weight, lam=lam, cache=cache), carried
    walkers, rates_out, state_out = evolve_walkers(
        walkers, hamiltonian, step_size, guide, method, rng_key, avg_log_weight,
        lam=lam, cache=cache, rates=rates, state=state, return_rates=True)
    return walkers, (rates_out if rates is not None else None,
                     state_out if state is not None else None)


def _state_after_move(hamiltonian, guide, cache, state, config, idx):
    """The guide state after transition ``idx`` from ``config`` (None stays None)."""
    if state is None:
        return None
    sites, deltas = hamiltonian.move(config, idx)
    return guide.update_state(state, config, sites, deltas, cache)


def _current_rates(config, hamiltonian, guide, cache, rates, state=None):
    """``(rates, new_configs)`` of ``config``: the carried ``rates`` when the
    caller has them (then without materialised successors), else evaluated
    (with the guide ``state`` of the incremental mode, when given).

    Every propagator ends a step with the rates of its final configuration
    (they give ``e_loc``), and the next step starts from that configuration
    after reconfiguration has copied it, so the drivers carry the vector
    across the step boundary instead of evaluating it again: one
    ``transition_rates`` call per walker and step saved.  The carried values
    are exactly the ones the evaluation would return.  Returning the final
    rates means XLA materialises them before the local-energy sum instead of
    fusing the two, which can change the reduction order: ``e_loc`` may
    differ from the fused value by one float32 ulp (seen for the
    Bose-Hubbard guide), while configurations and weights are identical.
    """
    if rates is not None:
        return rates, None
    return transition_rates(config, hamiltonian, guide, cache=cache, state=state)


def _with_rates(walker, rates, state, return_rates):
    return (walker, rates, state) if return_rates else walker


def evolve_walker_single_step(walker, hamiltonian, n_bra, guide, rng_key,
                              avg_log_weight, lam, cache=None, rates=None,
                              state=None, return_rates=False):
    """Evolve one walker for ``n_bra`` elementary sub-steps of (λ − H).

    At each sub-step a successor (including the diagonal "stay") is drawn
    from the guide-transformed column of λ − H and log(b_x), with
    b_x = λ − e_L(x), is added to ``log_weight``; ``avg_log_weight * n_bra``
    is subtracted at the end.  ``e_loc`` is set to the local energy of the
    final configuration.

    Args:
        walker:          Single-walker pytree (config, log_weight, e_loc).
        hamiltonian:     Hamiltonian object.
        n_bra:           Number of sub-steps (the integer ``step_size`` of
                         :func:`evolve_walkers`).
        guide:           Optional guiding wavefunction.
        rng_key:         JAX PRNGKey; sub-step ``i`` draws from
                         ``fold_in(rng_key, i)``.
        avg_log_weight:  Normalisation shift per sub-step.
        lam:             The GFMC shift λ.
        cache:           Optional ``guide.build_cache(hamiltonian)`` result.
        rates:           Optional guide-transformed rates of ``walker.config``
                         when the caller already has them (see
                         :func:`_current_rates`).
        state:           Optional guide state of the incremental mode,
                         updated after every move.
        return_rates:    Also return the rates and the guide state of the
                         final configuration.

    Returns:
        The evolved ``Walker``; with ``return_rates`` the tuple
        ``(walker, rates, state)``.
    """
    def sub_step(config, log_weight, state_, rates_, new_configs, step_idx):
        cdf = lam - jnp.cumsum(rates_)
        b_x = cdf[-1]
        k1 = jax.random.fold_in(rng_key, step_idx)
        idx = sample_index(cdf, (-rates_).at[0].add(lam), k1)
        state_ = _state_after_move(hamiltonian, guide, cache, state_, config, idx)
        # If get_rates already produced the M_total successor configs,
        # reuse them instead of recomputing via generate_state.  This branch is
        # Python-time (None vs array is a static distinction), so the JIT'd
        # body only contains the chosen path.
        if new_configs is None:
            config = hamiltonian.generate_state(config, idx)
        else:
            config = new_configs[idx]
        return config, log_weight + jnp.log(b_x), state_

    def body(step_idx, carry):
        config, log_weight, state_ = carry
        rates_, new_configs = transition_rates(config, hamiltonian, guide,
                                               cache=cache, state=state_)
        return sub_step(config, log_weight, state_, rates_, new_configs, step_idx)

    plain = rates is None and state is None and not return_rates
    if plain:
        # Nothing carried: the uncarried loop (see cont_time).
        def body0(step_idx, carry):
            config, log_weight = carry
            rates_, new_configs = transition_rates(config, hamiltonian, guide, cache=cache)
            config, log_weight, _ = sub_step(config, log_weight, None, rates_,
                                             new_configs, step_idx)
            return (config, log_weight)

        final_config, final_log_weight = jax.lax.fori_loop(
            0, n_bra, body0, (walker.config, walker.log_weight))
        rates_f, _ = transition_rates(final_config, hamiltonian, guide, cache=cache)
        final_e_loc = jnp.sum(rates_f)      # e_L(x) = Σ_y rates[y] = λ − b_x
        final_log_weight -= avg_log_weight * n_bra
        return Walker(config=final_config, log_weight=final_log_weight, e_loc=final_e_loc)

    # Something is carried: the first sub-step is peeled off the loop so that
    # it can start from the carried rates; the remaining ones evaluate their own.
    rates0, new_configs0 = _current_rates(walker.config, hamiltonian, guide,
                                          cache, rates, state)
    init = sub_step(walker.config, walker.log_weight, state, rates0, new_configs0, 0)
    final_config, final_log_weight, state_f = jax.lax.fori_loop(1, n_bra, body, init)
    rates_f, _ = transition_rates(final_config, hamiltonian, guide, cache=cache,
                                  state=state_f)
    final_e_loc = jnp.sum(rates_f)          # e_L(x) = Σ_y rates[y] = λ − b_x
    final_log_weight -= avg_log_weight * n_bra
    return _with_rates(Walker(config=final_config, log_weight=final_log_weight,
                              e_loc=final_e_loc), rates_f, state_f, return_rates)


def evolve_walker_multi_step(walker, hamiltonian, n_bra, guide, rng_key,
                             avg_log_weight, lam, cache=None, rates=None,
                             state=None, return_rates=False):
    """Evolve one walker for n_bra sub-steps, batching diagonal holds.

    Samples geometrically distributed hold times from the diagonal-stay
    probability p_d, accumulating log(b_x) * k into log_weight.  Statistically
    equivalent to ``single_step`` but batches k consecutive diagonal moves
    into a single geometric draw (Becca–Sorella §8.4), so ``get_rates`` is
    only evaluated at actual configuration changes.

    Args:
        walker:          Single-walker pytree (config, log_weight, e_loc).
        hamiltonian:     Hamiltonian object.
        n_bra:           Number of sub-steps (the integer ``step_size`` of
                         :func:`evolve_walkers`).
        guide:           Optional guiding wavefunction.
        rng_key:         JAX PRNGKey; hold ``i`` draws from
                         ``fold_in(rng_key, 2 i)`` and ``fold_in(rng_key, 2 i + 1)``.
        avg_log_weight:  Normalisation shift per sub-step.
        lam:             The GFMC shift λ.
        cache:           Optional ``guide.build_cache(hamiltonian)`` result.
        rates:           Optional guide-transformed rates of ``walker.config``
                         when the caller already has them (see
                         :func:`_current_rates`).
        state:           Optional guide state of the incremental mode,
                         updated after every completed hold.
        return_rates:    Also return the rates and the guide state of the
                         final configuration.

    Returns:
        The evolved ``Walker``; with ``return_rates`` the tuple
        ``(walker, rates, state)``.
    """
    # A purely diagonal Hamiltonian has no geometric transition to sample.
    # The configuration stays fixed and its branch weight is accumulated for
    # the complete interval.
    if hamiltonian.M_total == 1:
        rates, _ = _current_rates(walker.config, hamiltonian, guide, cache, rates, state)
        b_x = lam - rates[0]
        return _with_rates(Walker(
            config=walker.config,
            log_weight=(walker.log_weight + n_bra * jnp.log(b_x)
                        - avg_log_weight * n_bra),
            e_loc=rates[0],
        ), rates, state, return_rates)

    # while_loop is the intentional choice here (not fori_loop): expected
    # iterations are (1 - p_d) * n_bra, which is the whole point of batching
    # the diagonal holds.  A fori_loop would always run n_bra iterations
    # and compute get_rates every time, discarding the speedup.
    # iter_count tracks while_loop iterations; bounded by n_bra (steps_used += k ≥ 1)
    # so fold_in(rng_key, 2*iter) uses at most 2*n_bra distinct indices; no rng in carry
    def cond(carry):
        config, log_weight, steps_used, iter_count, state_ = carry
        return steps_used < n_bra

    def hold(config, log_weight, steps_used, state_, rates, new_configs, k1, k2):
        """One geometric hold from ``config`` with its ``rates``: the
        weight update and the hop, truncated at the end of the interval."""
        cdf = jnp.cumsum(-rates[1:])
        R_x = cdf[-1]
        b_x = lam - rates[0] + R_x
        state_idx = jnp.where(R_x > 0, sample_index(cdf, -rates[1:], k1), 0)
        # Reuse the precomputed successor configs when get_rates
        # already returned them (static None-vs-array distinction).
        if new_configs is None:
            new_config = hamiltonian.generate_state(config, state_idx + 1)
        else:
            new_config = new_configs[state_idx + 1]
        psi = jax.random.uniform(k2)
        # log1p(-psi) = log(1-psi) evaluated without cancellation near psi=0.
        steps_remaining = n_bra - steps_used
        # jnp.where evaluates both branches, so make the unused absorbing-row
        # denominator finite instead of masking an inf-to-integer conversion.
        # Form log(1 - R/b) directly: p_d may round to one even for R > 0.
        # Bound in floating point BEFORE conversion; long holds can exceed
        # int32 even when their logarithm is finite.
        hop_probability = R_x / b_x
        can_hop = hop_probability > 0
        log_p_d = jnp.where(can_hop, jnp.log1p(-hop_probability), -1.0)
        hold = jnp.floor(jnp.log1p(-psi) / log_p_d) + 1.0
        k_raw_sampled = jnp.minimum(hold, steps_remaining + 1).astype(jnp.int32)
        # With R_x == 0 the current configuration is absorbing.  Consume the
        # remaining interval without applying an off-diagonal transition.
        k_raw = jnp.where(can_hop, k_raw_sampled, steps_remaining + 1)
        k = jnp.minimum(k_raw, steps_remaining)
        new_log_weight = log_weight + jnp.log(b_x) * k
        new_steps = steps_used + k
        completed_hold = k_raw <= steps_remaining
        final_config = jnp.where(completed_hold, new_config, config)
        if state_ is not None:
            new_state = _state_after_move(hamiltonian, guide, cache, state_,
                                          config, state_idx + 1)
            state_ = jax.tree_util.tree_map(
                lambda new, old: jnp.where(completed_hold, new, old), new_state, state_)
        return final_config, new_log_weight, new_steps, state_

    def body(carry):
        config, log_weight, steps_used, iter_count, state_ = carry
        k1 = jax.random.fold_in(rng_key, 2 * iter_count)
        k2 = jax.random.fold_in(rng_key, 2 * iter_count + 1)
        rates_, new_configs = transition_rates(config, hamiltonian, guide,
                                               cache=cache, state=state_)
        config, log_weight, steps_used, state_ = hold(
            config, log_weight, steps_used, state_, rates_, new_configs, k1, k2)
        return (config, log_weight, steps_used, iter_count + 1, state_)

    plain = rates is None and state is None and not return_rates
    if plain:
        # Nothing carried: the uncarried loop (see cont_time).
        def cond0(carry):
            return carry[2] < n_bra

        def body0(carry):
            config, log_weight, steps_used, iter_count = carry
            k1 = jax.random.fold_in(rng_key, 2 * iter_count)
            k2 = jax.random.fold_in(rng_key, 2 * iter_count + 1)
            rates_, new_configs = transition_rates(config, hamiltonian, guide, cache=cache)
            config, log_weight, steps_used, _ = hold(
                config, log_weight, steps_used, None, rates_, new_configs, k1, k2)
            return (config, log_weight, steps_used, iter_count + 1)

        init = (walker.config, walker.log_weight, jnp.int32(0), jnp.int32(0))
        final_config, final_log_weight, _, _ = jax.lax.while_loop(cond0, body0, init)
        rates_f, _ = transition_rates(final_config, hamiltonian, guide, cache=cache)
        final_e_loc = jnp.sum(rates_f)      # e_L(x) = Σ_y rates[y] = λ − b_x
        final_log_weight -= avg_log_weight * n_bra
        return Walker(config=final_config, log_weight=final_log_weight, e_loc=final_e_loc)

    # Something is carried: the first hold is peeled off the loop so that it
    # can start from the carried rates (with the keys iteration 0 of the loop
    # had).
    rates0, new_configs0 = _current_rates(walker.config, hamiltonian, guide,
                                          cache, rates, state)
    config, log_weight, steps_used, state = hold(
        walker.config, walker.log_weight, jnp.int32(0), state, rates0, new_configs0,
        jax.random.fold_in(rng_key, 0), jax.random.fold_in(rng_key, 1))
    init = (config, log_weight, steps_used, jnp.int32(1), state)
    final_config, final_log_weight, _, _, state_f = jax.lax.while_loop(cond, body, init)
    # The last hold may complete exactly at the end of the interval and move
    # the walker, so the rates of the final configuration are evaluated here.
    rates_f, _ = transition_rates(final_config, hamiltonian, guide, cache=cache,
                                  state=state_f)
    final_e_loc = jnp.sum(rates_f)          # e_L(x) = Σ_y rates[y] = λ − b_x
    final_log_weight -= avg_log_weight * n_bra
    return _with_rates(Walker(config=final_config, log_weight=final_log_weight,
                              e_loc=final_e_loc), rates_f, state_f, return_rates)


def evolve_walker_cont_time(walker, hamiltonian, beta, guide, rng_key,
                            avg_log_weight, lam=None, cache=None, rates=None,
                            state=None, return_rates=False):
    """Evolve one walker for imaginary time ``beta`` using continuous-time GFMC.

    This is the $\\lambda \\to \\infty$ limit of
    :func:`evolve_walker_multi_step`: with $M = \\lambda\\,\\beta$
    elementary substeps, the discrete propagator
    $(\\lambda \\mathbb{1} - H)^M = \\lambda^M (\\mathbb{1} - H/\\lambda)^{\\lambda\\beta}$
    converges to the exact imaginary-time propagator $\\exp(-\\beta H)$.
    Hold times in a single configuration
    become exponentially distributed with rate
    $R(x) = \\sum_{y\\neq x} |H_{yx}|\\,\\psi_T(y)/\\psi_T(x)$, and over a
    hold of length $\\tau$ the walker weight evolves as
    $w \\to w\\,e^{-e_L(x)\\,\\tau}$, where
    $e_L(x) = \\sum_y H_{yx}\\,\\psi_T(y)/\\psi_T(x)$ is the local energy.

    Args:
        walker:          Single-walker pytree (config, log_weight, e_loc).
        hamiltonian:     Hamiltonian object.
        beta:            Imaginary-time length of the Markov step (the
                         ``step_size`` of :func:`evolve_walkers`).
        guide:           Optional guiding wavefunction.
        rng_key:         JAX PRNGKey.
        avg_log_weight:  Per-unit-time normalisation shift; subtracted as
                         ``avg_log_weight * beta``.  In this continuous-time
                         path it plays the role of the (negative) average
                         local energy and removes the leading population drift.
        lam:             Unused: continuous-time evolution is independent of
                         $\\lambda$.  Accepted so that the three propagators
                         share one signature.
        cache:           Optional ``guide.build_cache(hamiltonian)`` result.
        rates:           Optional guide-transformed rates of ``walker.config``
                         when the caller already has them (see
                         :func:`_current_rates`).
        state:           Optional guide state of the incremental mode,
                         updated after every completed hold.
        return_rates:    Also return the rates and the guide state of the
                         final configuration.

    Returns:
        The evolved single-walker ``Walker``; with ``return_rates`` the
        tuple ``(walker, rates, state)``.
    """
    del lam  # the continuous-time propagator is independent of the shift

    if hamiltonian.M_total == 1:
        rates, _ = _current_rates(walker.config, hamiltonian, guide, cache, rates, state)
        e_L = rates[0]
        return _with_rates(Walker(
            config=walker.config,
            log_weight=(walker.log_weight - e_L * beta
                        - avg_log_weight * beta),
            e_loc=e_L,
        ), rates, state, return_rates)

    # while_loop: expected number of body iterations is ~ R̄ · β, where R̄ is
    # the typical off-diagonal escape rate.  iter_count carries an int so we
    # can fold_in distinct PRNG sub-keys; the actual termination criterion is
    # time_used >= beta.
    #
    # The carry also holds e_L = sum(rates) of the CURRENT configuration.  The
    # loop only ends on a truncated hold (tau_raw >= time_remaining, see the
    # strict inequality below), which leaves the configuration unchanged, so
    # the rates evaluated in the last iteration are those of the final
    # configuration and e_loc comes out of the loop without a further
    # transition_rates call, which is the dominant cost of a step.
    def cond(carry):
        config, log_weight, time_used, iter_count, rates_, state_ = carry
        return time_used < beta

    def hold(config, log_weight, time_used, state_, rates, new_configs, k1, k2):
        """One exponential hold from ``config`` with its ``rates``: the
        weight update and the hop, truncated at the end of the interval."""
        cdf = jnp.cumsum(-rates[1:])
        R_x = cdf[-1]                          # total off-diagonal escape rate
        e_L = rates[0] - R_x             # branch weight (Becca–Sorella Eq. 8.55)
        state_idx = jnp.where(R_x > 0, sample_index(cdf, -rates[1:], k1), 0)
        # Reuse the precomputed successor configs when get_rates
        # already returned them (static None-vs-array distinction).
        if new_configs is None:
            new_config = hamiltonian.generate_state(config, state_idx + 1)
        else:
            new_config = new_configs[state_idx + 1]
        psi = jax.random.uniform(k2)
        # Exponential hold time τ ~ Exp(R_x): τ = -log(1-ψ)/R_x.
        # A configuration can have R_x == 0 even when other rows of the
        # Hamiltonian have off-diagonal couplings.  Such a row is absorbing,
        # so its hold lasts for the remainder of this interval.
        tau_raw = jnp.where(
            R_x > 0, -jnp.log1p(-psi) / R_x, jnp.inf)
        time_remaining = beta - time_used
        tau = jnp.minimum(tau_raw, time_remaining)
        new_log_weight = log_weight - e_L * tau
        new_time = time_used + tau
        # STRICT: a hold ending exactly on the boundary (a measure-zero
        # event) counts as truncated, so the loop always exits with the
        # configuration whose rates were just evaluated.
        completed_hold = tau_raw < time_remaining
        final_config = jnp.where(completed_hold, new_config, config)
        if state_ is not None:
            new_state = _state_after_move(hamiltonian, guide, cache, state_,
                                          config, state_idx + 1)
            state_ = jax.tree_util.tree_map(
                lambda new, old: jnp.where(completed_hold, new, old), new_state, state_)
        return final_config, new_log_weight, new_time, state_

    def body(carry):
        config, log_weight, time_used, iter_count, _, state_ = carry
        k1 = jax.random.fold_in(rng_key, 2 * iter_count)
        k2 = jax.random.fold_in(rng_key, 2 * iter_count + 1)
        rates_, new_configs = transition_rates(config, hamiltonian, guide,
                                               cache=cache, state=state_)
        config, log_weight, time_used, state_ = hold(
            config, log_weight, time_used, state_, rates_, new_configs, k1, k2)
        return (config, log_weight, time_used, iter_count + 1, rates_, state_)

    plain = rates is None and state is None and not return_rates
    if plain:
        # Nothing carried: every iteration evaluates its own rates and the
        # local energy of the final configuration is taken from the last one
        # (which left the configuration unchanged), compiled in one piece.
        # The carried path below peels the first hold off the loop, which
        # changes XLA's fusion; this path keeps the loop whole.
        def cond0(carry):
            return carry[2] < beta

        def body0(carry):
            config, log_weight, time_used, iter_count, _ = carry
            k1 = jax.random.fold_in(rng_key, 2 * iter_count)
            k2 = jax.random.fold_in(rng_key, 2 * iter_count + 1)
            rates_, new_configs = transition_rates(config, hamiltonian, guide, cache=cache)
            config, log_weight, time_used, _ = hold(
                config, log_weight, time_used, None, rates_, new_configs, k1, k2)
            return (config, log_weight, time_used, iter_count + 1, jnp.sum(rates_))

        zero = jnp.array(0.0, dtype=walker.log_weight.dtype)
        init = (walker.config, walker.log_weight, zero, jnp.int32(0), zero)
        final_config, final_log_weight, _, _, final_e_loc = jax.lax.while_loop(
            cond0, body0, init)
        final_log_weight -= avg_log_weight * beta
        return Walker(config=final_config, log_weight=final_log_weight, e_loc=final_e_loc)

    # Something is carried: the first hold is peeled off the loop so that it
    # can start from the carried rates (with the keys iteration 0 of the loop
    # had).  The loop carries the rates of the current configuration and only
    # ends on a truncated hold, which leaves the configuration unchanged, so
    # the rates it hands back are those of the final configuration: e_loc
    # comes out without a further transition_rates call, and the next step
    # can start from them.
    rates0, new_configs0 = _current_rates(walker.config, hamiltonian, guide,
                                          cache, rates, state)
    zero = jnp.array(0.0, dtype=walker.log_weight.dtype)
    config, log_weight, time_used, state = hold(
        walker.config, walker.log_weight, zero, state, rates0, new_configs0,
        jax.random.fold_in(rng_key, 0), jax.random.fold_in(rng_key, 1))
    init = (config, log_weight, time_used, jnp.int32(1), rates0, state)
    final_config, final_log_weight, _, _, rates_f, state_f = jax.lax.while_loop(
        cond, body, init)
    final_e_loc = jnp.sum(rates_f)          # e_L(x) = Σ_y rates[y]; λ-independent
    final_log_weight -= avg_log_weight * beta
    return _with_rates(Walker(config=final_config, log_weight=final_log_weight,
                              e_loc=final_e_loc), rates_f, state_f, return_rates)


def require_lam(method: str, lam: Optional[float]) -> Optional[float]:
    """Return ``lam`` validated for ``method``: a float for the discrete
    methods, which need the shift λ, and ``None`` for ``cont_time``, which
    does not use it.  Raises ``ValueError`` when a discrete method is called
    without a shift."""
    if method == "cont_time":
        return None
    if method in ("single_step", "multi_step"):
        if lam is None:
            raise ValueError(
                f"method={method!r} needs the GFMC shift: pass lam=<float> "
                f"with lam >= max_x H_xx over the reachable configurations "
                f"(cont_time is the only method that does without it).")
        value = float(lam)
        if not math.isfinite(value):
            raise ValueError(f"lam must be finite; got lam={lam!r}")
        return value
    raise ValueError(f"Unknown method: {method!r}")


def require_step_size(method, step_size):
    """Raise unless ``step_size`` is valid for ``method``.

    Every method needs ``step_size > 0``; ``single_step`` and ``multi_step``
    also need an integer (the number of branching sub-steps).  The drivers call
    this before the warm-up, so an invalid step fails before anything is walked.
    """
    if not math.isfinite(float(step_size)):
        raise ValueError(f"step_size must be finite; got step_size={step_size!r}")
    if not (float(step_size) > 0):
        raise ValueError(f"step_size must be > 0; got step_size={step_size!r}")
    if method != "cont_time" and float(int(step_size)) != float(step_size):
        raise ValueError(
            f"method={method!r} requires an integer step_size; "
            f"got step_size={step_size!r}")


def evolve_walkers(walkers: Walker, hamiltonian: Operator,
                   step_size: Union[int, float], guide: Optional[Guide],
                   method: str, rng_key: jax.Array, avg_log_weight: float,
                   lam: Optional[float] = None, cache: Any = None,
                   rates: Optional[jax.Array] = None, state: Any = None,
                   return_rates: bool = False) -> Any:
    """Evolve the full walker batch (vmapped over walkers).

    Args:
        walkers:         Walker batch, shape (N_walkers, ...)
        hamiltonian:     Hamiltonian
        step_size:       Per-Markov-step size.  Interpreted as the number of
                         imaginary-time sub-steps for the discrete methods
                         ``single_step`` and ``multi_step`` (must be
                         a positive integer-valued float), and as the
                         imaginary-time length $\\beta$ for the continuous
                         method ``cont_time`` (any positive float).
        guide:           guiding wavefunction or None
        method:          ``"single_step"``, ``"multi_step"``, or
                         ``"cont_time"``.
        rng_key:         JAX PRNGKey
        avg_log_weight:  normalisation shift subtracted from log_weight
                         (per substep for the discrete methods, per unit
                         imaginary time for ``cont_time``).
        lam:             GFMC shift λ.  Required by the discrete methods,
                         unused by ``cont_time`` (the default ``None``).
        cache:           Optional ``guide.build_cache(hamiltonian)`` result,
                         forwarded to the guide's ratio routine.
        rates:           Optional ``(N_walkers, M_total)`` guide-transformed
                         rates of the current configurations, as returned by
                         the previous step with ``return_rates``; the first
                         rate evaluation of every walker is then skipped.
        state:           Optional per-walker guide state of the incremental
                         mode (see :class:`pygfmc.Guide`), updated after
                         every move.
        return_rates:    Also return the rates and the guide state of the
                         final configurations.

    Returns:
        Updated Walker batch with new config, log_weight, e_loc; with
        ``return_rates`` the tuple ``(walkers, rates, state)``.
    """
    # Python-level validation: must happen outside @jax.jit to raise.
    require_step_size(method, step_size)
    lam = require_lam(method, lam)
    if method == "cont_time":
        n_bra = 0                              # unused by cont_time
        beta  = float(step_size)
    else:
        n_bra = int(step_size)
        beta  = 0.0                            # unused by the discrete methods
    return _evolve_walkers_jit(walkers, hamiltonian, lam, guide, n_bra, beta, method,
                               rng_key, avg_log_weight, cache, rates, state,
                               return_rates)


@partial(jax.jit, static_argnames=["lam", "method", "return_rates"])
def _evolve_walkers_jit(walkers, hamiltonian, lam, guide, n_bra, beta, method,
                        rng_key, avg_log_weight, cache, rates=None, state=None,
                        return_rates=False):
    """JITted body of :func:`evolve_walkers` (no validation)."""
    N_walkers = walkers.config.shape[0]
    if method == "single_step":
        def one(walker, key, r, st):
            return evolve_walker_single_step(
                walker, hamiltonian, n_bra, guide, key, avg_log_weight, lam, cache,
                rates=r, state=st, return_rates=return_rates)
    elif method == "multi_step":
        def one(walker, key, r, st):
            return evolve_walker_multi_step(
                walker, hamiltonian, n_bra, guide, key, avg_log_weight, lam, cache,
                rates=r, state=st, return_rates=return_rates)
    elif method == "cont_time":
        def one(walker, key, r, st):
            return evolve_walker_cont_time(
                walker, hamiltonian, beta, guide, key, avg_log_weight, lam, cache,
                rates=r, state=st, return_rates=return_rates)
    else:
        raise ValueError(f"Unknown method: {method}")
    keys = jax.random.split(rng_key, N_walkers)
    # None (nothing carried) is an empty pytree, so it maps along no axis.
    return jax.vmap(one, in_axes=(0, 0, None if rates is None else 0,
                                  None if state is None else 0))(walkers, keys, rates, state)


# ── forward-walking operator estimator ────────────────────────────────────────


def apply_operator_one(config, key, op, guide, cache_op):
    """Apply the measurement operator ``op`` to one walker: ``(new_config,
    total_weight)`` with ``new_config`` drawn from the guide-transformed
    rates of ``op`` at ``config`` (``key`` picks the term) and
    ``total_weight`` their sum, the operator value of the walker.  A purely
    diagonal operator (``M_total == 1``) keeps the configuration and returns
    its diagonal element.  Shared by :func:`forward_walk_operator` and
    :func:`pygfmc.engine.multi_gpu.forward_walk_operator_global`."""
    M_total = op.M_total   # static Python int: total rates (diagonal + off-diagonal elements)
    rates, new_configs = transition_rates(
        config, op, guide,
        return_configs=False,
        cache=cache_op,
    )
    if M_total == 1:
        return config, rates[0]

    cdf = jnp.cumsum(rates)
    total_weight = cdf[-1]
    # total_weight == 0 means no transition (diagonal or off-diagonal) has
    # any weight for this config under this operator, e.g. a single-bond
    # measurement operator whose bond sites are both empty or both at
    # max occupation, so neither hop direction is allowed. searchsorted
    # on an all-zero cdf for u=0 returns len(cdf) (one past the last
    # valid index), and indexing new_configs there does not safely clip:
    # it returns a corrupted config (observed: a site occupation of -1),
    # which then poisons the shadow chain with NaN. Guard explicitly:
    # stay at the current (diagonal) config with zero weight, mirroring
    # the M_total==1 branch's "no off-diagonal channel" semantics. Adds
    # one scalar compare + jnp.where; no extra rate/Hamiltonian work.
    is_zero = total_weight <= 0
    state_idx = jnp.where(is_zero, 0, sample_index(cdf, rates, key))
    if new_configs is None:
        new_config = op.generate_state(config, state_idx)
    else:
        new_config = new_configs[state_idx]
    new_config = jnp.where(is_zero, config, new_config)
    total_weight = jnp.where(is_zero, 0.0, total_weight)

    return new_config, total_weight


def forward_walk_operator(walkers: Walker, op: Operator, h_main: Operator,
                          step_size: Union[int, float], guide: Optional[Guide],
                          method: str, rng_key: jax.Array, avg_log_weight: float,
                          m_array: jax.Array, lam: Optional[float] = None,
                          cache_h: Any = None, cache_op: Any = None,
                          shadow_key: Optional[jax.Array] = None,
                          op_batch_size: Optional[int] = None,
                          ) -> tuple[jax.Array, jax.Array]:
    """Forward-walking pure estimator for a single (off-diagonal) operator.

    ``op`` may be any object implementing the operator protocol
    (:class:`pygfmc.Operator`: ``M_total``, ``get_rates``, and
    ``generate_state`` when ``M_total > 1``), e.g. SpinHalf,
    PlaquetteModel, or ToricCode instances used as observables, whose
    guide-transformed rates are all non-negative.  ``run_gfmc`` calls it at
    every measured production step (every ``off_diag_stride``-th step), after
    reconfiguration.  The drivers require even ``p`` (``m = p // 2`` shadow
    steps per window) for every operator with ``M_total > 1``; a purely
    diagonal operator (``M_total == 1``) may use odd ``p``, and at ``m = 0``
    its value is the plain mean of the diagonal elements (``log_norm = 0``).

      1. Apply op to the current walker ensemble:
           - each walker x_i picks a new config x'_i proportional to the
             operator transition weights w(x_i, x')
           - vals[i] = Σ_{x'} w(x_i, x')   (total operator weight, ≥ 0)
      2. Propagate the operator-applied ensemble along a shadow chain for
         max(m_array) steps with h_main.  After every step the shadow
         ensemble is reconfigured, vals is remapped through the parent map,
         and the log of the reconfiguration weight G_t is accumulated.
      3. Record, at each m in m_array, mean(vals) and the accumulated
         log weight product.

    Args:
        walkers:      post-reconfiguration Walker batch (uniform weights)
        op:           operator to measure (see above)
        h_main:       main Hamiltonian (governs forward evolution)
        step_size:    per-Markov-step size (integer sub-step count for the
                      discrete methods, imaginary time beta for cont_time);
                      forwarded to evolve_walkers.
        guide:        guiding wavefunction or None
        method:       "single_step", "multi_step", or "cont_time"
                      (forwarded to evolve_walkers for the shadow chain)
        rng_key:      JAX PRNGKey for the operator application (and for the
                      shadow chain when ``shadow_key`` is None)
        avg_log_weight:  log branch-weight normalisation scalar
        m_array:      list of shadow-chain depths (one per p value,
                      m = p // 2; m = 0 reads the un-propagated ensemble)
        lam:          GFMC shift for h_main; required by the discrete
                      methods, unused by ``cont_time``
        cache_h:      optional ``guide.build_cache(h_main)`` result
        cache_op:     optional ``guide.build_cache(op)`` result
        shadow_key:   optional PRNG key to start the shadow chain from.
                      ``run_gfmc`` passes the main chain's NEXT production
                      key: the shadow runs the same key recurrence, so from
                      the (near-identical) operator-applied ensemble it
                      replays the main chain's future draws, and the
                      population-level weight noise largely cancels between
                      the estimator's numerator and denominator (correlated
                      sampling).  Keys are state-independent, so expectations
                      are unchanged.  None derives an independent stream from
                      ``rng_key`` (same expectation, larger variance at deep
                      windows).
        op_batch_size:  optional walker-chunk size for the OPERATOR
                      APPLICATION (step 1).  None (default) applies the
                      operator to all walkers at once.  An int splits the
                      walker batch into chunks of that size via
                      ``jax.lax.map``, cutting the peak (N_walkers, n_off,
                      max_aff) guide intermediate by op_batch_size/N_walkers.
                      **Bit-identical**: the per-walker work and its PRNG key
                      are unchanged, so no float reassociation is introduced.
                      Must divide N_walkers.  Use it when a wide operator
                      (e.g. a site sum with many terms) would otherwise not
                      fit in memory; it costs sequential passes.

    Returns:
        (val, log_norm) : pair of jnp.ndarray, each shape (len(m_array),),
        ordered by sorted m.

            val[j]      = mean(vals after m_array[j] shadow steps), the
                          operator measurement alone, linear and SIGNED.
            log_norm[j] = log of the shadow chain's accumulated weight
                          product after m_array[j] steps.  It drifts roughly
                          linearly with m and can reach -100 and beyond at
                          deep windows, which is why it is kept in log space.

        The two channels are kept separate so that the large, compensating
        factors of the estimator cancel in the exponent before any exp()
        (a single linear product underflows float32 at deep windows), while
        the sign lives only in ``val`` (a diagonal operator routed through
        this pipeline has a signed rates[0]).  ``_estimate_observables``
        consumes the pair at measurement step k with m = p/2 as

            numerator term    = G_main^{-m} · G_aux^{+m} · val
            denominator term  = G_main^{-m} · G_main^{+m}

        i.e. ``log_norm`` = log G_aux^{+m} is a numerator FACTOR: the ket
        half of the window is supplied by the shadow chain because that half
        carries the operator, while the denominator is main chain only.
        G_aux does not cancel; its ratio to G_main^{+m} is what converts the
        mixed estimator into the pure one.
    """
    N_walkers = walkers.config.shape[0]
    if op_batch_size is not None and N_walkers % op_batch_size:
        raise ValueError(
            f"op_batch_size={op_batch_size} must divide N_walkers={N_walkers} "
            f"(a remainder would compile a second shape for the tail; keep "
            f"one shape so throughput and memory stay predictable)")
    dtype = walkers.log_weight.dtype

    def apply_op_one(config, key):
        return apply_operator_one(config, key, op, guide, cache_op)

    # Keys derive from the ABSOLUTE walker index and are materialised as
    # data BEFORE the map, which is exactly what makes op_batch_size
    # bit-identical: every walker keeps its own key whatever the batching.
    # Do not turn this into a per-chunk key derivation -- that silently
    # breaks the equivalence.
    apply_keys = jax.vmap(
        lambda i: jax.random.fold_in(rng_key, i)
    )(jnp.arange(N_walkers))
    if op_batch_size is None or op_batch_size >= N_walkers:
        new_configs, vals = jax.vmap(apply_op_one)(walkers.config, apply_keys)
    else:
        # Chunk the OPERATOR APPLICATION over walkers.  apply_op_one is
        # per-walker independent (no cross-walker term), so this is exact:
        # each walker's rate sum is the same tree reduction as before, only
        # fewer of them are in flight at once.  Peak intermediate scales as
        # op_batch_size/N_walkers.
        #
        # WHY THIS EXISTS.  A guide's smart update typically builds a
        # rectangular (n_off, max_aff) intermediate per walker before
        # reducing it (max_aff = clusters affected by one transition), so the
        # operator-application peak is (N_walkers, n_off, max_aff).  For a
        # wide operator (e.g. an N^2 pair sum) that can exceed device memory
        # by orders of magnitude.  Chunking is what makes such an operator
        # affordable; it does NOT touch the shadow chain, whose cost is
        # n_off-independent.
        new_configs, vals = jax.lax.map(
            lambda ck: apply_op_one(ck[0], ck[1]),
            (walkers.config, apply_keys),
            batch_size=op_batch_size,
        )

    op_walkers = Walker(
        config=new_configs,
        log_weight=jnp.zeros(N_walkers, dtype=dtype),
        e_loc=jnp.zeros(N_walkers, dtype=dtype),
    )

    sorted_m = sorted(m_array)
    m_max    = sorted_m[-1]

    def step_fn(carry, _):
        w, v, log_wm, key, (r, st) = carry
        # THE SAME RECURRENCE AS THE MAIN CHAIN: key_{t+1}, sub_t = split(key_t)
        # and the sub-keys are folds of sub_t.  Started from the main chain's
        # next key, the shadow therefore reproduces the main chain's future
        # draws step for step -- correlated sampling, with no step index and
        # no lookup anywhere.
        key, rng = jax.random.split(key)
        # sub-key derivation MUST match run_gfmc's main production loop
        # (fold_in 0 = evolve, 1 = reconfigure): started from the main
        # chain's next key, the shadow chain then replays the main chain's
        # future draws exactly on identical ensembles -- the basis of the
        # correlated-sampling estimator.
        k1 = jax.random.fold_in(rng, 0)
        k2 = jax.random.fold_in(rng, 1)
        w, (r, st) = evolve_carried(w, (r, st), h_main, step_size, guide, method,
                                    k1, avg_log_weight, lam=lam, cache=cache_h)
        w, log_G_t, _, indices = reconfiguration(w, k2)
        v = v[indices]
        r, st = gather_carried((r, st), indices)
        log_wm = log_wm + log_G_t
        # Return the accumulator in LOG space.  log_wm drifts roughly linearly
        # with the shadow depth (the chain starts from the OPERATOR-applied
        # ensemble, whose weight statistics differ from the main chain's
        # equilibrium, so no avg_log_weight choice centres it); exponentiating
        # here would underflow float32 at deep windows.  The drift is real but
        # harmless: it cancels against the main chain's weight ratio inside
        # _estimate_observables, and keeping both factors in the exponent lets
        # that cancellation happen BEFORE exponentiation.
        # The two channels are kept SEPARATE rather than combined into one
        # log: mean(v) is O(1) but SIGNED (a diagonal operator measured through
        # this pipeline carries its diagonal element in rates[0], which may be
        # negative), so log(mean(v)) would be NaN for it.  Only log_wm needs
        # the exponent range; only mean(v) needs the sign.
        return (w, v, log_wm, key, (r, st)), (jnp.mean(v), log_wm)

    # shadow_key = the main chain's NEXT key.  Absent it, an independent
    # stream (same expectation, larger deep-window variance).
    start_key = (jax.random.fold_in(rng_key, N_walkers)
                 if shadow_key is None else shadow_key)
    log_weight_multi = jnp.zeros((), dtype=dtype)
    # The shadow chain carries the rates of its walkers across the steps as
    # the main chain does; the first step's evaluation is done here.
    carried0 = initial_carried(op_walkers, h_main, guide, cache_h,
                               with_rates=rates_fit_budget(
                                   op_walkers.config.shape[0], h_main,
                                   op_walkers.log_weight.dtype, _carry_rates_bytes()))
    _, (per_step_fw, per_step_norm) = jax.lax.scan(
        step_fn, (op_walkers, vals, log_weight_multi, start_key, carried0),
        None, length=m_max)

    # Row m holds the value after m shadow steps.  Row 0 is the un-propagated
    # ensemble: the plain mean of the operator values at the measurement
    # step with no accumulated weight, which is what a purely diagonal
    # operator at an odd p (m = p // 2 = 0) measures.
    per_step_fw = jnp.concatenate([jnp.mean(vals)[None], per_step_fw])
    per_step_norm = jnp.concatenate([log_weight_multi[None], per_step_norm])
    pick = jnp.array(sorted_m, dtype=jnp.int32)
    return per_step_fw[pick], per_step_norm[pick]
