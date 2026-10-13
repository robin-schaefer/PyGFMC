"""The interfaces of a Hamiltonian, a measurement operator and a guiding
wavefunction.

They are :class:`typing.Protocol` classes: a class satisfies one by having
the listed attributes, not by inheriting from it, so the built-in
Hamiltonians (NamedTuples), the built-in guides and any user class qualify
as they are.  The classes exist to document the interfaces in one place and
to name them in type hints; the engine enforces them at the start of a run
with :func:`pygfmc.engine.evolution.require_operator_protocol` and
:func:`pygfmc.engine.evolution.require_guide_protocol`, which also know the
two conditional members (``generate_state`` only for ``M_total > 1``, and
one of the three guide ratio methods).

``config`` is one basis configuration, an array of shape ``(N_sites,)``.
Every method is called per walker and must be traceable by JAX (it runs
under ``jax.vmap`` inside a jitted scan).
"""
from typing import Any, Optional, Protocol

import jax

__all__ = ["Operator", "DiagonalOperator", "Guide"]


class Operator(Protocol):
    """The rate protocol of a Hamiltonian or of an off-diagonal observable.

    The operator is a sum ``H = sum_i h_i`` of ``M_total`` terms, each of
    which produces at most one new basis state from a given one.  Term 0 is
    the diagonal (all diagonal operators are absorbed into it).  An operator
    passed as ``hamiltonian`` or in ``off_diag_operators`` must be a JAX
    pytree, because it is passed into jitted functions.
    """

    M_total: int
    """Number of terms: one diagonal term plus one per possible transition."""

    def get_rates(self, config: jax.Array, return_configs: bool = False
                  ) -> tuple[jax.Array, Optional[jax.Array]]:
        """Matrix elements of all terms acting on ``config``.

        Returns ``(rates, new_configs)``: ``rates`` has shape ``(M_total,)``
        with the diagonal element first.  ``new_configs`` is the
        ``(M_total, N_sites)`` array of successor states (row 0 is ``config``
        itself) when ``return_configs=True``, else ``None``.
        """
        ...

    def generate_state(self, config: jax.Array, idx: jax.Array) -> jax.Array:
        """The basis state produced by term ``idx`` acting on ``config``.

        Required when ``M_total > 1``.  The propagators use it to avoid
        materialising all successors.
        """
        ...

    def move(self, config: jax.Array, idx: jax.Array
             ) -> tuple[jax.Array, jax.Array]:
        """Optional.  The sites changed by term ``idx`` and their increments,
        ``(sites, deltas)`` of a fixed shape (``(2,)`` each for the built-in
        classes; zero increments for ``idx == 0``), so that
        ``generate_state(config, idx)`` is ``config`` with ``deltas`` added
        at ``sites``.  Needed only by a guide in the incremental mode (see
        :class:`Guide`), which updates its per-walker state from them.
        """
        ...


class DiagonalOperator(Protocol):
    """An operator measured through its diagonal only (``diag_operators``).

    The built-in Hamiltonians provide ``diagonal_weight`` as well, so they
    can be measured as diagonal observables.
    """

    def diagonal_weight(self, config: jax.Array) -> jax.Array:
        """The diagonal element for ``config``: a scalar, or a vector of
        values for a vector-valued observable."""
        ...


class Guide(Protocol):
    """A guiding wavefunction ``psi_T``.

    The engine needs only the ratios ``psi_T(x') / psi_T(x)`` to the
    ``M_total`` successors of the current configuration, so a guide
    implements **at least one** of the three methods below; the engine
    uses the first one it finds, in this order.  ``build_cache`` is
    optional.  A guide that is passed into a run must be a JAX pytree.

    The class declares all four methods so that each is documented once; a
    guide that has only some of them is accepted at run time.
    """

    def log_diff_from_operator(self, config: jax.Array, op: Operator,
                               cache: Any = None, state: Any = None) -> jax.Array:
        """``log psi_T(x'_k) - log psi_T(x)`` for every term ``k`` of ``op``,
        shape ``(M_total,)``, computed from the operator's structure without
        materialising the successors.  The fastest path.  ``state`` is the
        walker's state of the incremental mode (passed as ``state=`` only
        when :attr:`incremental` is set; see :meth:`init_state`)."""
        ...

    def log_diff_from_configs(self, config: jax.Array, new_configs: jax.Array,
                              cache: Any = None) -> jax.Array:
        """The same log-ratios from the materialised successors
        ``new_configs`` of shape ``(M_total, N_sites)``."""
        ...

    def log_psi(self, config: jax.Array, cache: Any = None) -> jax.Array:
        """``log psi_T(config)``, a scalar.  The last fallback: the engine
        then evaluates it on every successor."""
        ...

    def build_cache(self, op: Operator) -> Any:
        """Optional.  Precompute operator-specific data once per run; the
        result is passed as ``cache`` to ``log_diff_from_operator``.  The
        configuration-based fallbacks are called without a cache."""
        ...

    incremental: bool
    """Optional.  True switches on the incremental mode: the engine keeps a
    per-walker state from :meth:`init_state`, updates it with
    :meth:`update_state` after every move (the Hamiltonian must provide
    :meth:`Operator.move`), carries it through reconfiguration, passes it as
    ``state=`` to :meth:`log_diff_from_operator`, and rebuilds it every 100 chain
    steps so that rounding cannot accumulate.  The built-in Jastrow
    guides keep their weighted neighbour sums this way (constructor flag
    ``incremental``).  A run in this mode is not bit-identical to the same
    run without it."""

    def init_state(self, config: jax.Array, cache: Any = None) -> Any:
        """Optional (incremental mode).  The per-walker state of ``config``,
        a pytree of arrays."""
        ...

    def update_state(self, state: Any, config: jax.Array, sites: jax.Array,
                     deltas: jax.Array, cache: Any = None) -> Any:
        """Optional (incremental mode).  ``state`` after the move that adds
        ``deltas`` to ``sites`` of ``config`` (the result of
        :meth:`Operator.move`); ``log_diff_from_operator`` then receives it
        as ``state=``."""
        ...
