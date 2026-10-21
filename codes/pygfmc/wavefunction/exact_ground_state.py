"""ExactGroundState guiding wavefunction.

ψ_T(x) = |ψ_0[idx(x)]|, where ψ_0 is the exact ground state computed by
:func:`pygfmc.hamiltonian.exact_diagonalization`, which needs only the Hamiltonian
protocol, and ``idx`` is the positional basis index built from
``local_hilbert_space``.

When ψ_T = ψ_0, the GFMC local energy collapses to a constant (the ground
state energy) and the Monte Carlo variance vanishes; this guide is the
controlled limit for diagnostics, not a production guide: the stored
amplitude vector has size ``d^N``, so it is only practical for systems
that fit in exact diagonalisation.

The basis index is ``Σ_i digit_i · d^(N-1-i)`` with site 0 most significant,
where digit ``i`` is the position of ``config[i]`` in ``local_hilbert_space``.
That defaults to the Hamiltonian's own ``local_hilbert_space``, which the
pre-implemented classes provide; a class that has none must supply it, along
with ``N`` if it exposes no ``N`` attribute.

Only ``log_diff_from_configs`` is implemented (no ``log_diff_from_operator``,
no ``build_cache``), so GFMC materialises the successor configurations for
this guide.
"""

from __future__ import annotations

import warnings

import h5py
import jax
import jax.numpy as jnp
import numpy as np

# floor on |ψ_0| inside the log, so a node of the exact ground state gives a
# large negative log amplitude instead of -inf (well below float32 epsilon)
_AMP_FLOOR = 1e-30


def _family(hamiltonian):
    return type(hamiltonian).__name__


class ExactGroundState:
    """Guiding wavefunction with amplitudes from the exact ground state.

    Args:
        hamiltonian: any Hamiltonian implementing the protocol (``M_total``,
                     ``get_rates``, ``generate_state``), e.g. a SpinHalf /
                     Spin / BoseHubbard / ToricCode instance.
        log_amp:     optional precomputed ``log |ψ_0|``, shape ``(d^N,)``.
                     If ``None``, ψ_0 is computed with
                     :func:`pygfmc.hamiltonian.exact.exact_diagonalization` and the
                     ground-state energy is stored as ``E0`` (NaN otherwise).
                     The diagonalization returns ONE lowest eigenvector.  When
                     the ground state is degenerate -- e.g. across sectors the
                     Hamiltonian does not connect, such as the flippability
                     sectors of ``PlaquetteModel`` or the ``B_p`` sectors of
                     ``ToricCode`` -- that vector is an arbitrary element of
                     the degenerate subspace and can vanish on part of the
                     basis (an entire sector), where ``|ψ_0|`` is floored at
                     ``1e-30`` before the log.  For a run inside one sector, supply
                     that sector's ground state through ``log_amp`` (e.g.
                     from ``exact_diagonalization(..., number_of_states=k,
                     return_states=True)`` restricted or combined as needed).
                     No degeneracy check is made.
        dtype:       JAX dtype for the stored ``log_amp``.
        local_hilbert_space: Per-site values, e.g. ``(-1, 1)`` for spin-1/2.
                     Defaults to ``hamiltonian.local_hilbert_space``.
        N:           Number of sites.  Defaults to ``hamiltonian.N``.
    """

    def __init__(self, hamiltonian, *, log_amp=None, dtype=jnp.float32,
                 local_hilbert_space=None, N=None):
        family = _family(hamiltonian)
        values = local_hilbert_space
        if values is None:
            values = getattr(hamiltonian, "local_hilbert_space", None)
        if values is None:
            raise ValueError(
                f"ExactGroundState needs the local Hilbert space and {family} "
                "exposes no local_hilbert_space attribute. Pass "
                "local_hilbert_space=(...), e.g. (-1, 1) for spin-1/2.")
        values = tuple(float(v) for v in
                       np.asarray(values, dtype=float).ravel())
        if len(values) < 2:
            raise ValueError(
                f"local_hilbert_space needs at least two values, got "
                f"{list(values)}.")
        d = len(values)

        N_res = N if N is not None else getattr(hamiltonian, "N", None)
        if N_res is None:
            raise ValueError(
                f"ExactGroundState could not determine the number of sites: "
                f"{family} exposes no N attribute. Pass N=... .")
        N_res = int(N_res)

        self.local_hilbert_space = values
        self.N = N_res
        # d**(N-1-i), site 0 most significant; a tuple so it stays hashable
        # pytree aux data
        self._powers = tuple(int(d ** k) for k in range(N_res - 1, -1, -1))

        if log_amp is None:
            from ..hamiltonian.exact import exact_diagonalization
            energies, _, _, states = exact_diagonalization(
                hamiltonian, values, N_res, number_of_states=1,
                return_states=True)
            E0, psi0 = float(energies[0]), states[:, 0]
            psi0 = np.asarray(psi0)
            # Perron–Frobenius for the SPF case: ψ_0 should be either fully
            # ≥ 0 or fully ≤ 0 (global sign flip is arbitrary).  Mixed signs
            # imply something other than the SPF ground state.
            if psi0.min() < -1e-6 and psi0.max() > 1e-6:
                warnings.warn(
                    f"ExactGroundState: ψ_0 has mixed signs "
                    f"(min={psi0.min():.3e}, max={psi0.max():.3e}); this guide "
                    f"assumes a sign-problem-free Hamiltonian (Perron–Frobenius "
                    f"non-negativity).  Using |ψ_0| anyway.",
                    UserWarning,
                )
            self.E0 = float(E0)
            log_amp = np.log(np.maximum(np.abs(psi0), _AMP_FLOOR))
        else:
            self.E0 = float("nan")

        log_amp = jnp.asarray(log_amp, dtype=dtype)
        if log_amp.shape[0] != d ** N_res:
            raise ValueError(
                f"log_amp has {log_amp.shape[0]} entries but the basis has "
                f"{d}**{N_res} = {d ** N_res}. An out-of-range gather is "
                "clamped rather than raised by JAX, so a wrong length would "
                "give a silently wrong guide.")
        self.log_amp     = log_amp
        self.hamiltonian = hamiltonian
        self.dtype       = dtype
        self.M_total     = int(hamiltonian.M_total)

    # ── basis index ──────────────────────────────────────────────────────

    def basis_index(self, config):
        """Index of ``config`` in the amplitude vector."""
        values = jnp.asarray(self.local_hilbert_space, dtype=config.dtype)
        powers = jnp.asarray(self._powers, dtype=jnp.int32)
        digit = jnp.argmin(jnp.abs(config[:, None] - values), axis=-1)
        return jnp.sum(digit.astype(jnp.int32) * powers)

    # ── core API ─────────────────────────────────────────────────────────

    def log_diff_from_configs(self, config, new_configs=None, cache=None):
        """Log amplitude ratios for every transition, shape ``(M_op,)``.

        When ``new_configs`` is given (rows = successor configs of some
        operator, as produced by ``op.get_rates(config, return_configs=True)``),
        the ratios follow that operator's transition layout; this is the path
        GFMC takes (``pygfmc.engine.evolution.transition_rates`` always passes the
        successors for this guide).  When ``new_configs`` is ``None`` the
        successors are generated from the bound Hamiltonian:
        ``x_k_new = hamiltonian.generate_state(x, k)``, and element 0 is pinned
        to exactly ``0.0`` (diagonal, x unchanged).

        For Hamiltonians with bounded local Hilbert space (e.g. Spin S = 1,
        BoseHubbard) successor configs may be out of range for transitions
        whose rate is zero (e.g. S+ acting on m = +S).  Those produce
        out-of-range basis indices, which we clip into the valid range so the
        lookup is well-defined.  The corresponding rate is 0, so the
        looked-up amplitude is multiplied by zero and never affects the
        dynamics.  ``cache`` is accepted for protocol parity and unused.
        """
        h    = self.hamiltonian
        dim  = self.log_amp.shape[0]
        idx  = self.basis_index
        old_index = idx(config)

        if new_configs is not None:
            new_indices = jax.vmap(idx)(new_configs)
            new_indices = jnp.clip(new_indices, 0, dim - 1)
            return (self.log_amp[new_indices]
                    - self.log_amp[old_index]).astype(self.dtype)

        ks          = jnp.arange(self.M_total)
        new_cfgs    = jax.vmap(h.generate_state, in_axes=(None, 0))(config, ks)
        new_indices = jax.vmap(idx)(new_cfgs)
        new_indices = jnp.clip(new_indices, 0, dim - 1)

        out = self.log_amp[new_indices] - self.log_amp[old_index]
        # generate_state(config, 0) == config by construction; pin to 0 to
        # absorb any floating-point fuzz in the log-difference.
        return out.at[0].set(0.0).astype(self.dtype)

    # ── HDF5 save / load ─────────────────────────────────────────────────

    def save(self, path, group: str = "exact_ground_state", **attrs):
        """Write ``log_amp`` plus metadata to an HDF5 file.

        Datasets: ``log_amp`` and ``local_hilbert_space``.  Group attributes:
        ``N``, ``family`` (e.g. ``"SpinHalf"``), ``E0``, plus any user keyword
        arguments.  :meth:`load` checks ``family``
        and ``N`` against the Hamiltonian it is rebound to.
        """
        from ..walkers._h5io import write_attrs
        family = _family(self.hamiltonian)
        with h5py.File(path, "w") as f:
            g = f.require_group(group)
            g.create_dataset("log_amp", data=np.asarray(self.log_amp))
            g.create_dataset("local_hilbert_space",
                             data=np.asarray(self.local_hilbert_space,
                                             dtype=float))
            write_attrs(g, dict(N=int(self.N), family=family,
                                 E0=float(self.E0), **attrs))

    @classmethod
    def load(cls, path, hamiltonian, group: str = "exact_ground_state",
             dtype=jnp.float32) -> "ExactGroundState":
        """Load a guide saved with :meth:`save` and bind it to ``hamiltonian``.

        Unlike the other guides' ``load``, ``hamiltonian`` is required: the
        guide keeps a reference to it (for ``M_total`` and ``generate_state``).
        Raises ``ValueError`` if the saved family or N does not match the
        Hamiltonian.
        """
        family = _family(hamiltonian)
        with h5py.File(path, "r") as f:
            g = f[group]
            missing = [k for k in ("family", "N", "E0") if k not in g.attrs]
            if missing or "log_amp" not in g:
                raise ValueError(
                    f"{path}:{group} is not an ExactGroundState file "
                    f"(missing {missing + ([] if 'log_amp' in g else ['log_amp'])})")
            log_amp     = np.asarray(g["log_amp"][()])
            saved_family = str(g.attrs["family"]).strip()
            saved_N      = int(g.attrs["N"])
            # the saved local Hilbert space restores the index map exactly
            # (a file without it falls back to the Hamiltonian's own)
            saved_lhs = (tuple(float(v) for v in g["local_hilbert_space"][()])
                         if "local_hilbert_space" in g else None)
            saved_E0 = float(g.attrs["E0"])
        if saved_family != family:
            raise ValueError(
                f"saved guide family={saved_family!r} but loading into "
                f"{family!r}"
            )
        h_N = getattr(hamiltonian, "N", None)
        if h_N is not None and saved_N != int(h_N):
            raise ValueError(
                f"saved guide N={saved_N} but Hamiltonian has N={h_N}"
            )
        guide = cls(hamiltonian, log_amp=log_amp, dtype=dtype,
                    local_hilbert_space=saved_lhs, N=saved_N)
        # supplying log_amp skips the diagonalization, so E0 would be NaN --
        # the file recorded it, so restore rather than discard it
        guide.E0 = saved_E0
        return guide


# ── pytree registration ───────────────────────────────────────────────────

def _egs_flatten(obj):
    # E0 is a CHILD, not aux: aux is compared on every jit-cache lookup, and
    # E0 is NaN when log_amp was supplied (NaN != NaN, so two identical guides
    # never shared a cache entry) and otherwise differs in the last bits
    # between two diagonalizations of the same Hamiltonian.  As a leaf it is
    # never compared; after a jit round trip it comes back as a 0-d array.
    children = (obj.log_amp, obj.hamiltonian, obj.E0)
    # local_hilbert_space and _powers are tuples, so aux stays hashable --
    # raw arrays there would be unhashable and break JIT
    aux      = (obj.dtype, obj.M_total,
                obj.local_hilbert_space, obj._powers, obj.N)
    return children, aux


def _egs_unflatten(aux, children):
    dtype, M_total, local_hilbert_space, powers, N = aux
    log_amp, hamiltonian, E0 = children
    g = object.__new__(ExactGroundState)
    g.log_amp     = log_amp
    g.hamiltonian = hamiltonian
    g.dtype       = dtype
    g.M_total     = M_total
    g.E0          = E0
    g.local_hilbert_space = local_hilbert_space
    g._powers     = powers
    g.N           = N
    return g


jax.tree_util.register_pytree_node(
    ExactGroundState, _egs_flatten, _egs_unflatten,
)
