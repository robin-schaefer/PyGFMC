"""Spin-1/2 XYZ model in longitudinal and transverse fields, and the
k-resolved structure-factor measurement operator built from it.

    H = Σ_<ij> [J_x S^x_i S^x_j + J_y S^y_i S^y_j + J_z S^z_i S^z_j]
        − Σ_i [h_z S^z_i + h_x S^x_i]

Configurations are ±1 floats (σ_i = 2 S^z_i).  See :class:`SpinHalf` for the
rate layout and the sign-problem-free convention, and :func:`StructureFactor`
for the measurement operator.
"""
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
from ._common import edges_array
import numpy as np


def _jpmpm_message(value):
    """Error text for a J_pmpm that violates the sign-problem-free convention.

    The model is never modified automatically.  Swapping J_x <-> J_y flips the
    sign of J_pmpm, but it is the global pi/2 rotation about z (S^x -> S^y,
    S^y -> -S^x), which leaves J_z S^z S^z and h_z S^z invariant yet maps
    h_x S^x -> h_x S^y.  With a transverse field the swapped model is
    therefore a different Hamiltonian, so the swap is left to the caller.
    """
    return (
        f"J_pmpm = (J_x - J_y) / 4 = {value:.6g} > 0 violates the "
        "sign-problem-free convention (J_pmpm <= 0).\n"
        "  Fix: swap S^x <-> S^y, i.e. pass J_x and J_y the other way round. "
        "That flips the sign of J_pmpm while leaving J_pm = (J_x + J_y) / 4 "
        "and J_z untouched.\n"
        "  CAUTION: the swap is a global pi/2 rotation about z, which is a "
        "symmetry only when h_x = 0.  It maps h_x S^x -> h_x S^y, so with a "
        "transverse field the swapped model is a DIFFERENT Hamiltonian, not "
        "an equivalent one.\n"
        "  Alternatively pass sign_check=False to build this model as given "
        "(correct for measurement operators, which are never propagated, but "
        "it means the walk itself is not sign-problem free)."
    )


def _jpm_message(value):
    """Error text for a J_pm > 0, which the propagator cannot sample.

    The propagator uses the NEGATED off-diagonal rates (``cumsum(-rates[1:])``),
    so it needs every off-diagonal matrix element ``<= 0``; an edge slot holds
    ``J_pm`` on anti-parallel spins, hence ``J_pm <= 0``.  An off-diagonal
    MEASUREMENT operator needs the opposite (rates ``>= 0``, i.e. ``J_pm >= 0``)
    and is built with ``sign_check=False``.
    """
    return (
        f"J_pm = (J_x + J_y) / 4 = {value:.6g} > 0 violates the "
        "sign-problem-free convention (J_pm <= 0): the off-diagonal elements "
        "of the propagator change sign and the walk cannot sample it.\n"
        "  Fix (Hamiltonian): on a BIPARTITE lattice apply the sublattice "
        "rotation (pi about z on one sublattice, S^x,S^y -> -S^x,-S^y), i.e. "
        "pass -J_x and -J_y.  It maps J_pm -> -J_pm and J_pmpm -> -J_pmpm and "
        "leaves J_z, h_z invariant, but makes a uniform h_x STAGGERED, so it "
        "is an equivalence only when h_x = 0.  On a frustrated (non-bipartite) "
        "lattice there is no such rotation: the model has a genuine sign "
        "problem.\n"
        "  For an off-diagonal MEASUREMENT operator J_pm >= 0 is exactly what "
        "is required (rates >= 0) -- build it with sign_check=False."
    )


def _negate_field(v):
    """Return ``-v`` for a scalar, list/tuple or array field (None stays None).

    Negation is exact in floating point, so negating here and storing keeps
    every rate bit-identical to passing the negated value to a
    ``+Σ``-convention constructor.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [-x for x in v]
    return -v


class SpinHalf(NamedTuple):
    """Spin-1/2 XYZ model with longitudinal and transverse fields.

        H = Σ_<ij> [J_x S^x_i S^x_j + J_y S^y_i S^y_j + J_z S^z_i S^z_j]
            − Σ_i [h_z S^z_i + h_x S^x_i]

    The fields enter with a MINUS sign (the same convention as
    :class:`ToricCode`), so a positive ``h_x`` is the sign-problem-free one.

    The exchange term is rewritten per edge as ``J_pm (S+_i S-_j + S-_i S+_j)
    + J_pmpm (S+_i S+_j + S-_i S-_j) + J_z S^z_i S^z_j`` with ``J_pm = (J_x + J_y)/4`` and
    ``J_pmpm = (J_x - J_y)/4``.  Configurations are ±1 floats, σ_i = 2 S^z_i.
    All couplings may be uniform or per-edge / per-site.  Build instances
    with :meth:`create`.

    Layout of the rates returned by :meth:`get_rates`::

        rates[0]                 : diagonal H_{σσ}
        rates[1 : 1+n_hx]        : single-site flips, −h_x/2 each (n_hx = N if h_x != 0)
        rates[1+n_hx : M_total]  : one slot per edge, J_pm if the two spins are
                                   anti-parallel, J_pmpm if parallel
                                   (n_jpm = n_edges if J_pm or J_pmpm != 0)

    Sign-problem-free convention: ``J_pm ≤ 0``, ``J_pmpm ≤ 0`` and
    ``h_x ≥ 0``; ``J_z`` and ``h_z`` are unrestricted.  ``create`` enforces
    all three by raising (never by swapping couplings).  An off-diagonal
    measurement operator needs the opposite sign (rates ≥ 0) and is built
    with ``sign_check=False``: for ``+Σ_i S^x_i`` that is ``create([], N=N,
    h_x=-1.0, sign_check=False)``, whose rates ``−h_x/2`` are non-negative.
    """

    dtype: object
    edge_i: jnp.ndarray
    edge_j: jnp.ndarray
    N: int
    J_z: object    # jnp array, shape (1,) uniform or (n_edges,) non-uniform
    J_pm: object   # jnp array, shape (1,) uniform or (n_edges,) non-uniform
    J_pmpm: object # jnp array, shape (1,) uniform or (n_edges,) non-uniform
    # The stored fields are the coefficients of +Σ S^z and +Σ (S^+ + S^-), i.e.
    # the NEGATED public arguments: h_z = −(h_z arg), h_x_half = −(h_x arg)/2.
    h_z: object      # jnp array, −h_z, shape (1,) uniform or (N,) per-site; or None
    h_x_half: object # jnp array, −h_x/2, shape (1,) uniform or (N,) per-site; or None
    # Static structure flags — hashable Python primitives, safe as pytree aux
    n_hx_static: int = 0
    n_jpm_static: int = 0
    has_jz: bool = False
    uniform_jz: bool = True   # True means J_z has shape (1,)
    has_hz: bool = False

    @property
    def edges(self):
        """Reconstruct (n_edges, 2) from edge_i/edge_j (Python-side only)."""
        return np.stack([np.asarray(self.edge_i), np.asarray(self.edge_j)], axis=1)

    @property
    def n_hx(self):
        """Number of transverse-field off-diagonal elements (static Python int)."""
        return self.n_hx_static

    @property
    def n_jpm(self):
        """Number of edge off-diagonal elements (static Python int)."""
        return self.n_jpm_static

    @property
    def n_off(self):
        """Total number of off-diagonal elements (static Python int)."""
        return self.n_hx_static + self.n_jpm_static

    @property
    def M_total(self):
        """Total number of transitions including diagonal (static Python int): n_off + 1."""
        return self.n_hx_static + self.n_jpm_static + 1

    @classmethod
    def create(cls, edges, J_x=0, J_y=0, J_z=0, h_z=None, h_x=None, N=None, *,
               dtype=jnp.float32, sign_check=True):
        """Build a spin-1/2 XYZ Hamiltonian.

        Args:
            edges:      (n_edges, 2) site index pairs (may be empty).
            J_x, J_y:   scalar or (n_edges,) exchange couplings.  They enter
                        only through ``J_pm = (J_x+J_y)/4`` and
                        ``J_pmpm = (J_x-J_y)/4``; ``J_pmpm > 0`` raises when
                        ``sign_check`` is True (pass J_x and J_y the other way
                        round yourself if that is a symmetry of your model --
                        it is not when ``h_x != 0``).
            J_z:        scalar or (n_edges,) S^z S^z coupling (any sign).
            h_z:        scalar, (N,) or None; longitudinal field, entering
                        as ``−h_z Σ S^z_i`` (any sign).
            h_x:        scalar, (N,) or None; transverse field, entering as
                        ``−h_x Σ S^x_i``.  Must be ≥ 0 when ``sign_check``
                        is True.
            N:          number of sites.  Inferred from ``edges`` and any
                        per-site array when None; give it explicitly for an
                        operator without two-body terms (``edges=[]``).
            dtype:      JAX dtype of the couplings.
            sign_check: enforce the sign-problem-free convention: raise on
                        ``J_pm > 0``, ``J_pmpm > 0`` or ``h_x < 0``.  Pass
                        False for measurement operators, which need
                        ``J_pm >= 0`` and are never propagated.
        """
        # Fields enter H with a MINUS sign, -Σ (h_z S^z + h_x S^x).  Negate
        # them once here; everything below works with the coefficients of
        # +Σ S^z and +Σ S^x (so the stored model, and every rate, is
        # bit-identical to a +Σ-convention model with negated fields).  The
        # h_x sign check below therefore tests the NEGATED value for > 0,
        # i.e. the public h_x for < 0.
        h_z = _negate_field(h_z)
        h_x = _negate_field(h_x)

        edges_arr, _, N_min = edges_array(edges)
        # site-only operators: take the size from a per-site array too
        for _site_arr in (h_z, h_x):
            if np.ndim(_site_arr) > 0:
                N_min = max(N_min, np.shape(_site_arr)[0])
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
        uniform = not (hasattr(J_x, "__len__") or hasattr(J_y, "__len__") or hasattr(J_z, "__len__"))

        if uniform:
            J_pmpm = (J_x - J_y) / 4.0
            if J_pmpm > 0 and sign_check:
                raise ValueError(_jpmpm_message(float(J_pmpm)))
            J_z_arr    = jnp.array([J_z], dtype=dtype)
            J_pm_arr   = jnp.array([(J_x + J_y) / 4.0], dtype=dtype)
            J_pmpm_arr = jnp.array([J_pmpm], dtype=dtype)
        else:
            # Any of the three may be scalar while the others are per-edge;
            # scalars are broadcast to (n_edges,).  This covers e.g. a
            # diagonal measurement operator
            #     create(edges, J_x=0.0, J_y=0.0, J_z=<per-edge array>).
            def _per_edge(v):
                a = jnp.asarray(v, dtype=dtype)
                if a.ndim == 0 or a.shape[0] == 1:
                    return jnp.broadcast_to(a.reshape(1), (len(edges_arr),))
                return a
            J_x = _per_edge(J_x)
            J_y = _per_edge(J_y)
            J_z_arr = _per_edge(J_z)
            if J_x.shape[0] != len(edges_arr) or J_y.shape[0] != len(edges_arr) or J_z_arr.shape[0] != len(edges_arr):
                raise ValueError("Non-uniform couplings must match the number of edges.")
            J_pm_arr   = (J_x + J_y) / 4.0
            J_pmpm_arr = (J_x - J_y) / 4.0
            if sign_check and jnp.any(J_pmpm_arr > 0):
                bad = np.flatnonzero(np.asarray(J_pmpm_arr) > 0)
                raise ValueError(
                    _jpmpm_message(float(np.asarray(J_pmpm_arr)[bad[0]]))
                    + f"\n  Offending edges: {bad.tolist()[:10]}"
                    + (" ..." if bad.size > 10 else "")
                    + f" ({bad.size} of {len(edges_arr)})."
                )

        if h_z is None:
            h_z_arr = None
        elif np.ndim(h_z) > 0:
            h_z_arr = jnp.asarray(h_z, dtype=dtype)
            if h_z_arr.shape[0] != N:
                raise ValueError(f"Non-uniform h_z must have length N={N}, got {h_z_arr.shape[0]}.")
        else:
            h_z_arr = jnp.array([h_z], dtype=dtype)

        if h_x is None:
            h_x_half_arr = None
        elif np.ndim(h_x) > 0:
            h_x_arr = np.asarray(h_x, dtype=float)
            if h_x_arr.shape[0] != N:
                raise ValueError(f"Non-uniform h_x must have length N={N}, got {h_x_arr.shape[0]}.")
            if sign_check and np.any(h_x_arr > 0):
                raise ValueError(
                    "h_x must be >= 0 for all sites (sign-problem-free "
                    "convention; H contains -h_x Σ S^x, so a field written "
                    "as +a Σ S^x is passed as h_x = -a).")
            h_x_half_arr = jnp.array(h_x_arr / 2, dtype=dtype)
        else:
            if sign_check and h_x > 0:
                raise ValueError(
                    "h_x must be >= 0 (sign-problem-free convention; H "
                    "contains -h_x Σ S^x, so a field written as +a Σ S^x is "
                    "passed as h_x = -a).")
            h_x_half_arr = jnp.array([h_x / 2], dtype=dtype)

        if sign_check and np.any(np.asarray(J_pm_arr) > 0):
            raise ValueError(_jpm_message(float(np.max(np.asarray(J_pm_arr)))))

        idx_dtype = jnp.int16 if N <= 32767 else jnp.int32
        n_edges = len(edges_arr)
        n_hx_s  = N if (h_x_half_arr is not None and float(np.linalg.norm(np.asarray(h_x_half_arr))) > 0) else 0
        n_jpm_s = n_edges if (float(np.linalg.norm(np.asarray(J_pm_arr))) > 0 or float(np.linalg.norm(np.asarray(J_pmpm_arr))) > 0) else 0
        has_jz_s     = float(np.linalg.norm(np.asarray(J_z_arr))) > 0
        uniform_jz_s = len(J_z_arr) == 1
        has_hz_s     = h_z_arr is not None and float(np.linalg.norm(np.asarray(h_z_arr))) > 0

        return cls(
            dtype=dtype,
            edge_i=jnp.array(edges_arr[:, 0], dtype=idx_dtype),
            edge_j=jnp.array(edges_arr[:, 1], dtype=idx_dtype),
            N=N,
            J_z=J_z_arr,
            J_pm=J_pm_arr,
            J_pmpm=J_pmpm_arr,
            h_z=h_z_arr,
            h_x_half=h_x_half_arr,
            n_hx_static=n_hx_s,
            n_jpm_static=n_jpm_s,
            has_jz=has_jz_s,
            uniform_jz=uniform_jz_s,
            has_hz=has_hz_s,
        )


    def diagonal_weight(self, config):
        """Return just the diagonal transition weight (scalar).

        Equivalent to ``get_rates(config)[0][0]`` but skips the
        (M_total,) rates allocation and the off-diagonal branches — useful
        for diagonal-operator measurement in hot loops.
        """
        if self.has_jz:
            s_i = config[self.edge_i]
            s_j = config[self.edge_j]
            anti_parallel = s_i != s_j
            if self.uniform_jz:
                N_anti_parallel = jnp.sum(anti_parallel)
                N_parallel = len(self.edge_i) - N_anti_parallel
                diag_weight = (self.J_z[0] / 4) * (N_parallel - N_anti_parallel)
            else:
                diag_weight = jnp.sum(jnp.where(anti_parallel, -self.J_z / 4.0, self.J_z / 4.0))
        else:
            diag_weight = jnp.array(0.0, dtype=self.dtype)

        if self.has_hz:
            diag_weight = diag_weight + jnp.sum(self.h_z * config) / 2

        return diag_weight

    def get_rates(self, config, return_configs=False):
        """Return transition rates, and optionally the full connected-configs tensor.

        The default skips the (M_total, N) connected-configs tensor to avoid an
        O(n_walkers * M_total * N) allocation inside vmapped evolution loops; use
        ``generate_state(config, idx)`` to materialize the single selected config.
        Pass ``return_configs=True`` to obtain the full tensor.

        Returns:
            ``(rates, new_configs)``:

            * ``rates``: shape (M_total,) — rates[0] = diagonal, rates[1:] =
              off-diagonal.
            * ``new_configs``: shape (M_total, N) when ``return_configs=True``
              (new_configs[0] = config unchanged, new_configs[1:] = connected
              configs for each off-diagonal element), else None.
        """
        M = self.M_total   # static Python int
        # The diagonal is cast to the rate dtype explicitly: with float64
        # configurations (the exact-diagonalization path) ``h_z * config`` is
        # float64, and scattering it into a float32 array would otherwise cast
        # implicitly, which JAX flags with a FutureWarning.
        rates = jnp.zeros(M, dtype=self.dtype).at[0].set(
            jnp.asarray(self.diagonal_weight(config), dtype=self.dtype))

        if self.n_hx_static > 0:
            rates = rates.at[1 : 1 + self.N].set(jnp.broadcast_to(self.h_x_half, (self.N,)))

        if self.n_jpm_static > 0:
            anti_parallel = config[self.edge_i] != config[self.edge_j]
            rates = rates.at[1 + self.n_hx_static :].set(
                jnp.where(anti_parallel, self.J_pm, self.J_pmpm))

        if not return_configs:
            return rates, None

        new_configs = jnp.zeros((M, self.N), dtype=config.dtype).at[0].set(config)

        if self.n_hx_static > 0:
            flip_masks  = jnp.eye(self.N, dtype=bool)
            site_configs = jnp.where(flip_masks, -config, config)
            new_configs = new_configs.at[1 : 1 + self.N].set(site_configs)

        if self.n_jpm_static > 0:
            n_edges = len(self.edge_i)
            rows    = jnp.arange(n_edges)
            base    = jnp.broadcast_to(config[None, :], (n_edges, self.N))
            flip_mask = (jnp.zeros((n_edges, self.N), dtype=bool)
                         .at[rows, self.edge_i].set(True)
                         .at[rows, self.edge_j].set(True))
            edge_configs = jnp.where(flip_mask, -base, base)
            new_configs = new_configs.at[1 + self.n_hx_static :].set(edge_configs)

        return rates, new_configs

    def generate_state(self, config, idx):
        """Generate new config by applying off-diagonal transition.

        idx == 0: return config unchanged (diagonal)
        idx >= 1: apply the (idx-1)-th off-diagonal flip
        """
        # Off-diagonal: shift index by 1 to map rates array index to flip index
        off_idx = idx - 1
        n_hx  = self.n_hx   # static Python int
        n_jpm = self.n_jpm  # static Python int

        if n_hx > 0 and n_jpm > 0:
            # Mixed: site flip for off_idx < n_hx, edge flip otherwise.  The
            # two sites to flip are selected first (a site flip names the
            # same site twice; both reads take the ORIGINAL value, so the
            # second write repeats the first), then the configuration is
            # updated once.
            is_site  = off_idx < n_hx
            edge_idx = jnp.clip(off_idx - n_hx, 0, n_jpm - 1)
            site_a   = jnp.where(is_site, off_idx, self.edge_i[edge_idx])
            site_b   = jnp.where(is_site, off_idx, self.edge_j[edge_idx])
            flipped = config.at[site_a].set(-config[site_a])
            flipped = flipped.at[site_b].set(-config[site_b])
        elif n_hx > 0:
            # Only site flips: off_idx is always a site index
            flipped = config.at[off_idx].set(-config[off_idx])
        else:
            # Only edge flips
            site_i = self.edge_i[off_idx]
            site_j = self.edge_j[off_idx]
            flipped = config.at[site_i].set(-config[site_i])
            flipped = flipped.at[site_j].set(-config[site_j])

        # Return config if diagonal (idx == 0), else return flipped
        return jnp.where(idx == 0, config, flipped)

    @property
    def local_hilbert_space(self):
        """Per-site values, ordered so that position i is basis digit i.

        Matches this class's own basis convention: digit = (1 - sigma)/2, so sigma = +1 is digit 0.
        """
        return (1.0, -1.0)




def _h_xyz_flatten(h):
    leaves = [h.edge_i, h.edge_j, h.J_z, h.J_pm, h.J_pmpm]
    has_h_z      = h.h_z      is not None
    has_h_x_half = h.h_x_half is not None
    if has_h_z:
        leaves.append(h.h_z)
    if has_h_x_half:
        leaves.append(h.h_x_half)
    aux = (h.N, h.dtype, has_h_z, has_h_x_half,
           h.n_hx_static, h.n_jpm_static, h.has_jz, h.uniform_jz, h.has_hz)
    return leaves, aux


def _h_xyz_unflatten(aux, leaves):
    N, dtype, has_h_z, has_h_x_half, n_hx_s, n_jpm_s, has_jz_s, uniform_jz_s, has_hz_s = aux
    edge_i, edge_j, J_z, J_pm, J_pmpm = leaves[:5]
    idx = 5
    h_z = leaves[idx] if has_h_z else None
    if has_h_z:
        idx += 1
    h_x_half = leaves[idx] if has_h_x_half else None
    return SpinHalf(
        dtype=dtype,
        edge_i=edge_i,
        edge_j=edge_j,
        N=N,
        J_z=J_z,
        J_pm=J_pm,
        J_pmpm=J_pmpm,
        h_z=h_z,
        h_x_half=h_x_half,
        n_hx_static=n_hx_s,
        n_jpm_static=n_jpm_s,
        has_jz=has_jz_s,
        uniform_jz=uniform_jz_s,
        has_hz=has_hz_s,
    )


jax.tree_util.register_pytree_node(SpinHalf, _h_xyz_flatten, _h_xyz_unflatten)


# ---------------------------------------------------------------------------
# k-resolved structure factor
# ---------------------------------------------------------------------------

_SF_CHANNELS = {                             # name -> (a, b) on (PM, PP)
    "spsm":      (1.0, 0.0),
    "spsp":      (0.0, 1.0),
    "spsm+spsp": (1.0, 1.0),
    "szsz":      None,                          # diagonal, built separately
}


def StructureFactor(positions, k, channel="SpSm", dtype=jnp.float32):
    r"""The k-resolved correlator ``AB(0) + AB(k)`` as ONE :class:`SpinHalf`.

        O     = sum_{i<j} [1 + cos(k.(r_i - r_j))] (A_iB_j + h.c.)
        O     = AB(0) + AB(k)
        AB(q) = sum_{i<j} cos(q.(r_i - r_j)) (A_iB_j + h.c.)

    Returns a `SpinHalf` with ``M_total = 1 + n_kept``, where ``n_kept`` is the
    number of pairs ``i < j`` with nonzero weight, for use as an off-diagonal
    measurement operator (``off_diag_operators`` of ``run_gfmc``).  Combined
    with the separately measured k = 0 operator it gives the second-moment
    correlation length ``xi^2 = [S(0)/S(k) - 1] / khat^2``.

    **Positions, as in** :class:`~pygfmc.hamiltonian.SzSzStructureFactor`.
    The displacement of a pair is ``r_i - r_j`` of the positions as given; no
    periodic wrap is applied and no image is chosen.  For a ``k`` in the
    reciprocal lattice of the simulation cell (``k.L = 2*pi*integer`` for every
    periodic vector ``L`` of the system, see :func:`is_valid_k`) every
    periodic image gives the same ``cos(k.dr)``, so the result does not depend
    on which image the positions encode.  ``k`` is not checked here.

    **``channel="SzSz"`` is diagonal and warns.**  ``A = B = S^z`` gives
    ``A_iB_j + h.c. = 2 S^z_iS^z_j``, so the operator is
    ``sum_{i<j} 2[1 + cos(k.dr)] S^z_iS^z_j``, which is exactly ``SzSzStructureFactor(positions, [0, k])`` summed over its two
    wavevectors, i.e. the same ``AB(0) + AB(k)`` structure as the other
    channels.  The result has ``M_total = 1`` and goes in ``diag_operators``.
    A ``UserWarning`` points out that
    :class:`~pygfmc.hamiltonian.SzSzStructureFactor` measures the same
    correlator for many ``k`` at once in ``O(N n_k)`` and needs neither the
    ``1 + cos`` shift nor the ``k = 0`` subtraction, since a diagonal
    operator has no sign constraint.

    Args:
        positions: (N, d) site coordinates; row ``i`` is the position of the
                site at index ``i`` in ``config``, so ``N`` must match the
                Hamiltonian's ``N``.  ``d`` is arbitrary; it only has to match
                ``k``.
        k:      (d,) wavevector, in the same units as ``positions``.  One
                operator measures one wavevector.
        channel: one of the four below; case- and space-insensitive.
        dtype:  JAX dtype of the couplings.

    ==================  ==============================
    channel             per-pair operator
    ==================  ==============================
    ``"SpSm"``          ``S+S- + S-S+`` = ``2(S^xS^x + S^yS^y)`` (transverse)
    ``"SpSp"``          ``S+S+ + S-S-``
    ``"SpSm+SpSp"``     ``4 S^xS^x``
    ``"SzSz"``          ``2 S^zS^z`` (diagonal, warns)
    ==================  ==============================

    **Why these off-diagonal three.**  `get_rates` emits one off-diagonal slot per edge,
    ``where(anti_parallel, J_pm, J_pmpm)``, so per edge only
    ``PM = S+S- + S-S+`` and ``PP = S+S+ + S-S-`` exist, and an off-diagonal
    measurement needs every rate >= 0.  The reachable set is therefore the
    non-negative cone ``a*PM + b*PP`` (entered via ``J_x = 2(a+b)W``,
    ``J_y = 2(a-b)W``).  Since ``PM + PP = 4 S^xS^x`` and
    ``PM - PP = 4 S^yS^y``, YY needs a negative coefficient: build it as the
    ``"SpSm"`` operator minus the ``"SpSp"`` one.  ``S.S = PM/2 + ZZ`` mixes the
    diagonal and off-diagonal paths: measure ``"SpSm"`` (off-diagonal) and
    ``"SzSz"`` (diagonal) separately, then add.  Mixed species (XY, XZ, ...) are not in the span.

    **The weight is ``1 + cos``, not ``cos``.**  Off-diagonal measurement
    samples a successor by inverse CDF over the rates, so mixed-sign weights
    are impossible; the ``+1`` shift makes every weight non-negative at the
    price of the additive ``AB(0)``, which the caller removes by subtracting
    the separately measured k = 0 operator.  Pairs whose weight vanishes
    (``cos(k.dr) = -1``, to within ``1e-12``, since an image other than the
    minimum one leaves a rounding residue) are dropped, so ``M_total`` does
    not reflect the pair count.

    **Memory.**  The operator has one transition slot per pair, about
    ``N^2/2``, and applying it to the ensemble materialises
    ``(n_walkers, M_total)`` rates (times ``N`` for a guide that needs the
    successor configurations).  Pass ``off_diag_batch_size`` to ``run_gfmc``
    to apply it in walker chunks; the result is bit-identical.

    **Sign conditions.**  ``create``'s ``J_pm <= 0`` / ``J_pmpm <= 0`` are
    propagator (sign-problem) conditions, whereas a measurement operator
    needs rates >= 0, i.e. ``J_pm >= 0`` on every channel and ``J_pmpm > 0``
    on the two with ``b > 0``.  Every channel therefore builds the operator
    with ``sign_check=False``.

    **Normalisation.**  For ``"SpSm"`` and ``"SzSz"``, ``AB(q)`` above equals the ordered sum
    ``Re sum_{i!=j} exp(-i q.r_ij) A_iB_j``, with no factor 2, because the
    adjoint of ``A_iB_j`` is its i<->j partner (``S+_jS-_i``, resp.
    ``S^z_jS^z_i``): listing each pair once and adding ``+h.c.`` reproduces
    the ordered sum.  For ``"SzSz"``, ``AB(q)`` is therefore the
    ``S^{zz}(q)`` of :class:`~pygfmc.hamiltonian.SzSzStructureFactor`.  The
    identity with the ordered sum fails for ``"SpSp"``, whose ordered sum is
    not Hermitian; the definition of ``AB(q)`` above holds for every channel.
    """
    key = str(channel).lower().replace(" ", "")
    if key not in _SF_CHANNELS:
        extra = ""
        if key in ("yy", "sysy"):
            extra = (" -- YY = (PM - PP)/4 needs a NEGATIVE J_pmpm, so the rate "
                     "on PARALLEL pairs would be < 0. Build it from two "
                     "operators: SpSm minus SpSp.")
        elif key in ("zz", "sz"):
            extra = " -- the diagonal channel is spelled 'SzSz'."
        elif key in ("ss", "s.s", "heisenberg"):
            extra = (" -- S.S = PM/2 + ZZ mixes the diagonal and off-diagonal "
                     "paths and its rates[0] is not sign-definite; measure the "
                     "SpSm channel (off-diagonal) and the SzSz channel "
                     "(diagonal) separately, then add.")
        raise ValueError(
            f"channel={channel!r} is not reachable. The complete set is "
            "['SpSm', 'SpSp', 'SpSm+SpSp'], the non-negative cone over "
            "PM = S+S-+S-S+ and PP = S+S++S-S-, plus the diagonal 'SzSz'"
            f"{extra}")

    # ---- validate the caller's geometry --------------------------------
    # Dimension is ARBITRARY: positions and k only have to agree with each
    # other. Nothing here is 3D-specific -- the weight is a dot product.
    r = np.asarray(positions, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"positions must be (N, d), got {r.shape}")
    N = r.shape[0]
    kv = np.asarray(k, dtype=np.float64)
    if kv.ndim > 1 and kv.shape[0] != 1:
        raise ValueError(
            f"k must be ONE wavevector (d,), got {kv.shape}; build one "
            "operator per wavevector.")
    kv = kv.reshape(-1)
    if kv.size != r.shape[1]:
        raise ValueError(
            f"k has {kv.size} components but positions are "
            f"{r.shape[1]}-dimensional; they must match.")
    if N < 2:
        raise ValueError(f"positions must hold at least two sites, got N={N}")
    # every pair i < j; the onsite term is k-independent and left out
    p = np.stack(np.triu_indices(N, k=1), axis=1)
    dr = r[p[:, 0]] - r[p[:, 1]]

    W = 1.0 + np.cos(dr @ kv)
    # zero up to the rounding residue a non-minimum periodic image leaves
    nz = W > 1e-12
    p, W = p[nz].astype(np.int32), W[nz]
    if p.shape[0] == 0:
        raise ValueError(
            "no pairs with nonzero weight -- every weight vanished, which "
            "means k maps every displacement to the zone boundary. Check k.")

    if key == "szsz":
        warnings.warn(
            "StructureFactor(channel='SzSz') is a DIAGONAL operator "
            "(M_total = 1): pass it in diag_operators. "
            "SzSzStructureFactor(positions, k_vectors) measures the same "
            "correlator for many k at once in O(N n_k), without the 1+cos "
            "shift and the k = 0 subtraction.", UserWarning, stacklevel=2)
        # A_iB_j + h.c. = 2 S^z_iS^z_j, and the diagonal weight of an edge is
        # J_z S^z_iS^z_j, so J_z = 2W
        return SpinHalf.create(p, J_z=jnp.asarray(2.0 * W, dtype=dtype),
                               N=N, dtype=dtype, sign_check=False)

    a_pm, b_pp = _SF_CHANNELS[key]
    # J_pm = (J_x + J_y)/4 and J_pmpm = (J_x - J_y)/4, so
    #     J_x = 2*(a + b)*W ,  J_y = 2*(a - b)*W
    return SpinHalf.create(p,
                           J_x=jnp.asarray(2.0 * (a_pm + b_pp) * W, dtype=dtype),
                           J_y=jnp.asarray(2.0 * (a_pm - b_pp) * W, dtype=dtype),
                           J_z=0.0, N=N, dtype=dtype,
                           sign_check=False)   # measurement op: rates >= 0
