"""The ``Walker`` pytree and the walker initializers."""
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from typing import NamedTuple, Sequence, Union


def key_data(rng_key):
    """``rng_key`` as a uint32 array of raw key data: a legacy ``PRNGKey``
    unchanged, a typed key (``jax.random.key``) through
    :func:`jax.random.key_data`. Stored keys come back as legacy Threefry
    keys. Other implementations are rejected rather than losing their type."""
    rng_key = jnp.asarray(rng_key)
    typed = jnp.issubdtype(rng_key.dtype, jax.dtypes.prng_key)
    if typed and rng_key.shape != ():
        raise ValueError("Checkpointing requires a single scalar PRNG key.")
    if (not typed
            and (rng_key.shape != (2,) or rng_key.dtype != jnp.uint32)):
        raise ValueError("Checkpointing requires a scalar threefry2x32 key "
                         "or a legacy uint32 key of shape (2,).")
    if (jax.config.jax_default_prng_impl != "threefry2x32"
            or str(jax.random.key_impl(rng_key)) not in ("threefry2x32", "fry")):
        raise ValueError("Checkpointing requires threefry2x32 keys and the default "
                         "threefry2x32 PRNG implementation.")
    if typed:
        rng_key = jax.random.key_data(rng_key)
    return np.asarray(rng_key, dtype=np.uint32)


def _compact_config(config):
    """Return the smallest lossless storage array for a walker config.

    Spin/toric configs are ±1 and occupation configs are small non-negative
    integers, so they fit in ``int8`` (4x smaller than float32) without loss.
    Falls back to the original array when the values are not integers in the
    int8 range (e.g. half-integer ``Spin`` configs), so storage stays lossless
    for every Hamiltonian family.  ``Walker.load`` casts whatever is stored
    back to the compute dtype, so this only affects on-disk size.
    """
    arr = np.asarray(config)
    rounded = np.rint(arr)
    if (np.array_equal(arr, rounded)
            and rounded.min() >= np.iinfo(np.int8).min
            and rounded.max() <= np.iinfo(np.int8).max):
        return rounded.astype(np.int8)
    return arr


class Walker(NamedTuple):
    """Walker population (or a single walker) as a JAX pytree.

    Attributes:
        config:     configurations, shape ``(N_walkers, N_sites)`` for a batch
                    (``(N_sites,)`` for a single walker).  Spin-1/2 models use
                    the +-1 encoding.
        log_weight: accumulated log weight since the last reconfiguration,
                    shape ``(N_walkers,)``.
        e_loc:      local energy e_L(x) of the current configuration,
                    shape ``(N_walkers,)``.
    """
    config: jnp.ndarray
    log_weight: jnp.ndarray
    e_loc: jnp.ndarray

    def save(self, path: str, rng_key=None, group: str = "walkers",
             **attrs) -> None:
        """Save walker arrays + arbitrary metadata to an HDF5 file.

        Walker arrays (``config``, ``log_weight``, ``e_loc``) are stored as
        HDF5 datasets; the JAX PRNG key (if any) is also a dataset.  ``config``
        is stored in the smallest lossless dtype (``int8`` for ±1 / small-integer
        configs, ~4x smaller than float32; original dtype otherwise) — see
        :func:`_compact_config`.  ``load`` casts it back to the compute dtype.
        Any additional keyword arguments — typically ``avg_log_weight=...``,
        but anything scalar — are stored as group attributes (same convention
        as :meth:`JastrowSpinHalf.save`).

        Args:
            path:    HDF5 file path (created or overwritten).
            rng_key: JAX PRNG key to checkpoint alongside the walkers (a
                     legacy ``PRNGKey`` or a typed ``jax.random.key``; stored
                     as raw key data and loaded back as a legacy key).
                     Only Threefry keys with the default Threefry PRNG are
                     supported; other implementations raise before writing. Pass
                     the key ``run_gfmc`` returned
                     (``stats['parameters/rng_key_next']``): the key carries
                     the chain position, so it is the whole resume state.
                     ``Walker.load`` gives it back, and passing it as
                     ``rng_key`` continues the stream instead of replaying it.
            group:   HDF5 group name to write under.
            **attrs: extra scalar metadata stored as group attributes.
        """
        from ._h5io import write_attrs
        stored_key = None if rng_key is None else key_data(rng_key)
        with h5py.File(path, "w") as f:
            g = f.require_group(group)
            g.attrs["config_dtype"] = str(self.config.dtype)
            g.create_dataset("config",     data=_compact_config(self.config))
            g.create_dataset("log_weight", data=np.asarray(self.log_weight))
            g.create_dataset("e_loc",      data=np.asarray(self.e_loc))
            if rng_key is not None:
                g.create_dataset("rng_key", data=stored_key)
            write_attrs(g, attrs)

    @classmethod
    def load(cls, path: str, group: str = "walkers",
             dtype=None, config_dtype=None):
        """Load a Walker (and optional RNG key / avg_log_weight) from an HDF5 checkpoint.

        Args:
            path:  path to the HDF5 file written by :meth:`save`.
            group: HDF5 group name to read from.
            dtype: JAX dtype for weights and local-energy estimators.  ``None``
                (the default) uses JAX's default float -- float32, or float64
                when ``jax_enable_x64`` is set -- exactly as the
                ``prepare_walkers_*`` initializers do, so a loaded ensemble
                has the same width as a freshly prepared one.
            config_dtype: Optional JAX dtype for configurations.  Defaults to
                ``dtype``.

        Returns:
            ``(walker, rng_key, avg_log_weight)`` where *rng_key* and
            *avg_log_weight* are ``None`` when not saved.
        """
        if dtype is None:
            dtype = jnp.zeros(()).dtype
        if config_dtype is None:
            config_dtype = dtype
        with h5py.File(path, "r") as f:
            g = f[group]
            config      = jnp.array(g["config"][()],     dtype=config_dtype)
            log_weight  = jnp.array(g["log_weight"][()], dtype=dtype)
            e_loc       = jnp.array(g["e_loc"][()],      dtype=dtype)
            rng_key     = jnp.array(g["rng_key"][()]) if "rng_key" in g else None
            if rng_key is not None:
                key_data(rng_key)  # reject legacy checkpoints with unsupported key bits
            avg_log_weight = jnp.array(g.attrs["avg_log_weight"], dtype=dtype) \
                          if "avg_log_weight" in g.attrs else None
        return cls(config=config, log_weight=log_weight, e_loc=e_loc), rng_key, avg_log_weight


def _base_config_for_sum(N_sites, target, states):
    """Multiset of local states of length ``N_sites`` summing exactly to ``target``.

    Returns the values themselves (ascending), ready to be permuted per walker.

    ``states`` may be any ladder of local values -- ``[-1, 1]`` for the +-1
    spin-1/2 encoding, ``[-0.5, 0.5]`` for physical S^z, ``[-1, 0, 1]`` for
    S=1, ``[0, 1, ..., n_max]`` for bosons.  Values are scaled to integers
    first (physical ladders are multiples of 1/2, 1/3, ...), so half-integer
    states are handled exactly rather than through floating-point comparison.

    The construction is greedy over the largest available increment, which is
    exact for an EQUALLY SPACED ladder -- the increments are then 1, 2, ..., K
    after scaling, so any representable remainder can be finished off.  A
    non-uniform ladder (e.g. ``[0, 3, 5]``) can strand the remainder; that is
    detected and raised rather than silently returning a config with the wrong
    sum.
    """
    vals = np.asarray(states, dtype=float).ravel()
    if vals.size < 1:
        raise ValueError("states must contain at least one local value.")
    if np.unique(vals).size != vals.size:
        raise ValueError(f"states must not repeat a value; got {list(states)}.")

    den = None
    for d in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        if (np.allclose(vals * d, np.round(vals * d))
                and abs(target * d - round(target * d)) < 1e-9):
            den = d
            break
    if den is None:
        raise ValueError(
            f"states={list(states)} and sz={target} are not commensurate with a "
            "small common denominator; pass values that are multiples of 1/2, "
            "1/3, 1/4, ...")

    S = np.sort(np.round(vals * den).astype(np.int64))
    Z = int(round(target * den))
    inc = S - S[0]                      # ascending, inc[0] == 0
    D = Z - N_sites * int(S[0])
    if D < 0 or D > N_sites * int(inc[-1]):
        lo, hi = N_sites * vals.min(), N_sites * vals.max()
        raise ValueError(
            f"sz={target} is outside the reachable range [{lo}, {hi}] for "
            f"N_sites={N_sites} and states={list(states)}.")

    counts = np.zeros(S.size, dtype=np.int64)
    remaining, d = N_sites, D
    for k in range(S.size - 1, 0, -1):
        take = min(d // int(inc[k]), remaining)
        counts[k] += take
        d -= take * int(inc[k])
        remaining -= take
    counts[0] += remaining
    if d != 0:
        raise ValueError(
            f"sz={target} is not reachable with N_sites={N_sites} sites drawn "
            f"from states={list(states)} (residual {d}/{den}). For a two-state "
            "ladder this is the usual parity constraint.")
    base = np.repeat(S, counts).astype(float) / den
    assert base.size == N_sites and abs(base.sum() - target) < 1e-9
    return base


def prepare_walkers_random(N_walkers: int, N_sites: int, rng_key: jax.Array,
                           states: Sequence[float] = (-1.0, 1.0)) -> Walker:
    """Initialize walkers by drawing every site independently from ``states``.

    Each site of each walker is an independent uniform draw, so the total
    magnetization/occupation fluctuates binomially about ``N_sites *
    mean(states)``.  Use this for Hamiltonians that do NOT conserve the
    corresponding charge (TFIM, toric code with h_x); use
    :func:`prepare_walkers_sz` to start inside a fixed sector.

    Args:
        N_walkers:    number of walkers.
        N_sites:      number of sites.
        rng_key:      JAX random key.
        states:       local values to draw from -- ``(-1, 1)`` for the +-1
                      spin-1/2 encoding (the default), ``(-0.5, 0.5)`` for
                      physical S^z, ``(-1, 0, 1)`` for S=1, ``(0, ..., n_max)``
                      for bosons.  Drawn with equal probability.

    Note there is no ``dtype`` argument: the configuration dtype is taken from
    ``states`` itself, so an integer ladder such as ``(-1, 0, 1)`` yields integer
    configurations and ``(-0.5, 0.5)`` yields floating-point ones.  The weights
    and local energies use JAX's default float dtype (float32, or float64 when
    ``jax_enable_x64`` is set); they never inherit an integer dtype.

    Returns:
        Walker batch with zero log-weights.
    """
    # The default is written as FLOATS, not (-1, 1): the configuration dtype is
    # inherited from states, so python ints would silently make the default
    # ensemble int32 -- which surfaces downstream as
    # an unsafe-cast FutureWarning from JAX's scatter when a guide writes into
    # the configuration.  Pass integers explicitly for a narrow int encoding.
    vals = jnp.asarray(states)
    if vals.ndim != 1 or vals.size < 1:
        raise ValueError(
            f"states must be a non-empty 1-D sequence of local values; got "
            f"{states!r}.")
    idx = jax.random.randint(rng_key, (N_walkers, N_sites), 0, vals.size)
    configs = vals[idx]
    # jnp.zeros WITHOUT an explicit dtype: JAX's default float follows the
    # jax_enable_x64 flag, so the weights come out float64 exactly when the rest
    # of the computation does.  Pinning float32 here breaks float64 runs -- the
    # multi_step while_loop carry is log_weight, and log_weight + log(b_x)*k
    # promotes to the Hamiltonian's float64, so the carry in/out dtypes stop
    # matching and JAX raises.
    zeros = jnp.zeros(N_walkers)
    return Walker(config=configs, log_weight=zeros, e_loc=zeros)


def prepare_walkers_sz(N_walkers: int, N_sites: int, rng_key: jax.Array,
                       sz: float = 0, states: Sequence[float] = (-1.0, 1.0)) -> Walker:
    """Initialize walkers in a fixed-charge sector drawn from ``states``.

    ``sz`` is simply the SUM over all sites of the local values, each of which
    is one of ``states``:

        sz = sum_i sigma_i,      sigma_i in states

    so its meaning follows the encoding rather than being fixed to spin-1/2.
    With the default ``states=(-1.0, 1.0)`` it is the sum of the +-1 config values,
    i.e. twice the physical S^z_tot, and the familiar constraint that
    ``N_sites + sz`` be even is just the two-state case of the general
    reachability condition.  With ``states=(-0.5, 0.5)`` the same ``sz`` is
    S^z_tot itself; with ``(-1, 0, 1)`` it is S^z_tot for S=1; with
    ``(0, ..., n_max)`` it is the total boson number.

    Every walker is an independent random permutation of one base multiset, so
    the sum is EXACT for every walker -- which is what makes this usable to pin
    a conserved sector: for a Hamiltonian that conserves the charge, walkers
    started in the wrong sector can never leave it.  A request outside
    the reachable range, or one the ladder cannot represent, raises rather than
    silently returning a config with the wrong sum.

    Args:
        N_walkers:    number of walkers.
        N_sites:      number of sites.
        rng_key:      JAX random key.
        sz:           target SUM over all sites of the local values. Default 0.
        states:       local values available per site; default ``(-1, 1)``.

    As in :func:`prepare_walkers_random` there is no ``dtype`` argument: the
    configuration dtype follows ``states``, while the weights and local
    energies use JAX's default float dtype.

    Returns:
        Walker batch with zero log-weights.
    """
    base_np = _base_config_for_sum(N_sites, sz, states)
    base    = jnp.asarray(base_np, dtype=jnp.asarray(states).dtype)
    keys    = jax.random.split(rng_key, N_walkers)
    configs = jax.vmap(lambda k: jax.random.permutation(k, base))(keys)
    # jnp.zeros WITHOUT an explicit dtype: JAX's default float follows the
    # jax_enable_x64 flag, so the weights come out float64 exactly when the rest
    # of the computation does.  Pinning float32 here breaks float64 runs -- the
    # multi_step while_loop carry is log_weight, and log_weight + log(b_x)*k
    # promotes to the Hamiltonian's float64, so the carry in/out dtypes stop
    # matching and JAX raises.
    zeros    = jnp.zeros(N_walkers)
    return Walker(config=configs, log_weight=zeros, e_loc=zeros)


def prepare_walkers_from_configs(configs: Union[Sequence, np.ndarray, jax.Array]) -> Walker:
    """Initialize walkers at user-supplied configurations.

    Like :func:`prepare_walkers_random` and :func:`prepare_walkers_sz`, this
    takes no ``dtype``: the configurations keep the dtype of ``configs``, so an
    ``int8`` input stays ``int8``.  Cast the input yourself --
    ``jnp.asarray(configs, dtype=...)`` -- to get something else.  The weights
    and local energies use JAX's default float dtype.

    Useful whenever a specific sector must be seeded exactly, e.g. the
    all-ones plaquette sector of the toric code
    (``prepare_walkers_from_configs(jnp.ones((N_walkers, N_sites)))``), or to
    start from configurations prepared elsewhere.

    Args:
        configs: ``(N_walkers, N_sites)`` array, one configuration per walker.

    Returns:
        Walker batch with zero log-weights.
    """
    configs = jnp.asarray(configs)
    if configs.ndim != 2:
        raise ValueError(
            f"configs must have shape (N_walkers, N_sites); got shape "
            f"{configs.shape}.  For a single configuration copied to every "
            f"walker use jnp.tile(config, (N_walkers, 1)).")
    # jnp.zeros WITHOUT an explicit dtype: JAX's default float follows the
    # jax_enable_x64 flag, so the weights come out float64 exactly when the rest
    # of the computation does.  Pinning float32 here breaks float64 runs -- the
    # multi_step while_loop carry is log_weight, and log_weight + log(b_x)*k
    # promotes to the Hamiltonian's float64, so the carry in/out dtypes stop
    # matching and JAX raises.
    zeros = jnp.zeros(configs.shape[0])
    return Walker(config=configs, log_weight=zeros, e_loc=zeros)
