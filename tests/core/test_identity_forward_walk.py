"""Identity-operator checks of the forward-walking pure estimator.

For the identity operator (M_total=1, rate 1) the forward-walk value is the
bare shadow weight product.  If the shadow chain also consumes the main
chain's own future keys it replays the main chain exactly, so every numerator
term equals its denominator term and <identity>_p == 1 for every p and stride
-- an exact check of the window / stride / row-mapping bookkeeping.  With
independent shadow keys the result is 1 only on average.

The forward walk is jitted here as it is inside the production scan; called
eagerly it retraces on every call.
"""
import jax
import jax.numpy as jnp
import numpy as np

from pygfmc import run_gfmc
from pygfmc.walkers import prepare_walkers_from_configs
from pygfmc.engine.estimators import _estimate_observables, estimate_average_log_branch_weight
from pygfmc.engine.evolution import evolve_walkers, forward_walk_operator
from pygfmc.walkers import reconfiguration
from pygfmc.hamiltonian import SpinHalf

_L        = 10
_N_WALKER = 64
_N_STEPS  = 200
_LAM      = 15.0
_P_ARRAY  = [8, 16, 32]


class IdentityOp:
    """Measurement-only identity operator: one diagonal element of weight 1."""
    M_total = 1
    n_off   = 0

    def get_rates(self, config, return_configs=False):
        return jnp.ones(1, dtype=config.dtype), None


jax.tree_util.register_pytree_node(
    IdentityOp, lambda op: ((), None), lambda aux, children: IdentityOp())

# step size, guide, method, depths and shift are static, as in the production scan
_forward_walk = jax.jit(forward_walk_operator,
                        static_argnames=("step_size", "guide", "method", "m_array", "lam"))


def _tfim(L, dtype):
    edges = np.asarray([(i, (i + 1) % L) for i in range(L)], dtype=np.int32)
    return SpinHalf.create(edges, J_x=0.0, J_y=0.0, J_z=-4.0, h_x=1.5, dtype=dtype)


def _run_chain(dtype, stride, shared_keys):
    """Main chain with run_gfmc's key recurrence; at every stride-th step launch
    the identity forward walk, seeding its shadow with the main chain's NEXT key
    when ``shared_keys``."""
    h = _tfim(_L, dtype)
    walkers = prepare_walkers_from_configs(jnp.ones((_N_WALKER, _L), dtype=dtype))
    key = jax.random.PRNGKey(11)
    m_array = tuple(p // 2 for p in sorted(_P_ARRAY))
    op      = IdentityOp()
    alw     = jnp.asarray(1.0, dtype=walkers.log_weight.dtype)

    weights, energies, fw_rows, norm_rows, steps = [], [], [], [], []
    for t in range(_N_STEPS):
        key, sub = jax.random.split(key)
        k1 = jax.random.fold_in(sub, 0)
        k2 = jax.random.fold_in(sub, 1)
        walkers = evolve_walkers(walkers, h, 1.0, None, "single_step",
                                 k1, alw, lam=_LAM)
        walkers, log_G, E, _ = reconfiguration(walkers, k2)
        weights.append(float(log_G))
        energies.append(float(E))

        if (t + 1) % stride == 0:
            fw, fw_norm = _forward_walk(
                walkers, op, h, step_size=1.0, guide=None, method="single_step",
                rng_key=jax.random.fold_in(key, 10_000 + t), avg_log_weight=alw,
                m_array=m_array, lam=_LAM, shadow_key=key if shared_keys else None)
            fw_rows.append(np.asarray(fw))
            norm_rows.append(np.asarray(fw_norm))
            steps.append(t)

    _, obs = _estimate_observables(
        sorted(_P_ARRAY), np.asarray(weights), np.asarray(energies),
        off_diag_np=np.stack(fw_rows)[:, None, :], op_names=["identity"],
        m_array=list(m_array), off_diag_steps=np.asarray(steps),
        off_diag_norm_np=np.stack(norm_rows)[:, None, :])
    return obs["identity"]


def test_identity_shared_keys_is_exactly_one(x64):
    """Shared shadow keys: <identity>_p == 1 to 1e-9 (float64) for every p and stride."""
    with x64():
        for stride in (1, 4):
            obs = _run_chain(jnp.float64, stride, shared_keys=True)
            for p in sorted(_P_ARRAY):
                assert abs(obs[p] - 1.0) < 1e-9, (
                    f"stride={stride}, p={p}: <identity> = {obs[p]:.12f} != 1")


def test_identity_independent_keys_is_one_on_average():
    """Independent shadow keys: <identity>_p is 1 up to lineage noise."""
    obs = _run_chain(jnp.float32, 1, shared_keys=False)
    for p in sorted(_P_ARRAY):
        assert abs(obs[p] - 1.0) < 0.2, (
            f"p={p}: <identity> = {obs[p]:.4f} deviates implausibly from 1")


def test_identity_run_gfmc_correlated_multistep(x64):
    """Through the full run_gfmc path, <identity>_p == 1 to 1e-9 at ss=8 multi_step and ss=1."""
    with x64():
        L = 6
        h = _tfim(L, jnp.float64)
        for ss, method in ((8, "multi_step"), (1, "single_step")):
            w = prepare_walkers_from_configs(jnp.ones((64, L), dtype=jnp.float64))
            alw = estimate_average_log_branch_weight(
                w, h, None, method=method, lam=10.0)
            _, _, obs, _, _ = run_gfmc(
                w, 600, h, ss, None, [16, 32, 64, 128],
                method=method, lam=10.0, rng_key=jax.random.PRNGKey(3),
                off_diag_operators={"one": IdentityOp()}, return_raw=True,
                avg_log_weight=alw, verbosity=0, off_diag_stride=4,
                warmup_steps=100)
            for p, v in obs["one"].items():
                assert abs(v - 1.0) < 1e-9, (ss, p, v)
