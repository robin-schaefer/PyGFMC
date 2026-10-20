"""run_gfmc_pmap on four fake CPU devices: return contract and resumability.

The device count is fixed when XLA initializes, so this file re-runs itself as
a worker subprocess with ``--xla_force_host_platform_device_count=4``; the
worker reports one JSON verdict per check and each becomes a test case.
Checked: return_raw=False / stride stay on pmap with and without
off-diagonal operators, a strided pmap run measures exactly the stride-1
run's values at the strided steps (main chain bit-identical) on the same
grid as run_gfmc, ``stats['parameters/*']`` incl. rng_key_next,
bit-identical N+N vs 2N resume (with an anti-vacuous control), production
starting at the caller's key, and a pinned alw still running the warm-up.
"""
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow  # the worker subprocess fixture takes several minutes on a CPU

_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODES = os.path.join(os.path.dirname(_TESTS), "codes")
_N_DEV = 4

CHECKS = [
    "pmap_stride_without_off_diag",
    "pmap_return_raw_false_with_off_diag",
    "run_dispatch_return_raw_false_uses_pmap",
    "run_dispatch_stride_uses_pmap",
    "pmap_stride_is_a_subset_of_stride_1",
    "pmap_gated_grid_matches_run_gfmc",
    "pmap_stats_parameters",
    "pmap_resume_bit_identical",
    "pmap_resume_control_differs",
    "pmap_production_starts_at_caller_key",
    "pmap_pinned_alw_still_warms_up",
    "run_gfmc_resume_bit_identical",
]


@pytest.fixture(scope="module")
def worker_verdicts():
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") +
                        f" --xla_force_host_platform_device_count={_N_DEV}"
                        ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker"],
        env=env, cwd=_TESTS, capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("VERDICTS ")]
    if proc.returncode != 0 or not lines:
        pytest.fail(f"worker failed (rc={proc.returncode})\n"
                    f"--- stdout ---\n{proc.stdout[-4000:]}\n"
                    f"--- stderr ---\n{proc.stderr[-4000:]}")
    return json.loads(lines[-1][len("VERDICTS "):])


@pytest.mark.parametrize("check", CHECKS)
def test_multidevice(worker_verdicts, check):
    """One worker check ran on 4 devices and passed."""
    assert worker_verdicts.get("n_devices") == _N_DEV, worker_verdicts
    verdict = worker_verdicts.get(check, "check did not run")
    assert verdict == "ok", f"{check}: {verdict}"


# worker side (runs in the subprocess only)
def _worker():
    sys.path[:0] = [_CODES, _TESTS]
    import traceback
    import numpy as np
    import jax
    import jax.numpy as jnp
    from pygfmc.hamiltonian import SpinHalf
    from pygfmc import run, run_gfmc
    from pygfmc.engine.estimators import dense_and_names
    from pygfmc.engine.multi_gpu import run_gfmc_pmap
    from pygfmc.engine.estimators import estimate_average_log_branch_weight
    from core.walkers import neel_walkers
    from pygfmc.wavefunction import JastrowSpinHalf

    L, NW, DT = 8, 32, jnp.float32
    edges = [(i, (i + 1) % L) for i in range(L)]
    h = SpinHalf.create(edges, J_x=0.0, J_y=0.0, J_z=1.0, h_x=0.5, dtype=DT)
    h_sx = SpinHalf.create(edges, J_x=0.0, J_y=0.0, J_z=0.0, h_x=-1.0,
                           sign_check=False, dtype=DT)
    guide = JastrowSpinHalf(h.N, h.edges, beta=0.2)
    lam = len(edges) / 4
    P = [2, 4]
    common = dict(method="cont_time", verbosity=0, number_of_outputs=3)
    key0 = jax.random.PRNGKey(3)

    def walkers0():
        return neel_walkers(NW, L, dtype=DT)

    def pmap(walkers, n, **kw):
        kw = {**common, **kw}
        return run_gfmc_pmap(walkers, n, h, 0.5, guide, P, lam=lam,
                             n_devices=_N_DEV, **kw)

    def series(raw):
        out = {"energy": np.asarray(raw["energy"]),
               "log_weights": np.asarray(raw["log_weights"])}
        if raw.get("off_diag") is not None:
            out["off_diag"] = dense_and_names(raw["off_diag"])[0]
            out["off_diag_norm"] = dense_and_names(raw["off_diag_norm"])[0]
        return out

    verdicts = {"n_devices": jax.device_count()}

    def check(fn):
        try:
            fn()
            verdicts[fn.__name__] = "ok"
        except Exception:                                   # noqa: BLE001
            verdicts[fn.__name__] = traceback.format_exc()[-1500:]
        return fn

    @check
    def pmap_stride_without_off_diag():
        # off_diag_stride has nothing to act on without off-diagonal
        # operators, and `run` does not fall back for it in that case
        out = pmap(walkers0(), 10, rng_key=key0, warmup_steps=8,
                   off_diag_stride=2)
        assert len(out) == 5

    @check
    def pmap_return_raw_false_with_off_diag():
        # equal windows (implied by return_raw=False) gate the measurement
        out = pmap(walkers0(), 12, rng_key=key0, warmup_steps=8,
                   return_raw=False, off_diag_operators={"sx": h_sx})
        assert len(out) == 4, f"expected 4-tuple, got {len(out)}"
        assert int(out[3]["parameters/off_diag_equal_windows"]) == 1
        assert np.isfinite(out[2]["sx"][max(P)])

    @check
    def run_dispatch_return_raw_false_uses_pmap():
        out = run(walkers0(), 10, h, 0.5, guide, P, lam=lam, rng_key=key0,
                  warmup_steps=8, return_raw=False, **common)
        assert len(out) == 4, f"expected 4-tuple, got {len(out)}"
        assert out[3]["number_of_devices"] == _N_DEV, (
            "run() should use the pmap backend here")
        # with off-diagonal operators it stays on pmap as well
        out = run(walkers0(), 12, h, 0.5, guide, P, lam=lam, rng_key=key0,
                  warmup_steps=8, return_raw=False,
                  off_diag_operators={"sx": h_sx}, **common)
        assert len(out) == 4 and out[3]["number_of_devices"] == _N_DEV

    @check
    def run_dispatch_stride_uses_pmap():
        out = run(walkers0(), 12, h, 0.5, guide, P, lam=lam, rng_key=key0,
                  warmup_steps=8, off_diag_stride=3,
                  off_diag_operators={"sx": h_sx}, **common)
        assert out[3]["number_of_devices"] == _N_DEV, (
            "run() should keep a strided off-diagonal run on pmap")

    NS, STRIDE = 24, 3

    @check
    def pmap_stride_is_a_subset_of_stride_1():
        # the measurement keys are drawn every step, so gating only skips
        # work: the main chain is bit-identical and every measured row equals
        # the stride-1 run's row at that step
        ops = {"sx": h_sx}
        _, _, _, _, r1 = pmap(walkers0(), NS, rng_key=key0, warmup_steps=8,
                              off_diag_operators=ops)
        _, _, obs_s, st_s, rs = pmap(walkers0(), NS, rng_key=key0,
                                     warmup_steps=8, off_diag_stride=STRIDE,
                                     off_diag_operators=ops)
        s1, ss = series(r1), series(rs)
        for k in ("energy", "log_weights"):
            assert np.array_equal(s1[k], ss[k]), f"main chain differs: {k}"
        steps = np.asarray(rs["off_diag_steps"])
        expect = np.arange(NS)[(np.arange(NS) + 1) % STRIDE == 0]
        assert np.array_equal(steps, expect), (steps, expect)
        for k in ("off_diag", "off_diag_norm"):
            assert ss[k].shape[0] == NS // STRIDE, (k, ss[k].shape)
            assert np.array_equal(ss[k], s1[k][steps]), (
                f"{k}: strided rows differ from the stride-1 rows")
        assert int(st_s["parameters/off_diag_stride"]) == STRIDE
        assert int(st_s["parameters/off_diag_offset"]) == STRIDE - 1
        assert all(np.isfinite(obs_s["sx"][p]) for p in P)

    @check
    def pmap_gated_grid_matches_run_gfmc():
        # both backends measure the same steps, for a stride and for the
        # equal windows of return_raw=False
        ops = {"sx": h_sx}
        for kw in (dict(off_diag_stride=STRIDE),
                   dict(off_diag_stride=2, return_raw=False)):
            out_p = pmap(walkers0(), NS, rng_key=key0, warmup_steps=8,
                         off_diag_operators=ops, **kw)
            out_s = run_gfmc(walkers0(), NS, h, 0.5, guide, P, lam=lam,
                             rng_key=key0, warmup_steps=8,
                             off_diag_operators=ops, **{**common, **kw})
            st_p, st_s = out_p[3], out_s[3]
            for k in ("parameters/off_diag_offset",
                      "parameters/off_diag_equal_windows"):
                assert int(st_p[k]) == int(st_s[k]), (kw, k, st_p[k], st_s[k])
            if kw.get("return_raw", True):
                assert np.array_equal(np.asarray(out_p[4]["off_diag_steps"]),
                                      np.asarray(out_s[4]["off_diag_steps"])), kw

    @check
    def pmap_stats_parameters():
        _, _, _, st, _ = pmap(walkers0(), 10, rng_key=key0, warmup_steps=8,
                              off_diag_operators={"sx": h_sx})
        for k in ("parameters/rng_key", "parameters/rng_key_next",
                  "parameters/n_steps", "parameters/n_walkers",
                  "parameters/p_array", "parameters/method",
                  "parameters/off_diag_operators"):
            assert k in st, f"stats lacks {k}"
        assert np.array_equal(st["parameters/rng_key"], np.asarray(key0))
        assert not np.array_equal(st["parameters/rng_key_next"],
                                  np.asarray(key0))
        assert int(st["parameters/n_steps"]) == 10

    N, W = 12, 16
    resume = {}

    def _resume_runs():
        if resume:
            return resume
        ops = {"sx": h_sx}
        _, _, _, st_full, raw_full = pmap(walkers0(), 2 * N, rng_key=key0,
                                          warmup_steps=W,
                                          off_diag_operators=ops)
        w_a, _, _, st_a, raw_a = pmap(walkers0(), N, rng_key=key0,
                                      warmup_steps=W, off_diag_operators=ops)
        alw = st_a["avg_log_weight"]
        assert alw == st_full["avg_log_weight"]
        _, _, _, _, raw_b = pmap(
            w_a, N, rng_key=jnp.asarray(st_a["parameters/rng_key_next"]),
            warmup_steps=0, avg_log_weight=alw, off_diag_operators=ops)
        _, _, _, _, raw_c = pmap(w_a, N, rng_key=key0, warmup_steps=0,
                                 avg_log_weight=alw, off_diag_operators=ops)
        resume.update(full=series(raw_full), a=series(raw_a),
                      b=series(raw_b), c=series(raw_c))
        return resume

    @check
    def pmap_resume_bit_identical():
        r = _resume_runs()
        for k, full in r["full"].items():
            joined = np.concatenate([r["a"][k], r["b"][k]], axis=0)
            assert joined.shape == full.shape, (k, joined.shape, full.shape)
            assert np.array_equal(joined, full), (
                f"{k}: max|d| = {np.max(np.abs(joined - full)):.3e}")

    @check
    def pmap_resume_control_differs():
        r = _resume_runs()
        second = r["full"]["energy"][N:]
        assert not np.array_equal(r["c"]["energy"], second), (
            "re-passing the original key reproduced the continuation: "
            "the resume test would be vacuous")

    @check
    def pmap_production_starts_at_caller_key():
        # alw estimated from the walkers vs the same value passed back from
        # stats, no warm-up either way: production starts at the caller key
        alw0 = estimate_average_log_branch_weight(walkers0(), h, guide,
                                                  method="cont_time")
        _, _, _, st1, raw1 = pmap(walkers0(), 10, rng_key=key0,
                                  avg_log_weight=alw0)
        _, _, _, _, raw2 = pmap(walkers0(), 10, rng_key=key0, warmup_steps=0,
                                avg_log_weight=st1["avg_log_weight"])
        assert np.array_equal(np.asarray(raw1["energy"]),
                              np.asarray(raw2["energy"]))

    @check
    def pmap_pinned_alw_still_warms_up():
        alw = 1.234
        _, _, _, st1, raw1 = pmap(walkers0(), 10, rng_key=key0,
                                  warmup_steps=16, avg_log_weight=alw)
        _, _, _, _, raw0 = pmap(walkers0(), 10, rng_key=key0,
                                warmup_steps=0, avg_log_weight=alw)
        assert abs(st1["avg_log_weight"] - alw) < 1e-6, "alw not kept fixed"
        assert not np.array_equal(np.asarray(raw1["energy"]),
                                  np.asarray(raw0["energy"])), (
            "warmup_steps=16 with a pinned alw gave the same chain as "
            "warmup_steps=0: the ramp was skipped")

    @check
    def run_gfmc_resume_bit_identical():
        # the single-device reference for the same contract
        def one(walkers, n, **kw):
            return run_gfmc(walkers, n, h, 0.5, guide, P, lam=lam,
                            **{**common, **kw})
        _, _, _, _, raw_full = one(walkers0(), 2 * N, rng_key=key0,
                                   warmup_steps=W)
        w_a, _, _, st_a, raw_a = one(walkers0(), N, rng_key=key0,
                                     warmup_steps=W)
        _, _, _, _, raw_b = one(
            w_a, N, rng_key=jnp.asarray(st_a["parameters/rng_key_next"]),
            warmup_steps=0, avg_log_weight=st_a["avg_log_weight"])
        joined = np.concatenate([np.asarray(raw_a["energy"]),
                                 np.asarray(raw_b["energy"])])
        assert np.array_equal(joined, np.asarray(raw_full["energy"]))

    print("VERDICTS " + json.dumps(verdicts), flush=True)


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
