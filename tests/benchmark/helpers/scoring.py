"""Observable scoring shared by the QuSpin and TeNPy benchmark tests.

  * Energies are compared per site (``ATOL_E``).
  * Extensive ``*_tot`` observables are compared per site, against the
    test's ``atol_obs``.
  * Bond correlators ``*_bond`` use ``BOND_ATOL + BOND_RTOL * |ref|``: a pure
    relative bound is unusable because guide-free GFMC carries a saturating
    ~0.06 bias on single-bond correlators.
  * Other intensive observables are compared relative to ``max(|ref|, 1)``.

These are broad regression tolerances, not confidence intervals or a
demonstration of convergence. Zero references are asserted with absolute
tolerances too; absence of a field does not establish degeneracy.
"""

from __future__ import annotations

import numpy as np

from .gfmc_bench import best_p_value

ATOL_E = 0.05          # per site
BOND_ATOL = 0.08
BOND_RTOL = 0.03


def expected_observables(cfg):
    from .gfmc_bench import measurement_operators
    diag, off_diag = measurement_operators(cfg)
    return set(diag) | set(off_diag)


def check_observable_contract(cfg, ref_obs, obs_results):
    """Never let a missing output/reference silently remove an assertion."""
    expected = expected_observables(cfg)
    assert set(obs_results) == expected, (cfg["id"], expected, set(obs_results))
    for name, windows in obs_results.items():
        if "gfmc" in cfg:
            assert set(windows) == set(cfg["gfmc"]["p_array"]), (name, "missing windows")
        assert windows and all(np.isfinite(v).all() for v in windows.values()), name
    unavailable = cfg.get("reference_unavailable", {})
    assert set(unavailable) <= expected
    assert all(isinstance(reason, str) and reason.strip() for reason in unavailable.values())
    assert not (set(unavailable) & set(ref_obs)), "remove obsolete reference exclusions"
    required = expected - set(unavailable)
    assert required <= set(ref_obs), (cfg["id"], "missing references", required - set(ref_obs))
    for name, reason in unavailable.items():
        print(f"    {name}: reference unavailable: {reason}")
    return required


def deviation(name, gfmc_val, ref_val, N):
    if name.endswith("_tot"):
        return abs(float(gfmc_val) - ref_val) / N
    if name.endswith("_bond"):
        return abs(float(gfmc_val) - ref_val)
    return abs(float(gfmc_val) - ref_val) / max(abs(ref_val), 1.0)


def tolerance(name, ref_val, atol_obs):
    if name.endswith("_bond"):
        return BOND_ATOL + BOND_RTOL * abs(float(ref_val))
    return atol_obs


def check_energy(cfg, method, e_gfmc, E0_ref, p_best):
    e_gfmc_site, e_ref_site = e_gfmc / cfg["N"], E0_ref / cfg["N"]
    print(f"\n  {cfg['id']:45s} [{method}]  E/site: gfmc={e_gfmc_site:.5f}  "
          f"ref={e_ref_site:.5f}  (p={p_best})")
    assert abs(e_gfmc_site - e_ref_site) < ATOL_E, (
        f"{cfg['id']} [{method}]: energy disagrees with reference: "
        f"gfmc={e_gfmc_site:.5f}, ref={e_ref_site:.5f}")


def check_observables(cfg, method, ref_obs, obs_results, atol_obs):
    """Assert every measured observable against its reference."""
    for name in sorted(check_observable_contract(cfg, ref_obs, obs_results)):
        ref_val = ref_obs[name]
        gfmc_val, _ = best_p_value(obs_results[name])
        diff = deviation(name, gfmc_val, ref_val, cfg["N"])
        line = (f"    {name:20s} gfmc={float(gfmc_val):.5f}  ref={ref_val:.5f}  "
                f"diff={diff:.3f}")
        print(line)
        assert diff < tolerance(name, ref_val, atol_obs), (
            f"{cfg['id']} [{method}]: observable {name!r} disagrees: "
            f"gfmc={gfmc_val!r}, ref={ref_val!r}")
