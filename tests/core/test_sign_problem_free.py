"""Sign-problem-free guards: a wrong-sign coupling must be REJECTED.

A silent acceptance would not crash but bias the energy.  ``sign_check=False``
bypasses the guard for measurement operators.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.rates import all_configs, rebuild_matrix
from pygfmc.hamiltonian import PlaquetteModel, SpinHalf, ToricCode, chain

# the J_pmpm message names the caller-side remedy, not only the violation
REMEDY = re.escape("Fix: swap S^x <-> S^y")
_PLAQS_2 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
_CFG_BOTH = jnp.array([1., -1., 1., -1., -1., 1., -1., 1.], dtype=jnp.float32)


def test_plaquette_site_count_comes_from_the_plaquettes():
    """No ``N`` argument: the site count comes from the plaquettes, and gaps raise."""
    m = PlaquetteModel.create(_PLAQS_2, g=1.0, mu=0.0)
    assert m.N == int(np.max(np.asarray(_PLAQS_2))) + 1

    with pytest.raises(ValueError, match="appear in none"):
        PlaquetteModel.create([[0, 1, 2, 3], [6, 7, 8, 9]], g=1.0, mu=0.0)
    with pytest.raises(ValueError, match="plaquette indices must be nonnegative integers"):
        PlaquetteModel.create([[-1, 0, 1, 2]], g=1.0, mu=0.0)


def test_plaquette_sign_check_rejects_negative_g():
    """``H = -g Sum_p flip_p`` needs g >= 0; measurement operators (g <= 0) bypass it."""
    with pytest.raises(ValueError, match="sign-problem-free"):
        PlaquetteModel.create(_PLAQS_2, g=-1.0, mu=0.0)
    with pytest.raises(ValueError, match="sign-problem-free"):
        PlaquetteModel.create(_PLAQS_2, g=[-0.1, 1.0], mu=0.0)
    # escape hatch for measurement operators
    m = PlaquetteModel.create(_PLAQS_2, g=-1.0, mu=0.0, sign_check=False)
    rates, _ = m.get_rates(_CFG_BOTH)
    assert float(rates[1]) > 0 and float(rates[2]) > 0


def test_toric_code_sign_check_rejects_negative():
    """Negative A_s or h_x is rejected unless sign_check=False."""
    with pytest.raises(ValueError, match="sign-problem-free"):
        ToricCode.create(2, 2, A_s=-0.5, B_p=1.0)
    with pytest.raises(ValueError, match="sign-problem-free"):
        ToricCode.create(2, 2, A_s=1.0, h_x=-0.1)
    h = ToricCode.create(2, 2, A_s=-0.5, sign_check=False)
    assert float(h.A_s[0]) == -0.5


def test_toric_code_sign_problem_free():
    """Every off-diagonal rate ``get_rates`` returns is <= 0 when A_s >= 0 and
    h_x >= 0, over the full basis; the rates are what the propagators sample."""
    configs = all_configs((1.0, -1.0), 8)
    for A_s, B_p, h_z, h_x in [
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.5),
        (0.5, 1.0, 0.2, 0.3),
        (1.0, 0.5, 0.2, 0.3),
    ]:
        h = ToricCode.create(2, 2, A_s=A_s, B_p=B_p, h_z=h_z, h_x=h_x)
        rates = np.asarray(jax.vmap(lambda c: h.get_rates(c)[0])(jnp.asarray(configs)))
        assert np.any(rates[:, 1:] < 0), "the model must have off-diagonal terms"
        assert rates[:, 1:].max() <= 0.0, (
            f"Sign problem: max off-diagonal rate={rates[:, 1:].max():.4e} "
            f"for A_s={A_s}, B_p={B_p}, h_z={h_z}, h_x={h_x}")


def test_spinhalf_rejects_positive_jpmpm_instead_of_swapping():
    """A wrong-sign J_pmpm must RAISE, not be silently "fixed" by a J_x <-> J_y swap.

    The swap is a pi/2 rotation about z, which does not preserve h_x.
    """
    edges = chain(8, periodic=True).edges

    # J_x > J_y  ->  J_pmpm = (J_x - J_y)/4 > 0, with a ferro J_pm
    with pytest.raises(ValueError, match="sign-problem-free"):
        SpinHalf.create(edges, J_x=-0.3, J_y=-1.1, J_z=0.5, h_x=0.7)
    # ... and the message must name the remedy, not just the violation
    with pytest.raises(ValueError, match=REMEDY):
        SpinHalf.create(edges, J_x=-0.3, J_y=-1.1, J_z=0.5)

    # per-edge path: one bad edge among many is enough, and it is reported
    n = len(edges)
    J_x = -np.ones(n)
    J_y = -np.ones(n)
    J_x[3] = -0.2                      # J_pmpm[3] = (−0.2 + 1)/4 > 0
    with pytest.raises(ValueError, match="Offending edges"):
        SpinHalf.create(edges, J_x=J_x, J_y=J_y, J_z=0.5)

    # the caller-side swap is accepted, and leaves J_pm / J_z untouched
    ok = SpinHalf.create(edges, J_x=-1.1, J_y=-0.3, J_z=0.5, h_x=0.7)
    assert float(ok.J_pmpm[0]) == pytest.approx(-0.2)
    assert float(ok.J_pm[0]) == pytest.approx(-0.35)

    # measurement operators still bypass the check
    meas = SpinHalf.create(edges, J_x=-0.3, J_y=-1.1, J_z=0.5, sign_check=False)
    assert float(meas.J_pmpm[0]) == pytest.approx(+0.2)


def test_spinhalf_field_sign_convention(x64):
    """``H = ... - Sum_i (h_z S^z_i + h_x S^x_i)``: h_x >= 0 is sign-problem free.

    Same convention as ToricCode, checked against an explicit 2x2 matrix and
    against ToricCode's own field terms (``sigma = 2 S``, so SpinHalf's fields
    are twice ToricCode's).  A negative h_x raises; ``sign_check=False``
    builds the measurement operator ``+Sum_i S^x_i`` as ``h_x=-1``.
    """
    hz, hx = 0.6, 0.8
    with x64():
        h = SpinHalf.create([], h_z=hz, h_x=hx, N=1, dtype=jnp.float64)
        H = rebuild_matrix(h, all_configs((1.0, -1.0), 1), dtype=jnp.float64)
        sz = np.diag([0.5, -0.5])
        sx = np.array([[0.0, 0.5], [0.5, 0.0]])
        np.testing.assert_allclose(H, -(hz * sz + hx * sx), rtol=0, atol=1e-14)
        assert H[0, 1] < 0 and H[1, 0] < 0

        # field-only ToricCode on the 1x1 torus (N = 2) has the same spectrum
        tc = ToricCode.create(1, 1, A_s=0.0, B_p=0.0, h_z=hz / 2, h_x=hx / 2,
                              dtype=jnp.float64)
        sh = SpinHalf.create([], h_z=hz, h_x=hx, N=2, dtype=jnp.float64)
        two = all_configs((1.0, -1.0), 2)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(rebuild_matrix(sh, two, dtype=jnp.float64)),
            np.linalg.eigvalsh(rebuild_matrix(tc, two, dtype=jnp.float64)), rtol=0, atol=1e-12)

    for bad in (-0.1, [0.2, -0.1, 0.3], np.array([0.2, -0.1, 0.3])):
        with pytest.raises(ValueError, match="h_x must be >= 0"):
            SpinHalf.create([(0, 1), (1, 2)], h_x=bad)
    SpinHalf.create([(0, 1), (1, 2)], h_x=[0.2, 0.0, 0.3])          # accepted

    meas = SpinHalf.create([], h_x=-1.0, N=3, sign_check=False)
    rates, _ = meas.get_rates(jnp.ones(3))
    assert np.all(np.asarray(rates[1:]) == 0.5)                     # +Sum S^x
