"""Check shared window selection against explicit scalar weighted sums."""
import numpy as np
import pytest

from pygfmc.engine.estimators import _off_diagonal_estimates


@pytest.mark.parametrize("trim", [0, 4])
def test_multiple_operators_on_irregular_measurement_grid(trim):
    rng = np.random.default_rng(19)
    n = 40
    windows = [2, 4, 8]
    steps = np.array([0, 2, 3, 6, 9, 10, 15, 18, 24, 31, 35, 39])
    names = ["first", "second", "third"]
    log_weights = rng.normal(0, 0.1, n)
    values = rng.normal(size=(len(steps), len(names), len(windows)))
    norms = rng.normal(0, 0.1, values.shape)
    actual = _off_diagonal_estimates(
        windows, [p // 2 for p in windows], np.cumsum(log_weights),
        values, norms, steps, names, n, trim)
    for j, p in enumerate(windows):
        m = p // 2
        for idx, name in enumerate(names):
            numerator, denominator = 0., 0.
            for row, k in enumerate(steps):
                if k < p + max(0, trim - m) or k >= n - m:
                    continue
                bra = np.exp(log_weights[k - m + 1:k + 1].sum())
                ket = np.exp(log_weights[k + 1:k + m + 1].sum())
                numerator += bra * np.exp(norms[row, idx, j]) * values[row, idx, j]
                denominator += bra * ket
            assert actual[name][p] == pytest.approx(numerator / denominator, abs=1e-14)
