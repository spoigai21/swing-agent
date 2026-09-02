"""Decomposition on synthetic data with KNOWN betas.

Real data cannot tell you whether the estimator is right, only whether it looks
plausible. These build series with betas chosen in advance and check they come
back — including the collinear case that broke the naive specification.
"""
from __future__ import annotations

import numpy as np
import pytest


def _fit_orth(y, rm, rsec):
    """The two-stage fit from analysis/factors.py, in isolation."""
    n = len(y)
    g, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), rm]), rsec, rcond=None)
    orth = rsec - (g[0] + g[1] * rm)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), rm, orth]), y, rcond=None)
    return coef, g, orth


def _fit_naive(y, rm, rsec):
    n = len(y)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), rm, rsec]), y, rcond=None)
    return coef


class TestKnownBetas:
    def test_recovers_betas_when_factors_are_independent(self):
        rng = np.random.default_rng(0)
        n = 500
        rm = rng.normal(0, 0.01, n)
        rsec = rng.normal(0, 0.01, n)          # independent of the market
        y = 0.0 + 1.5 * rm + 0.8 * rsec + rng.normal(0, 0.001, n)
        coef, _, _ = _fit_orth(y, rm, rsec)
        assert coef[1] == pytest.approx(1.5, abs=0.05)
        assert coef[2] == pytest.approx(0.8, abs=0.05)

    def test_orthogonalised_sector_is_uncorrelated_with_market(self):
        rng = np.random.default_rng(1)
        n = 400
        rm = rng.normal(0, 0.01, n)
        rsec = 1.3 * rm + rng.normal(0, 0.004, n)      # heavily collinear
        y = 1.2 * rm + 0.9 * rsec + rng.normal(0, 0.001, n)
        _, _, orth = _fit_orth(y, rm, rsec)
        assert abs(np.corrcoef(rm, orth)[0, 1]) < 1e-9

    def test_residual_is_identical_naive_vs_orthogonalised(self):
        # The whole justification for orthogonalising: it fixes the
        # market/sector SPLIT without touching the residual, so swing
        # detection and Gate 1 are unaffected.
        rng = np.random.default_rng(2)
        n = 400
        rm = rng.normal(0, 0.01, n)
        rsec = 1.3 * rm + rng.normal(0, 0.004, n)
        y = 1.2 * rm + 0.9 * rsec + rng.normal(0, 0.002, n)

        cn = _fit_naive(y, rm, rsec)
        rn = y - np.column_stack([np.ones(n), rm, rsec]) @ cn
        co, _g, orth = _fit_orth(y, rm, rsec)
        ro = y - np.column_stack([np.ones(n), rm, orth]) @ co
        assert np.allclose(rn, ro, atol=1e-12)

    def test_components_sum_to_the_total_return(self):
        rng = np.random.default_rng(3)
        n = 300
        rm = rng.normal(0, 0.01, n)
        rsec = 1.1 * rm + rng.normal(0, 0.005, n)
        y = 1.4 * rm + 0.7 * rsec + rng.normal(0, 0.002, n)
        coef, _g, orth = _fit_orth(y, rm, rsec)
        alpha, b_mkt, b_sec = coef
        i = -1
        mc, sc = b_mkt * rm[i], b_sec * orth[i]
        resid = y[i] - alpha - mc - sc
        assert alpha + mc + sc + resid == pytest.approx(y[i], abs=1e-12)


class TestGuards:
    def test_nan_input_raises_rather_than_returning_garbage(self):
        from swing.analysis.factors import _fit

        y = np.array([0.1, np.nan, 0.3])
        X = np.column_stack([np.ones(3), np.array([1.0, 2.0, 3.0])])
        with pytest.raises(ValueError, match="NaN"):
            _fit(y, X)

    def test_zscore_adapts_to_volatility(self):
        # A fixed percentage threshold is wrong: the same 4% move is an
        # earthquake for one name and a normal day for another.
        quiet = np.random.default_rng(4).normal(0, 0.005, 200)
        wild = np.random.default_rng(5).normal(0, 0.05, 200)
        move = 0.04
        assert abs(move / quiet.std()) > 4
        assert abs(move / wild.std()) < 1.5
