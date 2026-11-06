"""Deterministic numerical quantiles of a finite mixture of Student-t marginals."""
from __future__ import annotations

import numpy as np
from scipy.special import stdtr

from ..probability import LEVELS

SOLVER = {"method": "scipy.special.stdtr_bisection", "atol": 1e-7, "rtol": 1e-9,
          "cdf_tolerance": 1e-7, "max_iter": 100, "max_expand": 64}


def validate_parameters(weights, df, loc, scale):
    weights, df, loc, scale = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (weights, df, loc, scale)))
    if not all(np.isfinite(a).all() for a in (weights, df, loc, scale)):
        raise ValueError("Nonfinite Student-t mixture parameters")
    if np.any(df <= 0) or np.any(scale <= 0) or np.any(weights < 0):
        raise ValueError("Mixture degrees of freedom/scales must be positive and weights nonnegative")
    if not np.allclose(weights.sum(-1), 1, rtol=0, atol=1e-6):
        raise ValueError("Mixture weights must sum to one")
    return weights / weights.sum(-1, keepdims=True), df, loc, scale


def mixture_cdf(x, weights, df, loc, scale):
    return (weights * stdtr(df, (np.asarray(x)[..., None] - loc) / scale)).sum(-1)


def mixture_quantiles(weights, df, loc, scale, levels=LEVELS, **options):
    p = {**SOLVER, **options}
    w, df, loc, scale = validate_parameters(weights, df, loc, scale)
    levels = np.asarray(levels, dtype=np.float64)
    if levels.ndim != 1 or np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0):
        raise ValueError("Mixture quantile levels must be strictly increasing inside (0,1)")
    low, high = np.min(loc - 8 * scale, axis=-1), np.max(loc + 8 * scale, axis=-1)
    for _ in range(p["max_expand"]):
        left = mixture_cdf(low, w, df, loc, scale) > levels[0]
        right = mixture_cdf(high, w, df, loc, scale) < levels[-1]
        if not left.any() and not right.any():
            break
        width = np.maximum(high - low, 1)
        low = np.where(left, low - 2 * width, low)
        high = np.where(right, high + 2 * width, high)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("Student-t quantile bracketing overflow")
    else:
        raise ValueError("Could not bracket Student-t mixture quantiles")
    outputs = []
    for tau in levels:
        lo, hi = low.copy(), high.copy()
        for _ in range(p["max_iter"]):
            mid = lo + (hi - lo) / 2
            cdf = mixture_cdf(mid, w, df, loc, scale)
            left = cdf < tau
            lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
            tolerance = p["atol"] + p["rtol"] * np.abs(mid)
            if np.all((hi - lo <= tolerance) & (np.abs(cdf - tau) <= p["cdf_tolerance"])):
                break
        q = lo + (hi - lo) / 2
        if np.any(np.abs(mixture_cdf(q, w, df, loc, scale) - tau) > p["cdf_tolerance"]):
            raise ValueError("Student-t mixture quantile solver did not reach its CDF tolerance")
        outputs.append(q)
    return np.stack(outputs, axis=-1)
