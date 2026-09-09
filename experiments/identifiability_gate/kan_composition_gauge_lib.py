"""
Shared composition-gauge test for experiments/kan-identifiability Steps 8-9.

Theory being tested: for phi2 . phi1 = model, any invertible univariate h gives
    phi2 . phi1 = (phi2 . h^-1) . (h . phi1)
an identical composed map. h_ij := phi1_j . phi1_i^-1 (built from two
independently trained models i, j's inner functions) therefore satisfies
phi2_i ~= phi2_j . h_ij ALWAYS, as a near-tautology, WHENEVER phi1_i is
invertible on the domain used and model_i ~= model_j there (both true by
construction: models are trained to the same target and h_ij is built to make
phi1_j . phi1_i^-1 exactly cancel). That "always small residual" outcome is
expected and is a numerics sanity check, not evidence for the hypothesis on
its own -- the actual falsifiable content is:
  (1) whether phi1_i is even invertible (monotone) on a usable interval at
      all -- if not, the whole invertible-h story doesn't apply, full stop.
  (2) how far the CONSTRUCTED h_ij is from affine. If h_ij is close to affine,
      that is a genuine contradiction: the earlier affine-only quotient test
      (Step 3/4) should then have already succeeded, but it did not -- this
      must be flagged, not explained away.

Both scripts that use this (run_kan_composition_gauge_d1.py for Step 8,
run_kan_multivariate.py for Step 9) import from here so the test logic is
identical in both regimes and the results are directly comparable.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import interp1d

from run_kan_seed_variance import raw_and_quotiented  # noqa: E402  (reused, not reimplemented)


def make_marginal_layer_fn(layer, in_features: int, input_idx: int, fixed_value: float = 0.0):
    """fn(x_array) -> layer output with `input_idx` varying and every other
    input held at fixed_value, RAW (no baseline subtraction) -- composition
    testing needs genuine, unshifted functional values since h must invert the
    actual argument the next layer receives. Works for in_features=1 (d=1,
    input_idx=0 trivially) and in_features>=2 (marginal slice at d>=2)."""
    def fn(x_array: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x_array, dtype=np.float32)
        X = np.full((len(x_array), in_features), fixed_value, dtype=np.float32)
        X[:, input_idx] = x_array
        X_t = torch.from_numpy(X)
        with torch.no_grad():
            return layer(X_t)[:, 0].numpy()
    return fn


def make_layer_fn(layer):
    """fn(u_array) -> layer(u_array) for a single-input layer (e.g. layer2
    mapping one hidden unit to the output), RAW."""
    def fn(u_array: np.ndarray) -> np.ndarray:
        u_t = torch.from_numpy(np.asarray(u_array, dtype=np.float32)).unsqueeze(1)
        with torch.no_grad():
            return layer(u_t)[:, 0].numpy()
    return fn


def longest_monotone_interval(y_vals: np.ndarray) -> tuple[int, int]:
    """Longest contiguous index range [start, end) where y_vals is STRICTLY
    monotone (all increasing or all decreasing). A flat step (diff == 0) or a
    sign flip both count as a break -- no smoothing/tolerance is applied, so a
    genuinely wiggly curve is reported as genuinely wiggly."""
    diffs = np.diff(y_vals)
    signs = np.sign(diffs)
    best_start, best_end, best_len = 0, 1, 0
    start = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] or signs[i] == 0:
            length = i - start
            if length > best_len:
                best_len, best_start, best_end = length, start, i
            start = i
    length = len(signs) - start
    if length > best_len:
        best_len, best_start, best_end = length, start, len(signs)
    return best_start, best_end + 1  # +1: index range into the ORIGINAL y_vals/x_vals


def composition_gauge_test(x_grid: np.ndarray,
                            phi1_i_vals: np.ndarray, phi1_j_fn,
                            phi2_i_fn, phi2_j_fn,
                            min_interval_frac: float = 0.10,
                            n_h_pts: int = 300) -> dict:
    """
    x_grid:        common grid the edge functions were swept over (1D)
    phi1_i_vals:   phi1_i evaluated at x_grid (RAW, not baseline-subtracted --
                   genuine composition needs actual functional values, since
                   h must invert the actual argument phi2 receives)
    phi1_j_fn:     callable, phi1_j_fn(x_array) -> phi1_j evaluated at x_array
    phi2_i_fn, phi2_j_fn: callables, phi2_*_fn(u_array) -> phi2_* evaluated at u_array

    Returns a dict describing: monotone-interval usability, h_ij's deviation
    from affine, and the composition residual phi2_i vs phi2_j . h_ij.
    """
    start, end = longest_monotone_interval(phi1_i_vals)
    interval_len = end - start
    interval_frac = interval_len / len(x_grid)

    result = dict(interval_frac=float(interval_frac), interval_len=int(interval_len),
                  n_grid=int(len(x_grid)))

    if interval_frac < min_interval_frac or interval_len < 5:
        result["usable"] = False
        return result

    x_sub = x_grid[start:end]
    v_sub = phi1_i_vals[start:end]  # phi1_i restricted to its monotone stretch

    order = np.argsort(v_sub)
    v_sorted = v_sub[order]
    x_sorted = x_sub[order]
    if len(np.unique(v_sorted)) < len(v_sorted):
        # duplicate v-values would break interp1d's monotonicity assumption
        result["usable"] = False
        result["note"] = "phi1_i has duplicate values on its monotone interval"
        return result

    phi1_i_inverse = interp1d(v_sorted, x_sorted, bounds_error=False, fill_value=np.nan)

    v_grid = np.linspace(v_sorted.min(), v_sorted.max(), n_h_pts)
    x_at_v = phi1_i_inverse(v_grid)
    valid = ~np.isnan(x_at_v)
    v_grid = v_grid[valid]
    x_at_v = x_at_v[valid]

    h_vals = phi1_j_fn(x_at_v)  # h_ij(v) = phi1_j(phi1_i^{-1}(v))

    # how far is h_ij from affine? (best-fit a*v+b, normalized residual)
    h_affine_fit = raw_and_quotiented(h_vals, v_grid)

    # composition test: phi2_i(u) vs phi2_j(h_ij(u)) over u = v_grid
    phi2_i_vals = phi2_i_fn(v_grid)
    phi2_j_of_h = phi2_j_fn(h_vals)
    composition_check = raw_and_quotiented(phi2_i_vals, phi2_j_of_h)

    result.update(
        usable=True,
        x_interval=[float(x_grid[start]), float(x_grid[end - 1])],
        v_grid=v_grid.tolist(), h_vals=h_vals.tolist(), x_at_v=x_at_v.tolist(),
        h_affine_residual=h_affine_fit["quotiented"],
        h_affine_a=h_affine_fit["a"], h_affine_b=h_affine_fit["b"],
        composition_residual_raw=composition_check["raw"],
        composition_residual_quotiented=composition_check["quotiented"],
    )
    return result
