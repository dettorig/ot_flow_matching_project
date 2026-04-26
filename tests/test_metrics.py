"""Unit tests for the metrics module."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from otfm.metrics import (
    empirical_w2_squared_hungarian,
    mode_distance_metrics,
    occupancy_kl,
    occupancy_hist,
    normalized_path_energy,
    sliced_wasserstein_2,
)


def test_w2_self_is_zero():
    x = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
    assert empirical_w2_squared_hungarian(x, x) < 1e-10


def test_w2_translation():
    """W2^2(X, X+t) = ||t||^2 in expectation; with optimal matching the estimate is exact."""
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 2))
    t = jnp.array([1.0, 2.0])
    y = x + t
    val = empirical_w2_squared_hungarian(x, y)
    assert abs(val - float(jnp.sum(t**2))) < 1e-3


def test_sliced_wasserstein_self_is_zero():
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 2))
    assert sliced_wasserstein_2(x, x, n_projections=64) < 1e-10


def test_sliced_wasserstein_translation():
    x = jax.random.normal(jax.random.PRNGKey(0), (256, 2))
    t = jnp.array([1.0, 0.0])
    y = x + t
    sw = sliced_wasserstein_2(x, y, n_projections=512, key=jax.random.PRNGKey(1))
    # SW2^2 of translation by t is ||t||^2 / d in 2D under random projections
    # (E_theta (theta . t)^2 = ||t||^2 / 2). Allow generous tolerance.
    expected = float(jnp.sum(t**2)) / 2.0
    assert 0.5 * expected < sw < 1.5 * expected, (sw, expected)


def test_occupancy_hist_sums_to_one():
    centers = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
    counts, probs = occupancy_hist(x, centers, n_modes=3)
    assert counts.sum() == 50
    assert abs(probs.sum() - 1.0) < 1e-9


def test_mode_distance_metrics_non_negative():
    centers = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
    mean_d2, q90_d2 = mode_distance_metrics(x, centers)
    assert mean_d2 >= 0.0
    assert q90_d2 >= 0.0


def test_occupancy_kl_is_zero_for_identical_distributions():
    p = jnp.array([0.2, 0.3, 0.5])
    assert abs(occupancy_kl(p, p)) < 1e-12


def test_normalized_path_energy_zero_when_matching_w2():
    assert abs(normalized_path_energy(2.5, 2.5)) < 1e-12
