"""Sanity checks on the toy 2D samplers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from otfm.datasets import (
    eight_gaussians_centers,
    sample_8gaussians,
    sample_gaussian,
    sample_moons,
    sample_spiral,
)
from otfm.metrics import nearest_center_idx


def test_sample_gaussian_shape_and_dtype():
    x = sample_gaussian(jax.random.PRNGKey(0), 32)
    assert x.shape == (32, 2)
    assert x.dtype == jnp.float32


def test_sample_8gaussians_modes_covered():
    n = 4096
    x = sample_8gaussians(jax.random.PRNGKey(0), n)
    centers = eight_gaussians_centers()
    idx = np.asarray(nearest_center_idx(x, centers))
    counts = np.bincount(idx, minlength=8)
    # All 8 modes should be hit roughly uniformly.
    assert (counts > 0).all(), counts
    assert counts.min() / counts.max() > 0.5, counts


def test_sample_moons_shape():
    x = sample_moons(jax.random.PRNGKey(0), 100)
    assert x.shape == (100, 2)


def test_samples_are_finite():
    for sampler in (sample_gaussian, sample_8gaussians, sample_moons, sample_spiral):
        x = sampler(jax.random.PRNGKey(1), 64)
        assert jnp.all(jnp.isfinite(x))
