"""Marginal and structural checks on the coupling routines."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from otfm.couplings import (
    hungarian_pairing,
    independent_pairing,
    interpolate_plans,
    make_hungarian_ot_plan,
    make_independent_plan,
    plan_descriptors,
    sample_pairs_from_coupling,
    sinkhorn_barycentric_pairing,
    sinkhorn_coupling,
)
from otfm.datasets import sample_8gaussians, sample_moons


def _sample_pair(n=64, seed=0):
    key = jax.random.PRNGKey(seed)
    k0, k1 = jax.random.split(key)
    return sample_moons(k0, n), sample_8gaussians(k1, n)


def test_independent_returns_inputs_unchanged():
    x0, x1 = _sample_pair()
    y0, y1 = independent_pairing(x0, x1)
    assert np.allclose(np.asarray(y0), np.asarray(x0))
    assert np.allclose(np.asarray(y1), np.asarray(x1))


def test_hungarian_is_a_permutation_of_x1():
    x0, x1 = _sample_pair()
    y0, y1 = hungarian_pairing(x0, x1)
    # y1 should be a row-permutation of x1 -> lexicographic sort matches.
    x1_np = np.asarray(x1)
    y1_np = np.asarray(y1)
    x1_sorted = x1_np[np.lexsort(x1_np.T)]
    y1_sorted = y1_np[np.lexsort(y1_np.T)]
    assert np.allclose(x1_sorted, y1_sorted)
    assert y0.shape == x0.shape and y1.shape == x1.shape


def test_sinkhorn_plan_marginals_close_to_uniform():
    x0, x1 = _sample_pair(n=64)
    P, _ = sinkhorn_coupling(x0, x1, epsilon=0.05)
    n = x0.shape[0]
    row = jnp.sum(P, axis=1)
    col = jnp.sum(P, axis=0)
    assert jnp.allclose(row, jnp.ones(n) / n, atol=1e-3)
    assert jnp.allclose(col, jnp.ones(n) / n, atol=1e-3)


def test_barycentric_target_inside_target_cloud_bbox():
    x0, x1 = _sample_pair(n=128)
    P, _ = sinkhorn_coupling(x0, x1, epsilon=0.05)
    _, x1_bar = sinkhorn_barycentric_pairing(x0, x1, P)
    # Each barycentric target is a convex combination of x1 rows -> stays in their bbox.
    lo, hi = jnp.min(x1, axis=0), jnp.max(x1, axis=0)
    assert jnp.all(x1_bar >= lo - 1e-5)
    assert jnp.all(x1_bar <= hi + 1e-5)


def test_interpolate_plans_preserves_marginals():
    x0, x1 = _sample_pair(n=32)
    pi_ot, _ = make_hungarian_ot_plan(x0, x1)
    pi_ind = make_independent_plan(x0.shape[0])
    n = x0.shape[0]
    u = jnp.ones(n) / n
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pi = interpolate_plans(pi_ot, pi_ind, alpha)
        assert jnp.allclose(jnp.sum(pi, axis=1), u, atol=1e-6)
        assert jnp.allclose(jnp.sum(pi, axis=0), u, atol=1e-6)
        assert abs(float(jnp.sum(pi)) - 1.0) < 1e-6


def test_plan_descriptors_returns_floats():
    x0, x1 = _sample_pair(n=32)
    pi_ot, C = make_hungarian_ot_plan(x0, x1)
    desc = plan_descriptors(pi_ot, x1, C)
    for k in ("plan_entropy", "plan_sharpness", "plan_dispersion", "plan_transport_cost"):
        assert isinstance(desc[k], float)
        assert np.isfinite(desc[k])


def test_sample_pairs_from_coupling_identity_plan_returns_same_targets():
    x0, x1 = _sample_pair(n=16)
    n = x0.shape[0]
    P = jnp.eye(n)
    _, y1 = sample_pairs_from_coupling(jax.random.PRNGKey(0), x0, x1, P)
    assert jnp.allclose(y1, x1)
