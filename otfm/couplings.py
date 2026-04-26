"""Coupling / pairing strategies between source and target empirical distributions.

Each function returns a pair (x0_paired, x1_paired) of equal length so that the
i-th source sample is paired with the i-th target sample for Flow Matching training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from scipy.optimize import linear_sum_assignment


def independent_pairing(x0, x1):
    """Pair by index — equivalent to π = ν0 ⊗ ν1 when x0, x1 are i.i.d."""
    return x0, x1


def sinkhorn_coupling(x0, x1, epsilon: float = 0.1):
    """Solve entropic OT and return the transport matrix P and the solver output."""
    geom = pointcloud.PointCloud(x0, x1, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    return out.matrix, out


def sample_pairs_from_coupling(key, x0, x1, P):
    """Stochastic pairing: for each row i of P, sample j ~ P[i, :] / sum_j P[i, :]."""
    row_sums = jnp.sum(P, axis=1, keepdims=True) + 1e-12
    probs = P / row_sums

    n = x0.shape[0]
    keys = jax.random.split(key, n)

    sampled_indices = jnp.array(
        [jax.random.choice(keys[i], x1.shape[0], p=probs[i]) for i in range(n)]
    )
    paired_x1 = x1[sampled_indices]
    return x0, paired_x1


def hungarian_pairing(x0, x1):
    """Exact discrete OT via the Hungarian algorithm on squared Euclidean cost."""
    C = jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=-1)
    _, col_ind = linear_sum_assignment(np.asarray(C))
    return x0, x1[jnp.array(col_ind)]


def sinkhorn_barycentric_pairing(x0, x1, P):
    """Deterministic pairing via the barycentric projection y_i = sum_j P_ij x1_j / sum_j P_ij."""
    row_sums = jnp.sum(P, axis=1, keepdims=True) + 1e-12
    W = P / row_sums
    x1_bar = W @ x1
    return x0, x1_bar


def make_hungarian_ot_plan(x0, x1):
    """Return the Hungarian transport plan as a (N,N) matrix with mass 1/N on the assignment."""
    C = jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(np.asarray(C))
    N = x0.shape[0]
    pi_ot = np.zeros((N, N), dtype=np.float64)
    pi_ot[row_ind, col_ind] = 1.0 / N
    return jnp.array(pi_ot), C


def make_independent_plan(N: int):
    """Independent (product) plan u u^T with uniform marginals."""
    u = jnp.ones(N) / N
    return jnp.outer(u, u)


def interpolate_plans(pi_a, pi_b, alpha: float):
    """Linear interpolation between two plans on the simplex of plans."""
    return (1.0 - alpha) * pi_a + alpha * pi_b


def plan_descriptors(pi, x1, C):
    """Mean per-row entropy / sharpness / dispersion of a plan, plus its transport cost."""
    rowp = pi / (jnp.sum(pi, axis=1, keepdims=True) + 1e-12)
    H = -jnp.sum(rowp * jnp.log(rowp + 1e-12), axis=1)
    S = jnp.max(rowp, axis=1)
    xbar = rowp @ x1
    D = jnp.sum(rowp * jnp.sum((x1[None, :, :] - xbar[:, None, :]) ** 2, axis=-1), axis=1)
    return {
        "plan_entropy": float(jnp.mean(H)),
        "plan_sharpness": float(jnp.mean(S)),
        "plan_dispersion": float(jnp.mean(D)),
        "plan_transport_cost": float(jnp.sum(pi * C)),
    }
