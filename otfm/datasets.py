"""2D toy distribution samplers used as source / target in the Flow Matching experiments."""

from __future__ import annotations

import jax
import jax.numpy as jnp

EIGHT_GAUSSIANS_CENTERS = jnp.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
        [1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)],
        [-1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
        [-1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)],
    ]
)


def sample_gaussian(key, n: int, mean=(0.0, 0.0), scale: float = 1.0):
    mean = jnp.array(mean)
    x = jax.random.normal(key, (n, 2))
    return mean + scale * x


def sample_8gaussians(key, n: int, radius: float = 5.0, std: float = 0.4):
    centers = EIGHT_GAUSSIANS_CENTERS * radius
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (n,), 0, 8)
    noise = std * jax.random.normal(k2, (n, 2))
    return centers[idx] + noise


def sample_moons(key, n: int, noise: float = 0.08):
    n1 = n // 2
    n2 = n - n1
    k1, k2, k3 = jax.random.split(key, 3)

    t1 = jax.random.uniform(k1, (n1,), minval=0.0, maxval=jnp.pi)
    moon1 = jnp.stack([jnp.cos(t1), jnp.sin(t1)], axis=1)

    t2 = jax.random.uniform(k2, (n2,), minval=0.0, maxval=jnp.pi)
    moon2 = jnp.stack([1.0 - jnp.cos(t2), 1.0 - jnp.sin(t2) - 0.5], axis=1)

    x = jnp.concatenate([moon1, moon2], axis=0)
    x = x + noise * jax.random.normal(k3, x.shape)
    return 3.0 * x


def eight_gaussians_centers(radius: float = 5.0):
    return EIGHT_GAUSSIANS_CENTERS * radius
