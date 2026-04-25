"""Order-of-convergence checks for ODE solvers on a problem with a known closed form.

We use a constant velocity field v(x, t) = c so the exact flow is x(t) = x0 + c t.
On this trivial ODE all our solvers (Euler, Heun, RK4) are *exact* up to floating
point error, so we use it as a smoke test. For order-of-convergence we use a
linear vector field v(x, t) = -x whose flow is x(t) = x0 * exp(-t), and check that
the rk4 error scales as O(h^4) and Heun as O(h^2).
"""

from __future__ import annotations

import jax.numpy as jnp

from otfm.solvers import euler, heun, rk4


class _ConstField:
    """Stand-in for params: just a constant 2D velocity."""

    def __init__(self, c):
        self.c = jnp.asarray(c)


def _vf_const(params: _ConstField, x, t):
    # Match the signature used by solvers (which call vf_apply(params, x, t))
    return jnp.broadcast_to(params.c, x.shape)


def _vf_linear(params, x, t):
    return -x


def test_constant_field_is_exact(monkeypatch):
    import otfm.solvers as solvers_mod

    monkeypatch.setattr(solvers_mod, "vf_apply", _vf_const)
    params = _ConstField([1.0, -0.5])
    x0 = jnp.zeros((4, 2))
    expected = jnp.broadcast_to(jnp.array([1.0, -0.5]), (4, 2))

    for fn in (euler, heun, rk4):
        x_final = fn(params, x0, steps=8)
        assert jnp.allclose(x_final, expected, atol=1e-5), fn.__name__


def test_linear_field_convergence_orders(monkeypatch):
    import otfm.solvers as solvers_mod

    monkeypatch.setattr(solvers_mod, "vf_apply", _vf_linear)
    x0 = jnp.array([[1.0, 2.0]])
    exact = x0 * jnp.exp(-1.0)

    def err(fn, steps):
        return float(jnp.max(jnp.abs(fn(None, x0, steps=steps) - exact)))

    # Heun: error should be roughly O(h^2). Halving h reduces error by ~4x.
    e_heun_8 = err(heun, 8)
    e_heun_16 = err(heun, 16)
    assert e_heun_8 / max(e_heun_16, 1e-12) > 3.0

    # RK4: error should be O(h^4). Halving h reduces error by ~16x.
    e_rk4_8 = err(rk4, 8)
    e_rk4_16 = err(rk4, 16)
    assert e_rk4_8 / max(e_rk4_16, 1e-14) > 10.0


def test_solver_returns_trajectory_of_correct_length(monkeypatch):
    import otfm.solvers as solvers_mod

    monkeypatch.setattr(solvers_mod, "vf_apply", _vf_linear)
    x0 = jnp.ones((3, 2))
    traj = euler(None, x0, steps=10, return_traj=True)
    assert isinstance(traj, list)
    assert len(traj) == 11
    assert all(t.shape == x0.shape for t in traj)
