from typing import NamedTuple
from jax.typing import ArrayLike

import jax.numpy as jnp
from jax.lax import dynamic_slice


class interpolation_params(NamedTuple):
    """
    Parameters specifying an interpolant based on local 4th order Taylor
    expansion on a uniform grid.
    This is a reduced version of dbstein/fast_interp
    """

    f: ArrayLike
    a: float
    dx: float
    lb: float
    ub: float


class piecewise_constant_interpolation_params(NamedTuple):
    """
    Parameters specifying a piecewise constant interpolant on a uniform
    grid.
    """

    f_i: ArrayLike
    x0: float
    dx: float


def _extrapolate1d_x(f):
    return jnp.concatenate(
        (
            jnp.array([4 * f[0] - 6 * f[1] + 4 * f[2] - f[3]]),
            f,
            jnp.array([4 * f[-1] - 6 * f[-2] + 4 * f[-3] - f[-4]]),
        )
    )


def init_1d_interpolation_params(a, dx, f):
    """
    Initializes parameters for a smooth function f on [a, a + f.shape[0] * dx].
    Interpolation is accurate up to these boundaries.
    """
    N = f.shape[0]
    f = _extrapolate1d_x(f)
    lb, ub = a, a + (N - 1) * dx
    ub -= ub * jnp.finfo(ub.dtype).eps
    return interpolation_params(a=a, dx=dx, f=f, lb=lb, ub=ub)


def eval_interp1d(x, interpolation_params):
    # Taylor coefficients
    A = jnp.array([-1.0 / 16, 9.0 / 16, 9.0 / 16, -1.0 / 16])
    B = jnp.array([1.0 / 24, -9.0 / 8, 9.0 / 8, -1.0 / 24])
    C = jnp.array([1.0 / 4, -1.0 / 4, -1.0 / 4, 1.0 / 4])
    D = jnp.array([-1.0 / 6, 1.0 / 2, -1.0 / 2, 1.0 / 6])

    x = (
        jnp.minimum(jnp.maximum(x, interpolation_params.lb), interpolation_params.ub)
        - interpolation_params.a
    )
    ix = jnp.atleast_1d(jnp.array(x // interpolation_params.dx, int))
    ratx = x / interpolation_params.dx - (ix + 0.5)
    asx = A + ratx * (B + ratx * (C + ratx * D))
    return jnp.dot(dynamic_slice(interpolation_params.f, ix, (4,)), asx)


def eval_piecewise_constant_interpolation_deprecated(
    x, piecewise_constant_interpolation_params
):
    params = piecewise_constant_interpolation_params
    x = jnp.clip(
        x,
        params.x_i[0],
        params.x_i[-1],
    )
    i = jnp.searchsorted(params.x_i, x, side="right")
    return params.f_i[i - 1]


def eval_piecewise_constant_interpolation(x, piecewise_constant_interpolation_params):
    params = piecewise_constant_interpolation_params
    x = (
        jnp.clip(
            x,
            params.x0,
            params.x0 + params.f_i.shape[0] * params.dx,
        )
        - params.x0
    )
    ix = (x // params.dx).astype(int)
    return params.f_i[ix]
