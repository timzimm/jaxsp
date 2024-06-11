import jax
import jax.numpy as jnp
from jaxopt import Bisection

from .special import lambertw
from .utils import quad
from .potential import potential as gravitational_potential


_LOG_LINEAR_GRID_A = 1
_LOG_LINEAR_GRID_B = 10


@jax.jit
def x_of_r(r):
    """
    Transformation from loglinear (non-uniform) r to linear (uniform) x
    """
    return _LOG_LINEAR_GRID_A * r + jax.scipy.special.xlogy(_LOG_LINEAR_GRID_B, r)


@jax.jit
def r_of_x(x):
    """
    Transformation from linear (uniform) x to log-linear (non-uniform) r
    """

    def lambertw_exp_asymptotic(x, a, b):
        """
        Asymptotic approximation for W(a/b * exp(x/b)) for large x
        (which would overflow due to exp())

        See:
            https://en.wikipedia.org/wiki/Lambert_W_function#Asymptotic_expansions
        """
        L1 = jnp.log(a / b) + x / b
        L2 = jnp.log(L1)
        return L1 - L2 + L2 / L1 + L2 * (-2 + L2) / (2 * L1**2)

    a = _LOG_LINEAR_GRID_A
    b = _LOG_LINEAR_GRID_B

    if a == 0:
        return jnp.exp(x / b)
    if b == 0:
        return x / a
    return jax.lax.cond(
        x < 100,
        lambda x: b / a * lambertw(a / b * jnp.exp(x / b)),
        lambda x: b / a * lambertw_exp_asymptotic(x, a, b),
        x,
    )


def q(
    x,
    l,
    V0,
    potential_params,
    potential=gravitational_potential,
):
    """
    Sturm Liouville q-function of log-linear transformed radial Schroedinger equation.
    The full form SL-problem reads:
                    - t'' + q(x) t(x) = 2 E w(x) t(x)
    See:
        https://doi.org/10.1016/j.hedp.2023.101042
    """
    a = _LOG_LINEAR_GRID_A
    b = _LOG_LINEAR_GRID_B
    r = r_of_x(x)
    return (
        1.0
        / (b + a * r) ** 2
        * (
            l * (l + 1)
            + 2 * r**2 * (potential(r, potential_params) + V0)
            + b * (b + 4 * a * r) / (4 * (b + a * r) ** 2)
        )
    )


def w(x):
    """
    Sturm Liouville w-function of log-linear transformed radial Schroedinger equation
    The full form SL-problem reads:
                    - t'' + q(x) t(x) = 2 E w(x) t(x)
    See:
        https://doi.org/10.1016/j.hedp.2023.101042
    """
    a = _LOG_LINEAR_GRID_A
    b = _LOG_LINEAR_GRID_B
    r = r_of_x(x)
    return 2 * r**2 / (b + a * r) ** 2


def V_effective(r, l, potential_params, potential=gravitational_potential):
    return 0.5 * l * (l + 1) / r**2 + potential(r, potential_params)


# def bisect(obj, lower, upper, iterations=100, tol=1e-8):
#     objective = jax.jit(obj)

#     def bracket(i_obj_lower_obj_upper_lower_upper):
#         i, obj_lower, obj_upper, lower, upper = i_obj_lower_obj_upper_lower_upper
#         mid = 0.5 * (lower + upper)
#         obj_mid = objective(mid)
#         cond = jnp.sign(obj_lower) == jnp.sign(obj_mid)
#         upper = jnp.where(cond, upper, mid)
#         obj_upper = jnp.where(cond, obj_upper, obj_mid)
#         lower = jnp.where(cond, mid, lower)
#         obj_lower = jnp.where(cond, obj_mid, obj_lower)

#         return (i + 1, obj_lower, obj_upper, lower, upper)

#     def not_converged(i_obj_lower_obj_upper_lower_upper):
#         i, _, _, lower, upper = i_obj_lower_obj_upper_lower_upper
#         return jnp.logical_and(i < iterations, jnp.abs(upper - lower) > tol)

#     i, _, _, l, u = jax.lax.while_loop(
#         not_converged, bracket, (0, -18.0, objective(upper), lower, upper)
#     )
#     return 0.5 * (l + u)


@jax.jit
def wkb_estimate_of_rmax(r, l, potential_params, potential=gravitational_potential):
    def wkb_condition_Veff(r_lower, r_upper, Emax):
        return jnp.nan_to_num(
            (
                jnp.sqrt(2)
                * quad(
                    jax.vmap(
                        lambda r: jnp.sqrt(
                            V_effective(r, l, potential_params, potential=potential)
                            - Emax
                        )
                    ),
                    r_lower,
                    r_upper,
                )
                - 18
            ),
            nan=-18.0,
        )

    Emax = potential(r, potential_params)
    bisec = Bisection(
        optimality_fun=lambda logr: wkb_condition_Veff(r, jnp.exp(logr), Emax),
        lower=jnp.log(r),
        upper=jnp.log(10 * r),
        check_bracket=False,
    )
    logrmax = bisec.run().params

    return jnp.exp(logrmax)
