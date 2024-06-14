import jax
import jax.numpy as jnp
from jaxopt import Bisection

from .special import lambertw
from .utils import quad
from .potential import potential as gravitational_potential


def r_of_x(x, a, b):
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

    def lambertw_grid(x, a, b):
        return jax.lax.cond(
            x / b < 100,
            lambda x: b / a * lambertw(a / b * jnp.exp(x / b)),
            lambda x: b / a * lambertw_exp_asymptotic(x, a, b),
            x,
        )

    def exp_grid(x, a, b):
        return jnp.exp(x / b)

    def linear_grid(x, a, b):
        return x / a

    def linear_or_lambertw(x, a, b):
        return jax.lax.cond(b == 0, linear_grid, lambertw_grid, x, a, b)

    return jax.lax.cond(a == 0, exp_grid, linear_or_lambertw, x, a, b)


def x_of_r(r, a, b):
    """
    Transformation from loglinear (non-uniform) r to linear (uniform) x
    """
    return a * r + jax.scipy.special.xlogy(b, r)


def q(
    x,
    a,
    b,
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
    r = r_of_x(x, a, b)
    return (
        1.0
        / (b + a * r) ** 2
        * (
            l * (l + 1)
            + 2 * r**2 * (potential(r, potential_params) + V0)
            + b * (b + 4 * a * r) / (4 * (b + a * r) ** 2)
        )
    )


def w(x, a, b):
    """
    Sturm Liouville w-function of log-linear transformed radial Schroedinger equation
    The full form SL-problem reads:
                    - t'' + q(x) t(x) = 2 E w(x) t(x)
    See:
        https://doi.org/10.1016/j.hedp.2023.101042
    """
    r = r_of_x(x, a, b)
    return 2 * r**2 / (b + a * r) ** 2


def V_effective(r, l, potential_params, potential=gravitational_potential):
    return 0.5 * l * (l + 1) / r**2 + potential(r, potential_params)


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
