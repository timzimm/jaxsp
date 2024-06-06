from typing import NamedTuple
from functools import partial
import hashlib

import jax
import jax.numpy as jnp
from jaxopt import Bisection

from .interpolate import (
    init_1d_interpolation_params,
    eval_interp1d,
)
from .profiles import enclosed_mass
from .io_utils import hash_to_int64
from .utils import quad, leggauss_unit_interval


class potential_params(NamedTuple):
    name: int
    density_params: int
    interpolation_params: NamedTuple

    @classmethod
    def compute_name(cls, density_params, rmin, rmax, N):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.asarray(density_params.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(rmin)).digest())
        # combined.update(hashlib.md5(jnp.asarray(rmax)).digest())
        combined.update(hashlib.md5(jnp.asarray(N)).digest())
        return hash_to_int64(combined.hexdigest())

    def __repr__(self):
        return (
            f"potential_params:"
            f"\n\tname={self.name},"
            f"\n\tdensity_params={self.density_params},"
            f"\n\tN={self.interpolation_params.f.shape[0]-2},"
            f"\n\trmin={jnp.exp(self.interpolation_params.a)},"
            f"\n\trmax={jnp.exp(self.interpolation_params.a+self.interpolation_params.dx*(self.interpolation_params.f.shape[0]-3))},"
        )


def init_potential_params(density_params, rmin, rmax, N):
    """
    Initialises a gravitational potential interpolator from the enclosed mass
    profile M(<r)
    """

    def dPhi(r):
        return -1.0 / (4 * jnp.pi) * enclosed_mass(r, density_params) / r**2

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def dPhi_rescaled(r, x):
        """
        As above but transformed such that integration limits are x=0 to x=1 and
        radius r appears as parameter in the integrand
        """
        return dPhi(x / (1 - x) + r) / (1 - x) ** 2

    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        potential_params.compute_name, result_shape, density_params, rmin, rmax, N
    )

    tmin = jnp.log(rmin)
    tmax = jnp.log(rmax)
    t = jnp.linspace(tmin, tmax, N)
    r = jnp.exp(t)

    # dPhi vanishes at r=inf. We therefore exclude this point to avoid the
    # singularity of the jacobian factor in dPhi_rescaled. This is fine since
    # M(r)/r**2 decays faster than 1/r^2
    xj, wj = leggauss_unit_interval(128)
    potential_t = dPhi_rescaled(r, xj) @ wj
    params = init_1d_interpolation_params(tmin, t[1] - t[0], potential_t)
    return potential_params(
        name=name, density_params=density_params.name, interpolation_params=params
    )


def potential(r, potential_params):
    """
    Evaluates the gravitational potential interpolator
    """
    return eval_interp1d(jnp.log(r), potential_params.interpolation_params)


def V_effective(r, l, potential_params):
    return 0.5 * l * (l + 1) / r**2 + potential(r, potential_params)


@jax.jit
def wkb_estimate_of_rmax(r, l, potential_params):
    def wkb_condition_Veff(r_lower, r_upper, Emax):
        return (
            jnp.sqrt(2)
            * quad(
                jax.vmap(
                    lambda r: jnp.sqrt(V_effective(r, l, potential_params) - Emax)
                ),
                r_lower,
                r_upper,
            )
            - 18
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
