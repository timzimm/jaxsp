from typing import NamedTuple
from functools import partial
import hashlib

import jax
import jax.numpy as jnp

from .interpolate import (
    init_1d_interpolation_params,
    eval_interp1d,
)
from .profiles import enclosed_mass
from .chebyshev import chebyshev_pts, clenshaw_curtis_weights
from .io_utils import hash_to_int32


class potential_params(NamedTuple):
    name: int
    interpolation_params: NamedTuple

    @classmethod
    def compute_name(cls, enclosed_mass_params, rmin, rmax, N):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.asarray(rmin)).digest())
        combined.update(hashlib.md5(jnp.asarray(rmax)).digest())
        combined.update(hashlib.md5(jnp.asarray(N)).digest())
        combined.update(hashlib.md5(jnp.asarray(enclosed_mass_params.name)).digest())
        return hash_to_int32(combined.hexdigest())


def init_potential_params(enclosed_mass_params, rmin, rmax, N):
    """
    Initialises a gravitational potential interpolator from the enclosed mass
    profile M(<r)
    """

    def dPhi(r):
        return -1.0 / (4 * jnp.pi) * enclosed_mass(r, enclosed_mass_params) / r**2

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def dPhi_rescaled(r, x):
        """
        As above but transformed such that integration limits are x=-1 to x=1 and
        radius r appears as parameter in the integrand
        """
        return dPhi((x + 1) / (1 - x) + r) * 2 / (1 - x) ** 2

    result_shape = jax.ShapeDtypeStruct((), jnp.int32)
    name = jax.pure_callback(
        potential_params.compute_name, result_shape, enclosed_mass_params, rmin, rmax, N
    )

    tmin = jnp.log(rmin)
    tmax = jnp.log(rmax)
    t = jnp.linspace(tmin, tmax, N)
    r = jnp.exp(t)

    # dPhi vanishes at r=inf. We therefore exclude this point to avoid the
    # singularity of the jacobian factor in dPhi_rescaled. This is fine since
    # M(r)/r**2 decays faster than 1/r^2
    xj = chebyshev_pts(N)[1:]
    wj = clenshaw_curtis_weights(N)[1:]
    potential_t = dPhi_rescaled(r, xj) @ wj
    params = init_1d_interpolation_params(tmin, t[1] - t[0], potential_t)
    return potential_params(name=name, interpolation_params=params)


def potential(r, potenial_params):
    """
    Evaluates the gravitational potential interpolator
    """
    return eval_interp1d(jnp.log(r), potenial_params.interpolation_params)
