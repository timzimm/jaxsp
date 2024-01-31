from functools import partial

import jax
import jax.numpy as jnp

from .constants import GN
from .interpolate import init_1d_interpolation_params, eval_interp1d
from .chebyshev import chebyshev_pts, clenshaw_curtis_weights


def init_potential_params(enclosed_mass, rmin, rmax, N):
    """
    Initialises a gravitational potential interpolator from the enclosed mass
    profile M(<r)
    """

    def dPhi(r):
        return -GN.value * enclosed_mass(r) / r**2

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def dPhi_rescaled(r, x):
        """
        As above but transformed such that integration limits are x=-1 to x=1 and
        radius r appears as parameter in the integrand
        """
        return dPhi((x + 1) / (1 - x) + r) * 2 / (1 - x) ** 2

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
    return init_1d_interpolation_params(tmin, t[1] - t[0], potential_t)


def eval_potential(r, potenial_params):
    """
    Evaluates the gravitational potential interpolator
    """
    return eval_interp1d(jnp.log(r), potenial_params)
