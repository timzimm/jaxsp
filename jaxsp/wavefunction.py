from collections import namedtuple
import matplotlib.pyplot as plt

from jax import vmap, random, lax, jit, grad, nn
from jax.scipy.special import erf
from jaxopt._src.tree_util import tree_map
import jax.numpy as jnp

from jaxopt import GradientDescent, LBFGS

from .interpolate import eval_interp1d

wavefunction_params = namedtuple(
    "wavefunction_params",
    ["M_fit", "r_fit", "aj_2", "eigenstate_library"],
)

eval_mult_states = vmap(eval_interp1d, in_axes=(None, 0))
eval_mult_states_mult_x = vmap(eval_mult_states, in_axes=(0, None))


def enclosed_mass_from_gaussian_noise_model(key, mass, N, r_fit, method="equal_mass"):
    if method == "equal_mass":

        @jit
        def density(rkpc):
            return grad(lambda r: mass(r)[0])(rkpc) / (4 * jnp.pi * rkpc**2)

        # Find equal mass shell grid
        mass_inside_each_shell = mass(r_fit)[0] / N

        def next_point(r_ip1, x):
            r_i = r_ip1 - mass_inside_each_shell / (
                4 * jnp.pi * r_ip1**2 * density(r_ip1)
            )
            return r_i, r_i

        _, rkpc = lax.scan(
            next_point,
            r_fit,
            None,
            length=N,
            reverse=True,
        )
        rkpc = rkpc[rkpc > 0]
        N_fit = rkpc.shape[0]
    else:
        N_fit = N
        rkpc = jnp.logspace(jnp.log10(mass.rmin), jnp.log10(r_fit), N_fit)

    M_r_mean, M_r_sigma = mass(rkpc)
    M_r_realisation = M_r_mean + random.normal(key, shape=(N_fit,)) * M_r_sigma
    return rkpc, M_r_realisation, M_r_sigma


def init_wavefunction_params_least_square(
    eigenstate_library, mass, N_fit, r_fit, seed, verbose=True
):
    key_sampling = random.PRNGKey(seed)
    rkpc, M_r_realisation, M_r_sigma = enclosed_mass_from_gaussian_noise_model(
        key_sampling, mass, N_fit, r_fit, method=None
    )
    N_fit = rkpc.shape[0]

    M_fit = mass(r_fit)[0]

    M_r_realisation = M_fit - M_r_realisation
    M_r_psi = M_psi(
        rkpc,
        wavefunction_params(
            M_fit=M_fit,
            eigenstate_library=eigenstate_library,
            r_fit=rkpc,
            aj_2=jnp.eye(eigenstate_library.J),
        ),
    )

    def least_square_error(aj_2):
        return jnp.mean(
            1 / (2 * M_r_sigma**2) * (M_r_realisation - M_r_psi @ aj_2) ** 2
        )

    optimizer = LBFGS(fun=least_square_error, maxiter=1000, tol=1e-7)
    res = optimizer.run(jnp.ones_like(eigenstate_library.E_j))
    if verbose:
        print(
            f"Optimization stopped after {res.state.iter_num} "
            f"iterations (error = {res.state.error:.8f})"
        )
    aj_2 = res.params

    params = wavefunction_params(
        M_fit=M_fit,
        eigenstate_library=eigenstate_library,
        r_fit=rkpc,
        aj_2=aj_2,
    )
    if verbose:
        return params, {
            "rkpc": rkpc,
            "M_r_realisation": M_r_realisation,
            "M_r_sigma": M_r_sigma,
        }
    else:
        return params


def rho(rkpc, wavefunction_params):
    r_j2_r = (
        (2 * wavefunction_params.eigenstate_library.l_of_j[jnp.newaxis, :] + 1)
        * wavefunction_params.enclosed_mass
        / (4 * jnp.pi)
        * eval_mult_states_mult_x(
            rkpc, wavefunction_params.eigenstate_library.r_j_params
        )
        ** 2
    )
    return r_j2_r @ wavefunction_params.aj_2


def M_psi(rkpc, wavefunction_params):
    dr = jnp.diff(rkpc)
    M_psi_integrand = (
        (2 * wavefunction_params.eigenstate_library.l_of_j[jnp.newaxis, :] + 1)
        * wavefunction_params.M_fit
        * rkpc[:, jnp.newaxis] ** 2
        * eval_mult_states_mult_x(
            rkpc, wavefunction_params.eigenstate_library.R_j_params
        )
        ** 2
    )
    summand = 1 / 2 * (M_psi_integrand[1:] + M_psi_integrand[:-1]) * dr[:, jnp.newaxis]
    M = jnp.cumsum(summand[::-1], axis=0)[::-1]
    std = jnp.std(M, axis=0, keepdims=True)
    M_r_psi = jnp.pad(M / std, ((0, 1), (0, 0)))
    return M_r_psi @ wavefunction_params.aj_2
