from typing import NamedTuple
import hashlib

import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from jaxopt import GradientDescent, LBFGS

from .profiles import rho as rho, total_mass
from .chebyshev import clenshaw_curtis_weights, chebyshev_pts
from .eigenstates import eval_eigenstate
from .io_utils import hash_to_int32


class wavefunction_params(NamedTuple):
    eigenstate_library: NamedTuple
    aj_2: ArrayLike
    total_mass: float
    r_fit: float
    distance: float
    name: int

    @classmethod
    def compute_name(cls, eigenstate_library, density_params, r_fit):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.asarray(eigenstate_library.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(density_params.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(r_fit)).digest())
        return hash_to_int32(combined.hexdigest())

    def __repr__(self):
        return (
            f"wavefunction_params:"
            f"\n\tname={self.name},"
            f"\n\teigenstate_library={self.eigenstate_library.name},"
            f"\n\taj_2={[jnp.min(self.aj_2),jnp.max(self.aj_2)]},"
            f"\n\ttotal_mass={self.total_mass},"
            f"\n\tdistance={self.distance},"
        )


eval_eigenstates = jax.vmap(eval_eigenstate, in_axes=(None, 0))


def init_wavefunction_params_least_square(
    eigenstate_library, density_params, r_fit, verbose=True
):
    eval_eigenstates_mult_r = jax.vmap(eval_eigenstates, in_axes=(0, None))
    rho_in = jax.vmap(rho, in_axes=(0, None))

    N = 512
    weights = clenshaw_curtis_weights(N)
    r = (chebyshev_pts(N) + 1) * r_fit
    rho_in_r = jnp.nan_to_num(rho_in(r, density_params), nan=jnp.inf)
    R_j2_r = (
        total_mass(density_params)
        * (2 * eigenstate_library.l_of_j + 1)[jnp.newaxis, :]
        / (4 * jnp.pi)
        * eval_eigenstates_mult_r(r, eigenstate_library.R_j_params) ** 2
    )

    def square_distance(log_aj2):
        return weights @ (R_j2_r @ jnp.exp(log_aj2) / rho_in_r - 1) ** 2

    log_aj2 = jnp.log(jnp.ones(eigenstate_library.J) / eigenstate_library.J)

    gd = GradientDescent(fun=square_distance, maxiter=1000, tol=1e-7)
    lbfgs = LBFGS(fun=square_distance, maxiter=100000, tol=1e-7)
    res = gd.run(log_aj2)
    if verbose:
        print(
            f"GradientDescent stopped after {res.state.iter_num} "
            f"iterations (error = {res.state.error:.8f})"
        )
    res = lbfgs.run(res.params)
    if verbose:
        print(
            f"LBFGS stopped after {res.state.iter_num} "
            f"iterations (error = {res.state.error:.8f})"
        )

    result_shape = jax.ShapeDtypeStruct((), jnp.int32)
    name = jax.pure_callback(
        eigenstate_library.compute_name,
        result_shape,
        eigenstate_library,
        density_params,
        r_fit,
    )
    params = wavefunction_params(
        eigenstate_library=eigenstate_library,
        aj_2=jnp.exp(res.params),
        r_fit=r_fit,
        total_mass=total_mass(density_params),
        distance=square_distance(res.params),
        name=name,
    )
    return params


def rho_psi(r, wavefunction_params):
    R_j2_r = (
        (2 * wavefunction_params.eigenstate_library.l_of_j + 1)
        * wavefunction_params.total_mass
        / (4 * jnp.pi)
        * eval_eigenstates(r, wavefunction_params.eigenstate_library.R_j_params) ** 2
    )
    return R_j2_r @ wavefunction_params.aj_2
