import logging
from typing import NamedTuple
import hashlib

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.special import logsumexp
from jaxopt import GradientDescent, LBFGS

from .profiles import rho as rho, total_mass
from .eigenstates_pruess import eval_radial_eigenmode
from .io_utils import hash_to_int64
from .utils import _glx128 as x_i, _glw128 as w_i


class wavefunction_params(NamedTuple):
    name: int
    eigenstate_library: int
    density_params: int
    aj_2: ArrayLike
    total_mass: float
    r_min: float
    r_fit: float
    distance: float
    converged: bool

    @classmethod
    def compute_name(cls, eigenstate_library, density_params, r_min, r_fit):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.asarray(eigenstate_library.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(density_params.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(r_min)).digest())
        combined.update(hashlib.md5(jnp.asarray(r_fit)).digest())
        return hash_to_int64(combined.hexdigest())

    def __repr__(self):
        return (
            f"wavefunction_params:"
            f"\n\tname={self.name},"
            f"\n\teigenstate_library={self.eigenstate_library},"
            f"\n\tdensity_params={self.density_params},"
            f"\n\taj_2={[jnp.min(self.aj_2).item(),jnp.max(self.aj_2).item()]},"
            f"\n\ttotal_mass={self.total_mass},"
            f"\n\tr_min={self.r_min},"
            f"\n\tr_max={self.r_fit},"
            f"\n\tdistance={self.distance},"
            f"\n\tconverged={self.converged},"
        )


logger = logging.getLogger(__name__)
eval_library = jax.vmap(eval_radial_eigenmode, in_axes=(None, 0))


def init_wavefunction_params_jensen_shannon(
    eigenstate_library, density_params, r_min, r_fit, verbose=True
):
    eval_library_mult_r = jax.vmap(eval_library, in_axes=(0, None))
    rho_in = jax.vmap(rho, in_axes=(0, None))

    log_rj = jnp.log(r_min) + (jnp.log(r_fit) - jnp.log(r_min)) * x_i
    rho_in_log_rj = jnp.nan_to_num(
        rho_in(jnp.exp(log_rj), density_params), nan=jnp.inf
    ) / total_mass(density_params)
    R_j2_log_rj = (
        (2 * eigenstate_library.radial_eigenmode_params.l.T + 1)
        / (4 * jnp.pi)
        * eval_library_mult_r(
            jnp.exp(log_rj), eigenstate_library.radial_eigenmode_params
        )
        ** 2
    )

    jac = 4 * jnp.pi * (jnp.log(r_fit) - jnp.log(r_min)) * jnp.exp(3 * log_rj)

    def jensen_shannon_divergence(log_aj2):
        rho_psi_log_rj = R_j2_log_rj @ jnp.exp(log_aj2)
        log_M = jnp.log2(0.5 * (rho_psi_log_rj + rho_in_log_rj))
        kl_pm = rho_psi_log_rj * (jnp.log2(rho_psi_log_rj) - log_M)
        kl_qm = rho_in_log_rj * (jnp.log2(rho_in_log_rj) - log_M)
        return 0.5 * (jac * (kl_pm + kl_qm)) @ w_i

    log_aj2 = jnp.log(jnp.ones(eigenstate_library.J) / eigenstate_library.J)

    gd = GradientDescent(
        fun=jensen_shannon_divergence,
        maxiter=100,
        tol=1e-3,
    )
    lbfgs = LBFGS(
        fun=jensen_shannon_divergence,
        maxiter=100000,
        tol=1e-6,
        stop_if_linesearch_fails=True,
        linesearch="hager-zhang",
    )
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

    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        eigenstate_library.compute_name,
        result_shape,
        eigenstate_library,
        density_params,
        r_min,
        r_fit,
    )
    params = wavefunction_params(
        eigenstate_library=eigenstate_library.name,
        density_params=density_params.name,
        aj_2=jnp.exp(res.params),
        r_min=r_min,
        r_fit=r_fit,
        total_mass=total_mass(density_params),
        distance=jensen_shannon_divergence(res.params),
        name=name,
        converged=not res.state.failed_linesearch and res.state.error <= 1e-6,
    )
    return params


def init_wavefunction_params_square_distance(
    eigenstate_library, density_params, r_min, r_fit, verbose=True
):
    eval_library_mult_r = jax.vmap(eval_library, in_axes=(0, None))
    rho_in = jax.vmap(rho, in_axes=(0, None))

    log_weights = jnp.log(w_i)
    log_rj = jnp.log(r_min) + (jnp.log(r_fit) - jnp.log(r_min)) * x_i
    rho_in_log_rj = jnp.nan_to_num(rho_in(jnp.exp(log_rj), density_params), nan=jnp.inf)
    R_j2_log_rj = (
        total_mass(density_params)
        * (2 * eigenstate_library.radial_eigenmode_params.l.squeeze() + 1)[
            jnp.newaxis, :
        ]
        / (4 * jnp.pi)
        * eval_library_mult_r(
            jnp.exp(log_rj), eigenstate_library.radial_eigenmode_params.eigenmode_params
        )
        ** 2
    )

    def log_square_distance(log_aj2):
        log_integrand = log_rj + jnp.log(
            ((R_j2_log_rj @ jnp.exp(log_aj2)) / rho_in_log_rj - 1) ** 2
        )
        return jnp.log(jnp.log(r_fit) - jnp.log(r_min)) + logsumexp(
            log_weights + log_integrand
        )

    log_aj2 = jnp.log(jnp.ones(eigenstate_library.J) / eigenstate_library.J)

    gd = GradientDescent(
        fun=log_square_distance,
        maxiter=100,
        tol=1e-3,
    )
    lbfgs = LBFGS(
        fun=log_square_distance,
        maxiter=100000,
        tol=1e-6,
        stop_if_linesearch_fails=True,
        linesearch="hager-zhang",
    )
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

    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        eigenstate_library.compute_name,
        result_shape,
        eigenstate_library,
        density_params,
        r_min,
        r_fit,
    )
    params = wavefunction_params(
        eigenstate_library=eigenstate_library.name,
        density_params=density_params.name,
        aj_2=jnp.exp(res.params),
        r_min=r_min,
        r_fit=r_fit,
        total_mass=total_mass(density_params),
        distance=jnp.exp(log_square_distance(res.params)),
        name=name,
    )
    return params


def rho_psi(r, wavefunction_params, eigenstate_library):
    R_j2_r = (
        (2 * eigenstate_library.radial_eigenmode_params.l.squeeze() + 1)
        * wavefunction_params.total_mass
        / (4 * jnp.pi)
        * eval_library(r, eigenstate_library.radial_eigenmode_params) ** 2
    )
    return R_j2_r @ wavefunction_params.aj_2
