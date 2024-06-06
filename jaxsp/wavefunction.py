from typing import NamedTuple
import hashlib

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxopt import GradientDescent, LBFGS
from s2fft.transforms.spherical import inverse_jax as inverse_sht

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
    error: float
    steps: int
    failed: bool

    @classmethod
    def compute_name(cls, eigenstate_library, density_params, r_min, r_fit, tol):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.asarray(eigenstate_library.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(density_params.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(r_min)).digest())
        combined.update(hashlib.md5(jnp.asarray(tol)).digest())
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
        )


_eval_library = jax.vmap(eval_radial_eigenmode, in_axes=(None, 0))
_eval_library_mult_r = jax.vmap(_eval_library, in_axes=(0, None))
_rho_in = jax.vmap(rho, in_axes=(0, None))


def _sample_gauss_legendre_integrand(eigenstate_library, density_params, r_min, r_max):
    log_rj = jnp.log(r_min) + (jnp.log(r_max) - jnp.log(r_min)) * x_i
    rho_in_log_rj = jnp.nan_to_num(
        _rho_in(jnp.exp(log_rj), density_params), nan=jnp.inf
    ) / total_mass(density_params)
    R_j2_log_rj = (
        (2 * eigenstate_library.radial_eigenmode_params.l.T + 1)
        / (4 * jnp.pi)
        * _eval_library_mult_r(
            jnp.exp(log_rj), eigenstate_library.radial_eigenmode_params
        )
        ** 2
    )

    dlogr_jac = 4 * jnp.pi * (jnp.log(r_max) - jnp.log(r_min)) * jnp.exp(3 * log_rj)
    return R_j2_log_rj, rho_in_log_rj, dlogr_jac


@jax.tree_util.Partial
def jensen_shannon_divergence(log_aj2, precomputed_quantities):
    R_j2_log_rj, rho_in_log_rj, dlogr_jac = precomputed_quantities
    rho_psi_log_rj = R_j2_log_rj @ jnp.exp(log_aj2)
    log_M = jnp.log2(0.5 * (rho_psi_log_rj + rho_in_log_rj))
    kl_pm = rho_psi_log_rj * (jnp.log2(rho_psi_log_rj) - log_M)
    kl_qm = rho_in_log_rj * (jnp.log2(rho_in_log_rj) - log_M)
    return 0.5 * (dlogr_jac * (kl_pm + kl_qm)) @ w_i


jensen_shannon_divergence.precompute = _sample_gauss_legendre_integrand


def init_wavefunction_params(
    objective_function,
    eigenstate_library,
    density_params,
    r_min,
    r_fit,
    tol,
    verbose=True,
):
    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        wavefunction_params.compute_name,
        result_shape,
        eigenstate_library,
        density_params,
        r_min,
        r_fit,
        tol,
    )

    precomputed_quantities = objective_function.precompute(
        eigenstate_library, density_params, r_min, r_fit
    )

    gd = GradientDescent(
        fun=objective_function, maxiter=100, tol=1e-3, implicit_diff=False
    )
    lbfgs = LBFGS(
        fun=objective_function,
        maxiter=200000,
        tol=tol,
        stop_if_linesearch_fails=True,
        implicit_diff=False,
        linesearch="hager-zhang",
    )

    log_aj2 = jnp.log(jnp.ones(eigenstate_library.J) / eigenstate_library.J)
    res = gd.run(log_aj2, precomputed_quantities=precomputed_quantities)
    res = lbfgs.run(res.params, precomputed_quantities=precomputed_quantities)

    params = wavefunction_params(
        eigenstate_library=eigenstate_library.name,
        density_params=density_params.name,
        aj_2=jnp.exp(res.params),
        r_min=r_min,
        r_fit=r_fit,
        total_mass=total_mass(density_params),
        distance=jensen_shannon_divergence(res.params, precomputed_quantities),
        name=name,
        error=res.state.error,
        steps=res.state.iter_num,
        failed=not res.state.failed_linesearch,
    )
    return params


def init_wavefunction_params_jensen_shannon(
    eigenstate_library, density_params, r_min, r_fit, tol, verbose=True
):
    eval_library_mult_r = jax.vmap(_eval_library, in_axes=(0, None))
    rho_in = jax.vmap(rho, in_axes=(0, None))

    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        wavefunction_params.compute_name,
        result_shape,
        eigenstate_library,
        density_params,
        r_min,
        r_fit,
        tol,
    )

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

    def jensen_shannon_divergence(log_aj2, R_j2_log_rj, rho_in_log_rj):
        rho_psi_log_rj = R_j2_log_rj @ jnp.exp(log_aj2)
        log_M = jnp.log2(0.5 * (rho_psi_log_rj + rho_in_log_rj))
        kl_pm = rho_psi_log_rj * (jnp.log2(rho_psi_log_rj) - log_M)
        kl_qm = rho_in_log_rj * (jnp.log2(rho_in_log_rj) - log_M)
        return 0.5 * (jac * (kl_pm + kl_qm)) @ w_i

    log_aj2 = jnp.log(jnp.ones(eigenstate_library.J) / eigenstate_library.J)

    gd = GradientDescent(
        fun=jensen_shannon_divergence, maxiter=100, tol=1e-3, implicit_diff=False
    )
    lbfgs = LBFGS(
        fun=jensen_shannon_divergence,
        maxiter=200000,
        tol=tol,
        stop_if_linesearch_fails=True,
        implicit_diff=False,
        linesearch="hager-zhang",
    )
    res = gd.run(log_aj2, R_j2_log_rj=R_j2_log_rj, rho_in_log_rj=rho_in_log_rj)
    res = lbfgs.run(res.params, R_j2_log_rj=R_j2_log_rj, rho_in_log_rj=rho_in_log_rj)

    params = wavefunction_params(
        eigenstate_library=eigenstate_library.name,
        density_params=density_params.name,
        aj_2=jnp.exp(res.params),
        r_min=r_min,
        r_fit=r_fit,
        total_mass=total_mass(density_params),
        distance=jensen_shannon_divergence(res.params, R_j2_log_rj, rho_in_log_rj),
        name=name,
        error=res.state.error,
        steps=res.state.iter_num,
        failed=not res.state.failed_linesearch,
    )
    return params


def rho_psi(r, wavefunction_params, eigenstate_library):
    R_j2_r = (
        (2 * eigenstate_library.radial_eigenmode_params.l.squeeze() + 1)
        * wavefunction_params.total_mass
        / (4 * jnp.pi)
        * _eval_library(r, eigenstate_library.radial_eigenmode_params) ** 2
    )
    return R_j2_r @ wavefunction_params.aj_2


def psi(r, wavefunction_params, eigenstate_lib, l_max, n_max, key):
    coef_lm = _sph_harm_coeff_lm(
        r,
        wavefunction_params,
        eigenstate_lib.radial_eigenmode_params,
        l_max,
        n_max,
        key,
    )
    return inverse_sht(coef_lm, l_max + 1, sampling="healpix", nside=(l_max + 1) // 2)


def psi_exact(r, theta, phi, wavefunction_params, eigenstate_lib, l_max, n_max, key):
    sph_harm = lambda *args: jax.scipy.special.sph_harm(*args, n_max=l_max)
    theta = jnp.atleast_1d(theta)
    phi = jnp.atleast_1d(phi)

    def Y_lm(theta, phi):
        sph_harm_mat = jnp.zeros((l_max + 1, 2 * l_max + 1), dtype=jnp.complex_)

        def populate(lm, sph_harm_mat):
            l = jnp.atleast_1d((jnp.floor(jnp.sqrt(lm)))).astype(int)
            m = jnp.atleast_1d((lm - l**2 - l)).astype(int)
            sph_harm_mat = sph_harm_mat.at[l, m + l_max].set(
                sph_harm(l, m, theta, phi)[0]
            )
            return sph_harm_mat

        return jax.lax.fori_loop(0, (l_max + 1) ** 2, populate, sph_harm_mat)

    coef_lm_r = _sph_harm_coeff_lm(
        r,
        wavefunction_params,
        eigenstate_lib.radial_eigenmode_params,
        l_max,
        n_max,
        key,
    )
    Y_lm_thetaphi = Y_lm(theta, phi)
    return jnp.einsum("lm,lm->", coef_lm_r, Y_lm_thetaphi)


def _sph_harm_coeff_lm(
    r, wavefunction_params, radial_eigenmode_params, l_max, n_max, key
):
    l_max = jax.core.concrete_or_error(
        int,
        l_max,
        "The `l_max` argument must be statically"
        "specified to use within JAX transformations.",
    )
    n_max = jax.core.concrete_or_error(
        int,
        n_max,
        "The `n_max` argument must be statically"
        "specified to use within JAX transformations.",
    )

    def abs_a_ln_R_ln(r, wavefunction_params, radial_eigenmode_params):
        R_j = _eval_library(r, radial_eigenmode_params)
        abs_a_j = jnp.sqrt(wavefunction_params.aj_2)
        aR_ln = jnp.zeros((l_max + 1, n_max))

        def populate(j, mat_ln_mat_j_radial_eigenmode_params):
            (
                mat_ln,
                mat_j,
                radial_eigenmode_params,
            ) = mat_ln_mat_j_radial_eigenmode_params
            return (
                mat_ln.at[
                    radial_eigenmode_params.l[j], radial_eigenmode_params.n[j]
                ].set(mat_j[j]),
                mat_j,
                radial_eigenmode_params,
            )

        return jax.lax.fori_loop(
            0,
            radial_eigenmode_params.l.shape[0],
            populate,
            (aR_ln, abs_a_j * R_j, radial_eigenmode_params),
        )[0]

    phi_lmn = jnp.exp(
        1.0j
        * jax.random.uniform(
            key, shape=(l_max + 1, 2 * l_max + 1, n_max), maxval=2 * jnp.pi
        )
    )
    aR_ln_r = abs_a_ln_R_ln(r, wavefunction_params, radial_eigenmode_params)
    return jnp.sqrt(wavefunction_params.total_mass) * jnp.einsum(
        "lmn,ln->lm", phi_lmn, aR_ln_r
    )
