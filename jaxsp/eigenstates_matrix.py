from typing import NamedTuple
import hashlib

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy.linalg import eigh_tridiagonal
from jaxopt import Broyden, ProjectedGradient
from jaxopt.projection import projection_box

from .interpolate import init_1d_interpolation_params, eval_interp1d
from .potential import potential as gravitational_potential
from .utils import quad
from .io_utils import hash_to_int64

# from .chebyshev import (
#     chebyshev_d2x,
#     clenshaw_curtis_weights,
#     init_chebyshev_params_from_samples,
#     eval_chebyshev_polynomial,
# )
from .special import lambertw
from .radial_schroedinger import V_effective, wkb_estimate_of_rmax

import logging

logger = logging.getLogger(__name__)


class eigenstate_library_matrix(NamedTuple):
    """
    Parameters specifying the eigenstate_library
    """

    name: int
    potential_params: int
    R_j_params: NamedTuple
    E_j: ArrayLike
    l_of_j: ArrayLike
    n_of_j: ArrayLike

    @classmethod
    def compute_name(cls, potential_params, r_ta, N):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.asarray(potential_params.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(r_ta)).digest())
        combined.update(hashlib.md5(jnp.asarray(N)).digest())
        return hash_to_int64(combined.hexdigest())

    def j_of_nl(self, n, l):
        n = int(n)
        l_index = jnp.searchsorted(self.l_of_j, l)
        n_index = jnp.argmax(self.n_of_j[l_index:] == n)
        return l_index + n_index

    @property
    def J(self):
        return self.E_j.shape[0]

    def __repr__(self):
        return (
            f"wavefunction_params:"
            f"\n\tname={self.name},"
            f"\n\tpotential_params={self.potential_params},"
            f"\n\tJ={self.J},"
            f"\n\tlmax={jnp.max(self.l_of_j)},"
            f"\n\tnmax={jnp.max(self.n_of_j)},"
            f"\n\tEmin={jnp.min(self.E_j)},"
            f"\n\tEmax={jnp.max(self.E_j)},"
        )


init_mult_spline_params = jax.vmap(init_1d_interpolation_params)


@jax.jit
def minimum_of_effective_potential(
    r_ta, l, potential_params, potential=gravitational_potential
):
    V0 = potential(0.0, potential_params)
    pg = ProjectedGradient(
        fun=lambda logr: jnp.log(
            V_effective(jnp.exp(logr), l, potential_params, potential=potential) - V0
        ),
        projection=projection_box,
        tol=1e-8,
    )
    # logrmin = jax.lax.cond(
    #     l == 0,
    #     lambda _: jnp.log(1e-15),
    #     lambda _: pg.run(
    #         jnp.log(1e-1),
    #         bounds=(jnp.log(1e-15), jnp.log(r_ta)),
    #     ).params,
    #     None,
    # )
    logrmin = pg.run(
        jnp.log(1e-1), hyperparams_proj=(jnp.log(1e-15), jnp.log(r_ta))
    ).params
    return jnp.exp(logrmin)


@jax.jit
def lower_turning_point(
    upper_r_ta, l, potential_params, potential=gravitational_potential
):
    pg = ProjectedGradient(
        fun=lambda logr: jnp.log(
            (V_effective(jnp.exp(logr), l, potential_params, potential=potential) - E)
            ** 2
        ),
        projection=projection_box,
    )
    E = potential(upper_r_ta, potential_params)
    rmin = minimum_of_effective_potential(upper_r_ta, l, potential_params)
    log_lower_r_ta = jax.lax.cond(
        l == 0,
        lambda _: jnp.log(1e-15),
        lambda _: pg.run(
            jnp.log(rmin / 2),
            hyperparams_proj=(jnp.log(1e-15), jnp.log(rmin)),
        ).params,
        None,
    )
    return jnp.exp(log_lower_r_ta)


@jax.jit
def sommerfeld_estimate_of_nmax(
    r_ta, potential_params, potential=gravitational_potential
):
    _V = jax.vmap(potential, in_axes=(0, None))

    def p_effective(r, E):
        return jnp.sqrt(2 * (E - _V(r, potential_params)))

    E = potential(r_ta, potential_params)
    return (1.0 / jnp.pi * quad(lambda r: p_effective(r, E), 0, r_ta) + 1).astype(int)


@jax.jit
def wkb_estimate_of_rmin(r_ta, l, potential_params, potential=gravitational_potential):
    def wkb_condition_Veff(r_lower, r_upper, Emax):
        return (
            jnp.sqrt(2)
            * quad(
                jax.vmap(
                    lambda r: jnp.sqrt(
                        V_effective(r, l, potential_params, potential=potential) - Emax
                    )
                ),
                r_lower,
                r_upper,
            )
            - 36
        )

    Emax = potential(r_ta, potential_params)
    broyd = Broyden(
        fun=lambda logr: wkb_condition_Veff(jnp.exp(logr), r_ta, Emax),
    )
    logrmin = jax.lax.cond(
        l == 0,
        lambda _: jnp.log(r_ta),
        lambda _: broyd.run(jnp.array(jnp.log(r_ta))).params,
        None,
    )
    logrmin = jnp.nan_to_num(logrmin, nan=jnp.log(1e-15))

    return jnp.exp(logrmin)


@jax.jit
def lmax_from_effective_potential(
    r_ta, potential_params, potential=gravitational_potential
):
    E_max = potential(r_ta, potential_params)
    return jax.lax.while_loop(
        lambda l: V_effective(
            minimum_of_effective_potential(
                r_ta, l, potential_params, potential=potential
            ),
            l,
            potential_params,
            potential=potential,
        )
        < E_max,
        lambda l: l + 1,
        1,
    )


def check_mode_heath(E_n, E_min, E_max):
    if E_n.shape[0] == 0:
        Exception(f"No modes inside [{E_min:.2f}, {E_max:2f}]")
    if np.any(E_n.imag > 1e-10 * E_n.real):
        Exception("Eigenvalue with significant imaginary part found")
    if np.any(np.unique(E_n, return_counts=True)[1] > 1):
        Exception("Degeneracy detected. This is impossible in 1D.")


# def init_eigenstate_library_cheb(
#     potential_params, r_ta, N, potential=gravitational_potential
# ):
#     @jax.jit
#     @jax.vmap
#     def init_mult_chebyshev_modes(f_j, rmax):
#         f_j = jnp.pad(f_j, (1, 1))
#         return init_chebyshev_params_from_samples(f_j, 0, rmax)

#     result_shape = jax.ShapeDtypeStruct((), jnp.int64)
#     name = jax.pure_callback(
#         eigenstate_library_matrix.compute_name, result_shape, potential_params, r_ta, N
#     )

#     # Discetized radial eigenstate
#     u_j_r = []
#     rmax_j = []

#     # Eigenvalues
#     E_j = []

#     # Quantum numbers
#     l_of_j = []
#     n_of_j = []

#     if sommerfeld_estimate_of_nmax(r_ta, potential_params, potential=potential) > N:
#         logger.info("N is too small to construct the entire library!")
#     E_max = potential(r_ta, potential_params)

#     # Construct chebyshev derivative operators in normalized x space
#     x, d2_dx = chebyshev_d2x(N)
#     # Impose Dirichlet boundary condition
#     x = x[1:-1]
#     d2_dx = d2_dx[1:-1, 1:-1]
#     # Clenshaw-Curis weights for Dirichlet BC ( f(-1) = f(1) = 0 )
#     weights = jnp.asarray(clenshaw_curtis_weights(N)[1:-1])

#     for l in range(lmax(r_ta, potential_params)):
#         print(l)
#         E_min_l = V_effective(
#             minimum_of_effective_potential(r_ta, l, potential_params),
#             l,
#             potential_params,
#         )
#         rmax = wkb_estimate_of_rmax(r_ta, l, potential_params)
#         r = (x + 1) * rmax / 2

#         d2_dr = 4.0 / rmax**2 * d2_dx

#         H = -1 / 2 * d2_dr + jnp.diag(
#             0.5 * l * (l + 1) / r**2 + _V(r, potential_params)
#         )

#         E_n, u_n = jnp.linalg.eig(H)
#         check_mode_heath(E_n, E_min_l, E_max)

#         order = jnp.argsort(E_n)
#         E_n = E_n[order].real
#         u_n = u_n[:, order].real
#         u_n = u_n[:, E_n <= E_max]
#         E_n = E_n[E_n <= E_max]
#         n = E_n.shape[0]
#         norm = rmax / 2 * weights @ u_n**2
#         u_n = jnp.transpose(u_n / jnp.sqrt(norm))

#         u_j_r.append(u_n)
#         rmax_j.append(jnp.repeat(rmax, n))

#         E_j.append(jnp.asarray(E_n))
#         l_of_j.append(l * jnp.ones(n, dtype=int))
#         n_of_j.append(jnp.arange(n, dtype=int))

#         l += 1

#     u_j_r = jnp.concatenate(u_j_r)
#     # r0_j = jnp.concatenate(r0_j)
#     # dr_j = jnp.concatenate(dr_j)
#     rmax_j = jnp.concatenate(rmax_j)
#     # R_j_params = init_mult_spline_params(r0_j, dr_j, R_j_r)
#     u_j_params = init_mult_chebyshev_modes(u_j_r, rmax_j)

#     E_j = jnp.concatenate(E_j)
#     l_of_j = jnp.concatenate(l_of_j)
#     n_of_j = jnp.concatenate(n_of_j)

#     return eigenstate_library_matrix(
#         R_j_params=u_j_params,
#         E_j=E_j,
#         l_of_j=l_of_j,
#         n_of_j=n_of_j,
#         name=name,
#         potential_params=potential_params.name,
#     )


def x_of_r(r):
    return r + jnp.log(r)


def r_of_x(x):
    return lambertw(jnp.exp(x))


def init_eigenstate_library_fd(
    potential_params, r_ta, N, potential=gravitational_potential
):
    _V = jax.vmap(potential, in_axes=(0, None))
    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        eigenstate_library_matrix.compute_name, result_shape, potential_params, r_ta, N
    )

    # Discetized radial eigenstate
    R_j_r = []
    dr_j = []
    r0_j = []

    # Eigenvalues
    E_j = []
    # Quantum numbers
    l_of_j = []
    n_of_j = []

    E_max = jnp.asarray(potential(r_ta, potential_params))
    if sommerfeld_estimate_of_nmax(r_ta, potential_params, potential=potential) > N:
        logger.info("N is too small to construct the entire library!")

    for l in range(
        lmax_from_effective_potential(r_ta, potential_params, potential=potential)
    ):
        E_min_l = V_effective(
            minimum_of_effective_potential(
                r_ta, l, potential_params, potential=potential
            ),
            l,
            potential_params,
            potential=potential,
        )

        rmax = wkb_estimate_of_rmax(r_ta, l, potential_params, potential=potential)
        dr = rmax / N
        r = dr * jnp.arange(1, N)

        H_off_diag = -1 / (2 * dr**2) * jnp.ones(r.shape[0] - 1)
        H_diag = (
            -1 / (2 * dr**2) * -2 * jnp.ones_like(r)
            + 0.5 * l * (l + 1) / r**2
            + _V(r, potential_params)
        )
        E_n, u_n = eigh_tridiagonal(
            H_diag, H_off_diag, select="v", select_range=(E_min_l.item(), E_max.item())
        )
        check_mode_heath(E_n, E_min_l, E_max)
        n = E_n.shape[0]

        R_n = u_n / (r[:, np.newaxis] * jnp.sqrt(dr))
        R_j_r.append(jnp.asarray(R_n.T))
        r0_j.append(jnp.repeat(r[0], n))
        dr_j.append(jnp.repeat(dr, n))

        E_j.append(jnp.asarray(E_n))
        l_of_j.append(l * jnp.ones(n, dtype=int))
        n_of_j.append(jnp.arange(n, dtype=int))

        l += 1

    R_j_r = jnp.concatenate(R_j_r)
    r0_j = jnp.concatenate(r0_j)
    dr_j = jnp.concatenate(dr_j)
    R_j_params = init_mult_spline_params(r0_j, dr_j, R_j_r)

    E_j = jnp.concatenate(E_j)
    l_of_j = jnp.concatenate(l_of_j)
    n_of_j = jnp.concatenate(n_of_j)

    return eigenstate_library_matrix(
        R_j_params=R_j_params,
        E_j=E_j,
        l_of_j=l_of_j,
        n_of_j=n_of_j,
        name=name,
        potential_params=potential_params.name,
    )


def init_eigenstate_library_fd_loglin(
    potential_params, r_ta, N, potential=gravitational_potential
):
    _V = jax.vmap(potential, in_axes=(0, None))
    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        eigenstate_library_matrix.compute_name, result_shape, potential_params, r_ta, N
    )

    # Discetized radial eigenstate
    R_j_x = []
    dx_j = []
    x0_j = []

    # Eigenvalues
    E_j = []
    # Quantum numbers
    l_of_j = []
    n_of_j = []

    E_max = potential(r_ta, potential_params)
    if sommerfeld_estimate_of_nmax(r_ta, potential_params) > N:
        logger.info("N is too small to construct the entire library!")

    for l in range(lmax(r_ta, potential_params)):
        E_min_l = V_effective(
            minimum_of_effective_potential(r_ta, l, potential_params),
            l,
            potential_params,
        )

        rmin = wkb_estimate_of_rmin(
            lower_turning_point(r_ta, l, potential_params), l, potential_params
        )
        rmax = wkb_estimate_of_rmax(r_ta, l, potential_params)
        xmax = x_of_r(rmax)
        xmin = x_of_r(rmin)
        x = jnp.linspace(xmin, xmax, N)
        dx = x[1] - x[0]
        r = r_of_x(x)

        H_upper = (
            (1 + r[:-1]) ** 2
            / r[:-1] ** 2
            * -1
            / (2 * dx**2)
            * jnp.ones(x.shape[0] - 1)
        )
        H_lower = (
            (1 + r[1:]) ** 2
            / r[1:] ** 2
            * -1
            / (2 * dx**2)
            * jnp.ones(x.shape[0] - 1)
        )
        H_off_diag = -jnp.sqrt(H_upper * H_lower)
        H_diag = (
            (1 + r) ** 2 / r**2 * -1 / (2 * dx**2) * -2 * jnp.ones_like(x)
            + 0.5 * l * (l + 1) / r**2
            + _V(r, potential_params)
            + (1 + 4 * r) / (8 * r**2 * (1 + r) ** 2)
        )
        E_n, t_n = eigh_tridiagonal(
            H_diag,
            H_off_diag,
            select="v",
            select_range=(E_min_l.item(), E_max.item()),
        )

        check_mode_heath(E_n, E_min_l, E_max)
        n = E_n.shape[0]

        D = jnp.sqrt(jnp.cumprod(H_lower / H_upper))
        D = jnp.insert(D, 0, 1)
        t_n = jnp.diag(D) @ t_n

        norm = np.sqrt(
            jnp.sum(1 / (1 / r[:, np.newaxis] + 1) ** 2 * t_n**2 * dx, axis=0)
        )
        t_n = jnp.transpose(t_n / norm)
        R_j_x.append(jnp.asarray(t_n))
        x0_j.append(jnp.repeat(x[0], n))
        dx_j.append(jnp.repeat(dx, n))

        E_j.append(jnp.asarray(E_n))
        l_of_j.append(l * jnp.ones(n, dtype=int))
        n_of_j.append(jnp.arange(n, dtype=int))

        l += 1

    R_j_x = jnp.concatenate(R_j_x)
    x0_j = jnp.concatenate(x0_j)
    dx_j = jnp.concatenate(dx_j)
    R_j_params = init_mult_spline_params(x0_j, dx_j, R_j_x)

    E_j = jnp.concatenate(E_j)
    l_of_j = jnp.concatenate(l_of_j)
    n_of_j = jnp.concatenate(n_of_j)

    return eigenstate_library_matrix(
        R_j_params=R_j_params,
        E_j=E_j,
        l_of_j=l_of_j,
        n_of_j=n_of_j,
        name=name,
        potential_params=potential_params.name,
    )


def eval_eigenstate(r, R_j_params):
    return eval_interp1d(r, R_j_params)


def eval_eigenstate_loglin(r, R_j_params):
    return eval_interp1d(x_of_r(r), R_j_params) / (r * jnp.sqrt(1 / r + 1))


def eval_eigenstate_cheb(r, R_j_params):
    return eval_chebyshev_polynomial(r, R_j_params) / r
