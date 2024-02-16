from typing import NamedTuple
import hashlib

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy.linalg import eigh_tridiagonal
from jaxopt import Bisection, Broyden, LBFGSB, ProjectedGradient
from jaxopt.projection import projection_box

from .interpolate import init_1d_interpolation_params, eval_interp1d
from .potential import potential
from .utils import quad
from .io_utils import hash_to_int64

import logging

logger = logging.getLogger(__name__)


class eigenstate_library(NamedTuple):
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
        combined.update(hashlib.md5(jnp.asarray(r_ta)).digest())
        combined.update(hashlib.md5(jnp.asarray(N)).digest())
        combined.update(hashlib.md5(jnp.asarray(potential_params.name)).digest())
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


init_mult_spline_params = jax.vmap(
    init_1d_interpolation_params, in_axes=(None, None, 0)
)


def V_effective(r, l, potential_params):
    return 0.5 * l * (l + 1) / (1e-10 + r**2) + potential(r, potential_params)


# def minimum_of_effective_potential(r_ta, l, potential_params):
#     obj = jax.grad(lambda r: V_effective(r, l, potential_params))
#     bisec = Bisection(
#         optimality_fun=obj,
#         lower=1e-10,
#         upper=r_ta - 1e-10,
#         tol=1e-3,
#         check_bracket=False,
#     )
#     rmin = bisec.run().params
#     return rmin


@jax.jit
def minimum_of_effective_potential(r_ta, l, potential_params):
    pg = ProjectedGradient(
        fun=lambda logr: V_effective(jnp.exp(logr), l, potential_params),
        projection=projection_box,
        verbose=False,
    )
    logrmin = pg.run(
        jnp.log(r_ta / 2), hyperparams_proj=(jnp.log(1e-10), jnp.log(r_ta - 1e-10))
    ).params
    return jnp.exp(logrmin)


@jax.jit
def wkb_estimate_of_rmax(r_ta, l, potential_params):
    @jax.vmap
    def r_classical_Veff(r):
        return V_effective(r, l, potential_params) - potential(r_ta, potential_params)

    def wkb_condition_Veff(r_lower, r_upper):
        return (
            jnp.sqrt(2)
            * quad(lambda r: jnp.sqrt(r_classical_Veff(r)), r_lower, r_upper)
            - 20
        )

    # Determine radius according to WKB decay in forbidden region
    bisec = Broyden(
        fun=lambda logr: wkb_condition_Veff(r_ta, jnp.exp(logr)),
    )
    logrmax = bisec.run(jnp.array(jnp.log(r_ta))).params

    return jnp.exp(logrmax)


def check_mode_heath(E_n):
    if np.any(np.unique(E_n, return_counts=True)[1] > 1):
        raise Exception("Degeneracy detected. This is impossible in 1D.")
    # Addtional tests go here (e.g. number of roots = n?)


def init_eigenstate_library(potential_params, r_ta, N):
    V = jax.vmap(potential, in_axes=(0, None))
    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        eigenstate_library.compute_name, result_shape, potential_params, r_ta, N
    )

    # Discetized radial eigenstate
    R_j_r = []
    # Eigenvalues
    E_j = []
    # Quantum numbers
    l_of_j = []
    n_of_j = []

    lmax = 0
    E_max = potential(r_ta, potential_params)
    lmax = jax.lax.while_loop(
        lambda l: V_effective(
            minimum_of_effective_potential(r_ta, l, potential_params),
            l,
            potential_params,
        )
        < E_max,
        lambda l: l + 1,
        1,
    )
    for l in range(lmax):
        rmin = minimum_of_effective_potential(r_ta, l, potential_params)
        E_min_l = V_effective(rmin, l, potential_params)

        rmax = wkb_estimate_of_rmax(r_ta, l, potential_params)
        dr = rmax / N
        r = dr * jnp.arange(1, N)
        print(dr, rmin, rmax, r_ta)

        H_off_diag = -1 / (2 * dr**2) * jnp.ones(r.shape[0] - 1)
        H_diag = (
            -1 / (2 * dr**2) * -2 * jnp.ones_like(r)
            + 0.5 * l * (l + 1) / r**2
            + V(r, potential_params)
        )
        E_n, u_n = eigh_tridiagonal(
            H_diag,
            H_off_diag,
            select="v",
            select_range=(E_min_l.item(), E_max.item()),
        )

        check_mode_heath(E_n)
        n = E_n.shape[0]

        R_n = jnp.transpose(u_n / (r[:, np.newaxis] * jnp.sqrt(dr)))
        R_j_r.append(jnp.asarray(R_n))
        E_j.append(jnp.asarray(E_n))
        l_of_j.append(l * jnp.ones(n, dtype=int))
        n_of_j.append(jnp.arange(n, dtype=int))

        l += 1

    R_j_r = jnp.concatenate(R_j_r)
    R_j_params = init_mult_spline_params(r[0], dr, R_j_r)
    E_j = jnp.concatenate(E_j)
    l_of_j = jnp.concatenate(l_of_j)
    n_of_j = jnp.concatenate(n_of_j)

    return eigenstate_library(
        R_j_params=R_j_params,
        E_j=E_j,
        l_of_j=l_of_j,
        n_of_j=n_of_j,
        name=name,
        potential_params=potential_params.name,
    )


def eval_eigenstate(r, R_j_params):
    return eval_interp1d(r, R_j_params)
