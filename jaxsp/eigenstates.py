from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import eigh_tridiagonal
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

from .constants import mah_factor, maV_factor, to_kpc_factor
from .interpolate import init_1d_interpolation_params

_eigenstate_library = namedtuple(
    "eigenstate_library", ["R_j_params", "E_j", "l_of_j", "n_of_j"]
)


class eigenstate_library(_eigenstate_library):
    def j_of_nl(self, n, l):
        n = int(n)
        l_index = jnp.searchsorted(self.l_of_j, l)
        n_index = jnp.argmax(self.n_of_j[l_index:] == n)
        return l_index + n_index

    @property
    def J(self):
        return self.E_j.shape[0]

    def __repr__(self):
        return f"eigenstate_library(\nJ={len(self.l_of_j)},\nl_of_j={self.l_of_j},\nn_of_j={self.n_of_j})"


init_mult_spline_params = jax.vmap(
    init_1d_interpolation_params, in_axes=(None, None, 0)
)


def init_eigenstate_library(V, ma, E_max, rmax, N=4096):
    dr = rmax / N
    rkpc = dr * jnp.arange(1, N)

    # Discetized radial eigenstate
    R_j_r = []
    # Eigenvalues
    E_j = []
    # Quantum numbers
    l_of_j = []
    n_of_j = []

    l = 0
    while True:
        H_diag, H_off_diag = construct_spherical_hamiltonian(rkpc, ma, l, V)
        rmin = minimum_of_effective_potential(ma, l, V, dr, rmax)
        E_min_l = V_effective(rmin, ma, l, V)

        # If circular orbit above energy cutoff...stop
        if E_min_l > E_max:
            break
        E_n, u_n = eigh_tridiagonal(
            H_diag, H_off_diag, select="v", select_range=(E_min_l, E_max)
        )

        # If no mode exists in interval...stop
        if E_n.shape[0] == 0:
            break

        # Add noise floor to all modes so that the number of roots is equal
        # to n
        u_n = u_n + 10 * jnp.finfo(jnp.float64).eps

        # Check if states are degenerate (this is impossible in 1D and bound,
        # normalizable states. If it happens, numerics is tha cause.
        if np.any(np.unique(E_n, return_counts=True)[1] > 1):
            print(
                "Degeneracy detected. This is impossible in 1D. "
                "Consider tweaking N/L."
            )
            break

        R_n = u_n / (rkpc[:, np.newaxis] * jnp.sqrt(rkpc[1] - rkpc[0]))
        R_j_r.append(jnp.asarray(R_n.T))
        E_j.append(jnp.asarray(E_n))
        l_of_j.append(l * jnp.ones_like(E_n))
        n_of_j.append(jnp.arange(E_n.shape[0]))

        l += 1

    R_j_r = jnp.concatenate(R_j_r)
    E_j = jnp.concatenate(E_j)
    l_of_j = jnp.concatenate(l_of_j)
    n_of_j = jnp.concatenate(n_of_j)

    R_j_params = init_mult_spline_params(rkpc[0], dr, R_j_r)

    return eigenstate_library(
        R_j_params=R_j_params,
        E_j=E_j,
        l_of_j=l_of_j,
        n_of_j=n_of_j,
    )


def V_effective(rkpc, ma, l, V):
    return (
        0.5 * l * (l + 1) / rkpc**2
        + mah_factor.value
        * maV_factor.value
        * ma**2
        * to_kpc_factor.value
        * V(jnp.array([rkpc]))[0]
    )


def minimum_of_effective_potential(ma, l, V, rmin, rmax):
    if l == 0:
        return rmin
    pg = ProjectedGradient(
        fun=lambda r: V_effective(r, ma, l, V),
        projection=projection_box,
        tol=1e-3,
    )
    return jnp.min(jnp.array([pg.run(1.0, hyperparams_proj=(rmin, rmax)).params, rmax]))


def construct_spherical_hamiltonian(rkpc, ma, l, V):
    N = rkpc.shape[0]
    dr = rkpc[1] - rkpc[0]
    H_off_diag = -1 / (2 * dr**2) * np.ones(N - 1)

    H_diag = (
        -1 / (2 * dr**2) * -2 * np.ones(N)
        + 0.5 * l * (l + 1) / rkpc**2
        + mah_factor.value * maV_factor.value * ma**2 * to_kpc_factor.value * V(rkpc)
    )
    return H_diag, H_off_diag
