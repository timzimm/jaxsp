import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxopt import Bisection

from .interpolate import piecewise_constant_interpolation_params
from .potential import potential as gravitational_potential
from .special import lambertw
from .io_utils import hash_to_int64
from .utils import map_vmap, quad


class eigenmode_params(NamedTuple):
    tk: ArrayLike
    wk2: ArrayLike
    x0: float
    dx: float


class radial_eigenmode_params(NamedTuple):
    eigenmode_params: NamedTuple
    l: int
    n: int
    E: float


class eigenstate_library(NamedTuple):
    """
    Parameters specifying the eigenstate_library
    """

    name: int
    potential_params: int
    radial_eigenmode_params: NamedTuple

    @classmethod
    def compute_name(cls, potential_params, r_min, r_max, N):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.asarray(potential_params.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(r_min)).digest())
        # combined.update(hashlib.md5(jnp.asarray(r_max)).digest())
        combined.update(hashlib.md5(jnp.asarray(N)).digest())
        return hash_to_int64(combined.hexdigest())

    def j_of_nl(self, n, l):
        n = int(n)
        l_index = jnp.searchsorted(self.radial_eigenmode_params.l, l)
        n_index = jnp.argmax(self.radial_eigenmode_params.n[l_index:] == n)
        return l_index + n_index

    @property
    def J(self):
        return self.radial_eigenmode_params.E.shape[0]

    def __repr__(self):
        return (
            f"eigenstate_library:"
            f"\n\tname={self.name},"
            f"\n\tpotential_params={self.potential_params},"
            f"\n\ttotal number of modes={self.J},"
            f"\n\tlmax={jnp.max(self.radial_eigenmode_params.l)},"
            f"\n\tnmax={jnp.max(self.radial_eigenmode_params.n)},"
            f"\n\tEmin={jnp.min(self.radial_eigenmode_params.E)},"
            f"\n\tEmax={jnp.max(self.radial_eigenmode_params.E)}"
        )


def select_eigenmode_nl(n, l, eigenstate_library):
    radial_eigenmode_params = eigenstate_library.radial_eigenmode_params
    single_mode_params = jax.tree_util.tree_map(
        lambda param: param[
            jnp.logical_and(
                radial_eigenmode_params.n == n,
                radial_eigenmode_params.l == l,
            )
        ],
        radial_eigenmode_params,
    )

    return eigenstate_library(
        radial_eigenmode_params=single_mode_params,
        name=eigenstate_library.name,
        potential_params=eigenstate_library.potential_params,
    )


a = 1
b = 10


@jax.jit
def x_of_r(r):
    """
    Transformation from loglinear (non-uniform) r to linear (uniform) x
    """
    return a * r + jax.scipy.special.xlogy(b, r)


@jax.jit
def r_of_x(x):
    """
    Transformation from linear (uniform) x to log-linear (non-uniform) r
    """

    def lambertw_exp_asymptotic(x, a, b):
        """
        Asymptotic approximation for W(a/b * exp(x/b)) for large x
        (which would overflow due to exp())

        See:
            https://en.wikipedia.org/wiki/Lambert_W_function#Asymptotic_expansions
        """
        L1 = jnp.log(a / b) + x / b
        L2 = jnp.log(L1)
        return L1 - L2 + L2 / L1 + L2 * (-2 + L2) / (2 * L1**2)

    if a == 0:
        return jnp.exp(x / b)
    if b == 0:
        return x / a
    return jax.lax.cond(
        x < 100,
        lambda x: b / a * lambertw(a / b * jnp.exp(x / b)),
        lambda x: b / a * lambertw_exp_asymptotic(x, a, b),
        x,
    )


def q(x, l, V0, potential_params, potential=gravitational_potential):
    """
    Sturm Liouville q-function of log-linear transformed radial Schroedinger equation.
    The full form SL-problem reads:
                    - t'' + q(x) t(x) = 2 E w(x) t(x)
    See:
        https://doi.org/10.1016/j.hedp.2023.101042
    """
    r = r_of_x(x)
    return (
        1.0
        / (b + a * r) ** 2
        * (
            l * (l + 1)
            + 2 * r**2 * (potential(r, potential_params) + V0)
            + b * (b + 4 * a * r) / (4 * (b + a * r) ** 2)
        )
    )


def w(x):
    """
    Sturm Liouville w-function of log-linear transformed radial Schroedinger equation
    The full form SL-problem reads:
                    - t'' + q(x) t(x) = 2 E w(x) t(x)
    See:
        https://doi.org/10.1016/j.hedp.2023.101042
    """
    r = r_of_x(x)
    return 2 * r**2 / (b + a * r) ** 2


def init_piecewise_constant_q(
    potential_params, l, V0, rmin, rmax, N, potential=gravitational_potential
):
    qq = jax.vmap(q, in_axes=(0, None, None, None))
    x = jnp.linspace(x_of_r(rmin), x_of_r(rmax), N)
    return piecewise_constant_interpolation_params(
        f_i=qq(1 / 2 * (x[:-1] + x[1:]), l, V0, potential_params, potential=potential),
        x_i=x,
    )


def init_piecewise_constant_w(rmin, rmax, N):
    ww = jax.vmap(w)
    x = jnp.linspace(x_of_r(rmin), x_of_r(rmax), N)
    return piecewise_constant_interpolation_params(
        f_i=ww(1 / 2 * (x[:-1] + x[1:])),
        x_i=x,
    )


def B_k(wk2, lk):
    def allowed(wk, wk_lk):
        return wk / jnp.sin(wk_lk)

    def on_turning_point(wk, wk_lk):
        return wk / wk_lk

    def forbidden(wk, wk_lk):
        return wk / jnp.sinh(wk_lk)

    wk = jnp.sqrt(jnp.abs(wk2))
    wk_lk = wk * lk

    return jax.lax.switch(
        (jnp.sign(wk2) + 1).astype(int),
        [forbidden, on_turning_point, allowed],
        wk,
        wk_lk,
    )


def A_k(wk2, lk):
    def allowed(wk, wk_lk):
        return wk / jnp.tan(wk_lk)

    def on_turning_point(wk, wk_lk):
        return wk / wk_lk

    def forbidden(wk, wk_lk):
        return wk / jnp.tanh(wk_lk)

    wk = jnp.sqrt(jnp.abs(wk2))
    wk_lk = wk * lk

    return jax.lax.switch(
        (jnp.sign(wk2) + 1).astype(int),
        [forbidden, on_turning_point, allowed],
        wk,
        wk_lk,
    )


def right_turning_point(E, params_w, params_q):
    wk2 = params_w.f_i * E - params_q.f_i
    k = jnp.argmax(wk2[::-1] >= 0)
    return r_of_x(params_q.x_i[-1 - k])


def number_of_eigenvalues_up_to(E, params_w, params_q):
    """
    Wittrick-Williams (WW) algorithm for finding all eigenvalues of a full form
    SL problem for E' in [0, E]

    See:
        10.1016/0021-9991(83)90101-8
    """

    def compute_principal_minors_ratio_of_symmetric_tridiagonal_matrix(A, B):
        def R_k(Rkm1, k):
            Rk = A[k] - 1.0 / Rkm1 * B[k - 1] ** 2
            return Rk, Rk

        return jax.lax.scan(R_k, A[0], jnp.arange(1, A.shape[0] - 1))[1]

    wk2 = params_w.f_i * E - params_q.f_i
    wk_allowed = jnp.sqrt(jax.nn.relu(params_w.f_i * E - params_q.f_i))
    lk = jnp.diff(params_q.x_i)

    N0 = jnp.sum((wk_allowed * lk / jnp.pi).astype(int))
    Ak = jax.vmap(A_k)(wk2, lk)
    Bk = jax.vmap(B_k)(wk2, lk)
    R = compute_principal_minors_ratio_of_symmetric_tridiagonal_matrix(
        Ak[:-1] + Ak[1:], Bk[1:-1]
    )
    sign_count = jnp.sum(R < 0)
    return N0 + sign_count


def V_effective(r, l, potential_params, potential=gravitational_potential):
    return 0.5 * l * (l + 1) / r**2 + potential(r, potential_params)


@jax.jit
def wkb_estimate_of_rmax(r, l, potential_params, potential=gravitational_potential):
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


def lmax(potential_params, r_max, N, potential=gravitational_potential):
    def nmax(l, Emax, potential):
        r_wkb_for_r_ta = wkb_estimate_of_rmax(
            r_max, l, potential_params, potential=potential
        )
        rmin = 1e-6

        params_q = init_piecewise_constant_q(
            potential_params, l, V0, rmin, r_wkb_for_r_ta, N, potential=potential
        )
        params_w = init_piecewise_constant_w(rmin, r_wkb_for_r_ta, N)
        return number_of_eigenvalues_up_to(Emax, params_w, params_q)

    V0 = jnp.abs(potential(0.0, potential_params))
    Emax = potential(r_max, potential_params) + V0
    return jax.lax.while_loop(
        lambda l: nmax(l, Emax, potential) > 0,
        lambda l: l + 1,
        0,
    ).astype(jnp.int32)


def nmax(l, potential_params, r_min, r_max, N, potential=gravitational_potential):
    V0 = jnp.abs(potential(0.0, potential_params))
    Emax = potential(r_max, potential_params) + V0
    r_wkb_for_r_ta = wkb_estimate_of_rmax(
        r_max, l, potential_params, potential=potential
    )

    params_q = init_piecewise_constant_q(
        potential_params, l, V0, r_min, r_wkb_for_r_ta, N, potential=potential
    )
    params_w = init_piecewise_constant_w(r_min, r_wkb_for_r_ta, N)
    return number_of_eigenvalues_up_to(Emax, params_w, params_q).astype(jnp.int32)


def bound_eigenvalue_k(k, Emax, params_w, params_q, eps=1e-8):
    """
    WW-algorithm based, modified bisection to bound eigenvalue E_k.

    See:
        10.1016/0021-9991(83)90101-8 (this method)
        10.1002/nme.1620260810 (better method TODO)
    """

    def not_converged(E_l_E_E_u):
        E_l, E, E_u = E_l_E_E_u
        return jnp.abs(E_u - E_l) > eps

    def adjust_bracket(E_l_E_E_u):
        E_l, E, E_u = E_l_E_E_u
        E = 0.5 * (E_l + E_u)
        return jax.lax.cond(
            number_of_eigenvalues_up_to(E, params_w, params_q) >= k,
            lambda E_l, E, E_u: (E_l, E, E),
            lambda E_l, E, E_u: (E, E, E_u),
            E_l,
            E,
            E_u,
        )

    E_l, E_u = 0.0, Emax
    E = 0.5 * (E_u + E_l)
    E_l, _, E_u = jax.lax.while_loop(not_converged, adjust_bracket, (E_l, E, E_u))
    return E_l, E_u


compute_diagonal_elements_of_adjacency_mat = jax.vmap(A_k)
compute_offdiagonal_elements_of_adjacency_mat = jax.vmap(B_k)


def init_eigenmode_params_between(E_l, E_u, params_w, params_q, eps=1e-8):
    """
    See:
        10.1016/s0022-460x(03)00126-3
    """

    def K(E):
        """
        Generates the tridiagonal "stiffness matrix" K(E), which at E_true is
        singular i.e.:
                            det(K(E_true)) = 0
        """
        wk2 = params_w.f_i * E - params_q.f_i

        A = compute_diagonal_elements_of_adjacency_mat(wk2, lk)
        B = compute_offdiagonal_elements_of_adjacency_mat(wk2, lk)
        d = A[1:] + A[:-1]
        du = -B[1:].at[-1].set(0)
        dl = -B[:-1].at[0].set(0)
        return dl, d, du

    lk = jnp.diff(params_q.x_i)
    dl_l, d_l, du_l = K(E_l)
    dl_u, d_u, du_u = K(E_u)
    dl_ul = dl_u - dl_l
    d_ul = d_u - d_l
    du_ul = du_u - du_l
    K_m = K(0.5 * (E_l + E_u))

    def not_converged(vk_vkm1):
        vk, vkm1 = vk_vkm1
        return jnp.max(jnp.abs(jnp.abs(vk) - jnp.abs(vkm1))) > eps

    def inverse_iteration(vk_vkm1):
        vk, _ = vk_vkm1
        rhs = dl_ul * vk + d_ul * vk + du_ul * vk
        vkp1 = jax.lax.linalg.tridiagonal_solve(*K_m, rhs[:, jnp.newaxis]).ravel()
        return vkp1 / jnp.max(jnp.abs(vkp1)), vk

    key = jax.random.PRNGKey(42)
    key, key2 = jax.random.split(key)

    N = params_q.x_i.shape[0] - 2
    tk = jax.lax.while_loop(
        not_converged,
        inverse_iteration,
        (jax.random.uniform(key, shape=(N,)), jax.random.uniform(key2, shape=(N,))),
    )[0]

    # Normalisation
    wk2 = params_w.f_i * 0.5 * (E_l + E_u) - params_q.f_i
    A = compute_diagonal_elements_of_adjacency_mat(wk2, lk)
    B = compute_offdiagonal_elements_of_adjacency_mat(wk2, lk)
    N = jnp.sum(
        params_w.f_i[1:-1]
        / (2 * jnp.abs(wk2[1:-1]))
        * (
            (tk[:-1] ** 2 + tk[1:] ** 2) * (B[1:-1] ** 2 * lk[1:-1] - A[1:-1])
            + 2 * B[1:-1] * tk[:-1] * tk[1:] * (1 - A[1:-1] * lk[1:-1])
        )
    )
    N += (
        params_w.f_i[0]
        / (2 * jnp.abs(wk2[0]))
        * (tk[0] ** 2)
        * (B[0] ** 2 * lk[0] - A[0])
    )
    N += (
        params_w.f_i[-1]
        / (2 * jnp.abs(wk2[-1]))
        * (tk[-1] ** 2)
        * (B[-1] ** 2 * lk[-1] - A[-1])
    )

    return eigenmode_params(
        tk=tk / jnp.sqrt(N),
        wk2=wk2,
        x0=params_q.x_i[0],
        dx=params_q.x_i[1] - params_q.x_i[0],
    )


def init_radial_eigenmode_params(
    l, n, potential_params, r_min, r_max, N, potential=gravitational_potential
):
    k = n + 1
    V0 = jnp.abs(potential(0.0, potential_params))
    Emax = potential(r_max, potential_params) + V0
    rwkb = wkb_estimate_of_rmax(r_max, l, potential_params, potential=potential)
    params_q = init_piecewise_constant_q(
        potential_params, l, V0, r_min, rwkb, N, potential=potential
    )
    params_w = init_piecewise_constant_w(r_min, rwkb, N)
    E_l, E_u = bound_eigenvalue_k(k, Emax, params_w, params_q)

    return radial_eigenmode_params(
        eigenmode_params=init_eigenmode_params_between(E_l, E_u, params_w, params_q),
        l=l,
        n=n,
        E=0.5 * (E_l + E_u) - V0,
    )


def init_eigenstate_library(
    potential_params, r_min, r_max, N, batch_size=16, potential=gravitational_potential
):
    init_radial_eigenmodes = jax.jit(
        map_vmap(
            init_radial_eigenmode_params,
            in_axes=(0, 0, None, None, None, None),
            batch_size=batch_size,
        ),
        static_argnames="N",
    )
    compute_all_nmax = jax.vmap(nmax, in_axes=(0, None, None, None, None))
    ll = lmax(potential_params, r_max, N, potential=potential)
    nn = compute_all_nmax(
        jnp.arange(ll), potential_params, r_min, r_max, N, potential=potential
    )

    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        eigenstate_library.compute_name, result_shape, potential_params, r_min, r_max, N
    )
    if nn.shape[0] == 0:
        return None

    ls = jnp.repeat(jnp.arange(ll), nn)
    ns = jnp.concatenate([jnp.arange(n) for n in nn])

    return eigenstate_library(
        radial_eigenmode_params=init_radial_eigenmodes(
            ls, ns, potential_params, r_min, r_max, N, potential=potential
        ),
        name=name,
        potential_params=potential_params.name,
    )


def eval_eigenmode(x, eigenmode_params):
    def allowed(x, A, B, wk, xk, xkm1):
        lk = xk - xkm1
        wk_lk = wk * lk
        return (
            1.0
            / jnp.sin(wk_lk)
            * (A * jnp.sin(wk * (xk - x)) + B * jnp.sin(wk * (x - xkm1)))
        )

    def on_turning_point(x, A, B, wk, xk, xkm1):
        lk = xk - xkm1
        return 1.0 / lk * (A * (xk - x) + B * (x - xkm1))

    def forbidden(x, A, B, wk, xk, xkm1):
        lk = xk - xkm1
        wk_lk = wk * lk
        return (
            1.0
            / jnp.sinh(wk_lk)
            * (A * jnp.sinh(wk * (xk - x)) + B * jnp.sinh(wk * (x - xkm1)))
        )

    dx = eigenmode_params.dx
    x0 = eigenmode_params.x0
    x = jnp.clip(x, x0 + dx, x0 + dx * eigenmode_params.tk.shape[0])
    k = ((x - x0) // dx + 1).astype(int)

    wk2 = eigenmode_params.wk2[k]
    wk = jnp.sqrt(jnp.abs(wk2))

    xk = dx * k + x0
    xkm1 = xk - dx

    A = eigenmode_params.tk[k - 1]
    B = eigenmode_params.tk[k]

    tk = jax.lax.switch(
        (jnp.sign(wk2) + 1).astype(int),
        [forbidden, on_turning_point, allowed],
        x,
        A,
        B,
        wk,
        xk,
        xkm1,
    )
    return jnp.nan_to_num(tk)


def eval_radial_eigenmode(r, radial_eigenmode_params):
    x = x_of_r(r)
    return eval_eigenmode(x, radial_eigenmode_params.eigenmode_params) / jnp.sqrt(
        b * r + a * r**2
    )
