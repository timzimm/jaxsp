import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .interpolate import piecewise_constant_interpolation_params
from .potential import potential as gravitational_potential
from .io_utils import hash_to_int64
from .utils import map_vmap
from .radial_schroedinger import (
    x_of_r,
    r_of_x,
    wkb_estimate_of_rmax,
    w,
    q,
)


class eigenmode_params(NamedTuple):
    tk: ArrayLike
    wk2: ArrayLike
    x0: float
    dx: float


class radial_eigenmode_params(NamedTuple):
    eigenmode_params: NamedTuple
    a: float
    b: float
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
    def compute_name(cls, potential_params, r_min, r_max, a, b, N):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.asarray(potential_params.name)).digest())
        combined.update(hashlib.md5(jnp.asarray(r_min)).digest())
        # combined.update(hashlib.md5(jnp.asarray(r_max)).digest())
        combined.update(hashlib.md5(jnp.asarray(a)).digest())
        combined.update(hashlib.md5(jnp.asarray(b)).digest())
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


def select_eigenmode_nl(n, l, lib):
    radial_eigenmode_params = lib.radial_eigenmode_params
    n = radial_eigenmode_params.n == n
    l = radial_eigenmode_params.n == l
    single_mode_params = jax.tree_util.tree_map(
        lambda param: param[jnp.logical_and(n, l)],
        radial_eigenmode_params,
    )

    return eigenstate_library(
        radial_eigenmode_params=single_mode_params,
        name=lib.name,
        potential_params=lib.potential_params,
    )


def init_piecewise_constant_q(
    potential_params, l, V0, rmin, rmax, a, b, N, potential=gravitational_potential
):
    qq = jax.vmap(q, in_axes=(0, None, None, None, None, None))
    x = jnp.linspace(x_of_r(rmin, a, b), x_of_r(rmax, a, b), N)
    return piecewise_constant_interpolation_params(
        f_i=qq(
            1 / 2 * (x[:-1] + x[1:]), a, b, l, V0, potential_params, potential=potential
        ),
        x0=x[0],
        dx=x[1] - x[0],
    )


def init_piecewise_constant_w(rmin, rmax, a, b, N):
    ww = jax.vmap(w, in_axes=(0, None, None))
    x = jnp.linspace(x_of_r(rmin, a, b), x_of_r(rmax, a, b), N)
    return piecewise_constant_interpolation_params(
        f_i=ww(1 / 2 * (x[:-1] + x[1:]), a, b),
        x0=x[0],
        dx=x[1] - x[0],
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


def right_turning_point(E, a, b, params_w, params_q):
    wk2 = params_w.f_i * E - params_q.f_i
    k = jnp.argmax(wk2[::-1] >= 0)
    return r_of_x(params_q.x0 + (params_q.f_i.shape[0] - k - 0.5) * params_q.dx, a, b)


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
    wk_allowed = jnp.sqrt(jax.nn.relu(wk2))
    lk = params_q.dx

    N0 = jnp.sum((wk_allowed * lk / jnp.pi).astype(int))
    Ak = jax.vmap(A_k, in_axes=(0, None))(wk2, lk)
    Bk = jax.vmap(B_k, in_axes=(0, None))(wk2, lk)
    R = compute_principal_minors_ratio_of_symmetric_tridiagonal_matrix(
        Ak[:-1] + Ak[1:], Bk[1:-1]
    )
    sign_count = jnp.sum(R < 0)
    return N0 + sign_count


def lmax(potential_params, r_min, r_max, a, b, N, potential=gravitational_potential):
    def nmax(l, Emax, potential):
        r_wkb_for_r_ta = wkb_estimate_of_rmax(
            r_max, l, potential_params, potential=potential
        )

        params_q = init_piecewise_constant_q(
            potential_params, l, V0, r_min, r_wkb_for_r_ta, a, b, N, potential=potential
        )
        params_w = init_piecewise_constant_w(r_min, r_wkb_for_r_ta, a, b, N)
        return number_of_eigenvalues_up_to(Emax, params_w, params_q)

    V0 = jnp.abs(potential(0.0, potential_params))
    Emax = potential(r_max, potential_params) + V0
    return jax.lax.while_loop(
        lambda l: nmax(l, Emax, potential) > 0,
        lambda l: l + 1,
        0,
    ).astype(jnp.int32)


def nmax(l, potential_params, r_min, r_max, a, b, N, potential=gravitational_potential):
    V0 = jnp.abs(potential(0.0, potential_params))
    Emax = potential(r_max, potential_params) + V0
    r_wkb_for_r_ta = wkb_estimate_of_rmax(
        r_max, l, potential_params, potential=potential
    )

    params_q = init_piecewise_constant_q(
        potential_params, l, V0, r_min, r_wkb_for_r_ta, a, b, N, potential=potential
    )
    params_w = init_piecewise_constant_w(r_min, r_wkb_for_r_ta, a, b, N)
    return number_of_eigenvalues_up_to(Emax, params_w, params_q).astype(jnp.int32)


def bound_eigenvalue_k(k, Emax, params_w, params_q, eps=1e-10):
    """
    WW-algorithm based, modified bisection to bound eigenvalue E_k.

    See:
        10.1016/0021-9991(83)90101-8 (this method)
        10.1002/nme.1620260810 (better method TODO)
    """

    def adjust_bracket(i, E_l_E_u):
        E_l, E_u = E_l_E_u
        E = 0.5 * (E_l + E_u)
        return jax.lax.cond(
            number_of_eigenvalues_up_to(E, params_w, params_q) >= k,
            lambda E_l, E_u: (E_l, E),
            lambda E_l, E_u: (E, E_u),
            E_l,
            E_u,
        )

    E_l, E_u = 0.0, Emax
    n = (jnp.ceil(jnp.log2(Emax / eps))).astype(int)
    E_l, E_u = jax.lax.fori_loop(1, n, adjust_bracket, (E_l, E_u))
    return E_l, E_u


compute_diagonal_elements_of_adjacency_mat = jax.vmap(A_k, in_axes=(0, None))
compute_offdiagonal_elements_of_adjacency_mat = jax.vmap(B_k, in_axes=(0, None))


def _double(f, args):
    return (f(*args), f(*args))


def _tridiagonal_solve_first_stage(dl, d, du):
    def fwd1(tu_, x):
        return x[1] / (x[0] - x[2] * tu_)

    # Move relevant dimensions to the front for the scan.
    dl = jnp.moveaxis(dl, -1, 0)
    d = jnp.moveaxis(d, -1, 0)
    du = jnp.moveaxis(du, -1, 0)

    # Forward pass.
    _, tu_ = jax.lax.scan(
        lambda tu_, x: _double(fwd1, (tu_, x)), du[0] / d[0], (d, du, dl), unroll=32
    )

    return dl, d, tu_


def _tridiagonal_solve_second_stage(dl, d, tu_, b):
    def prepend_zero(x):
        return jnp.append(jnp.zeros((1,) + x.shape[1:], dtype=x.dtype), x[:-1], axis=0)

    def fwd2(b_, x):
        return (x[0] - x[3][jnp.newaxis, ...] * b_) / (x[1] - x[3] * x[2])[
            jnp.newaxis, ...
        ]

    def bwd1(x_, x):
        return x[0] - x[1][jnp.newaxis, ...] * x_

    # Move relevant dimensions to the front for the scan.
    b = jnp.moveaxis(b, -1, 0)
    b = jnp.moveaxis(b, -1, 0)

    # Forward pass.
    _, b_ = jax.lax.scan(
        lambda b_, x: _double(fwd2, (b_, x)),
        b[0] / d[0:1],
        (b, d, prepend_zero(tu_), dl),
        unroll=32,
    )

    # Backsubstitution.
    _, x_ = jax.lax.scan(
        lambda x_, x: _double(bwd1, (x_, x)), b_[-1], (b_[::-1], tu_[::-1]), unroll=32
    )

    result = x_[::-1]
    result = jnp.moveaxis(result, 0, -1)
    result = jnp.moveaxis(result, 0, -1)
    return result


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

    lk = params_q.dx
    dl_l, d_l, du_l = K(E_l)
    dl_u, d_u, du_u = K(E_u)
    dl_ul = dl_u - dl_l
    d_ul = d_u - d_l
    du_ul = du_u - du_l

    E_m = 0.5 * (E_l + E_u)
    K_m = _tridiagonal_solve_first_stage(*K(E_m))

    def not_converged(vk_muk_vkm1):
        vk, _, vkm1 = vk_muk_vkm1
        return jnp.max(jnp.abs(jnp.abs(vk) - jnp.abs(vkm1))) > eps

    def inverse_iteration(vk_muk_vkm1):
        vk, _, _ = vk_muk_vkm1
        rhs = dl_ul * jnp.roll(vk, 1) + d_ul * vk + du_ul * jnp.roll(vk, -1)
        vkp1 = _tridiagonal_solve_second_stage(*K_m, rhs[:, jnp.newaxis]).ravel()
        mukp1 = 1.0 / jnp.max(jnp.abs(vkp1))
        return mukp1 * vkp1, mukp1, vk

    key = jax.random.PRNGKey(42)
    key, key2 = jax.random.split(key, 2)

    N = params_q.f_i.shape[0] - 1
    tk, muk, _ = jax.lax.while_loop(
        not_converged,
        inverse_iteration,
        (
            jax.random.uniform(key, shape=(N,)),
            0.5,
            jax.random.uniform(key2, shape=(N,)),
        ),
    )
    # mu check + extrapolation
    E = jnp.where(jnp.abs(muk) < 1 / 2, E_m - muk * (E_u - E_l), E_m)

    # Normalisation
    wk2 = params_w.f_i * E - params_q.f_i
    A = compute_diagonal_elements_of_adjacency_mat(wk2, lk)
    B = compute_offdiagonal_elements_of_adjacency_mat(wk2, lk)
    N = jnp.sum(
        params_w.f_i[1:-1]
        / (2 * jnp.abs(wk2[1:-1]))
        * (
            (tk[:-1] ** 2 + tk[1:] ** 2) * (B[1:-1] ** 2 * lk - A[1:-1])
            + 2 * B[1:-1] * tk[:-1] * tk[1:] * (1 - A[1:-1] * lk)
        )
    )
    N += (
        params_w.f_i[0] / (2 * jnp.abs(wk2[0])) * (tk[0] ** 2) * (B[0] ** 2 * lk - A[0])
    )
    N += (
        params_w.f_i[-1]
        / (2 * jnp.abs(wk2[-1]))
        * (tk[-1] ** 2)
        * (B[-1] ** 2 * lk - A[-1])
    )

    return eigenmode_params(
        tk=jnp.pad(tk / jnp.sqrt(N), 1),
        wk2=wk2,
        x0=params_q.x0,
        dx=params_q.dx,
    )


def init_radial_eigenmode_params(
    l, n, potential_params, r_min, r_max, a, b, N, potential=gravitational_potential
):
    k = n + 1
    V0 = jnp.abs(potential(0.0, potential_params))
    Emax = potential(r_max, potential_params) + V0
    rwkb = wkb_estimate_of_rmax(r_max, l, potential_params, potential=potential)
    params_q = init_piecewise_constant_q(
        potential_params, l, V0, r_min, rwkb, a, b, N, potential=potential
    )
    params_w = init_piecewise_constant_w(r_min, rwkb, a, b, N)
    E_l, E_u = bound_eigenvalue_k(k, Emax, params_w, params_q)

    # Second pass
    # r_max = right_turning_point(0.5 * (E_l + E_u), a, b, params_w, params_q)
    # rwkb = wkb_estimate_of_rmax(
    #     r_max, l, potential_params, nfold=50, potential=potential
    # )
    # params_q = init_piecewise_constant_q(
    #     potential_params, l, V0, r_min, rwkb, a, b, N, potential=potential
    # )
    # params_w = init_piecewise_constant_w(r_min, rwkb, a, b, N)
    # E_l, E_u = bound_eigenvalue_k(k, Emax, params_w, params_q)

    return radial_eigenmode_params(
        eigenmode_params=init_eigenmode_params_between(E_l, E_u, params_w, params_q),
        a=a,
        b=b,
        l=l,
        n=n,
        E=0.5 * (E_l + E_u) - V0,
    )


def init_eigenstate_library(
    potential_params,
    r_min,
    r_max,
    a,
    b,
    N,
    batch_size=16,
    potential=gravitational_potential,
):
    init_radial_eigenmodes = jax.jit(
        map_vmap(
            init_radial_eigenmode_params,
            in_axes=(0, 0, None, None, None, None, None, None),
            batch_size=batch_size,
        ),
        static_argnames="N",
    )
    compute_all_nmax = jax.vmap(nmax, in_axes=(0, None, None, None, None, None, None))
    ll = lmax(potential_params, r_min, r_max, a, b, N, potential=potential)
    nn = compute_all_nmax(
        jnp.arange(ll), potential_params, r_min, r_max, a, b, N, potential=potential
    )

    result_shape = jax.ShapeDtypeStruct((), jnp.int64)
    name = jax.pure_callback(
        eigenstate_library.compute_name,
        result_shape,
        potential_params,
        r_min,
        r_max,
        a,
        b,
        N,
    )
    if nn.shape[0] == 0:
        return None

    ls = jnp.repeat(jnp.arange(ll), nn)
    ns = jnp.concatenate([jnp.arange(n) for n in nn])

    return eigenstate_library(
        radial_eigenmode_params=init_radial_eigenmodes(
            ls, ns, potential_params, r_min, r_max, a, b, N, potential=potential
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
    x = jnp.clip(x, x0, x0 + dx * eigenmode_params.wk2.shape[0])
    k = ((x - x0) // dx).astype(int)

    wk2 = eigenmode_params.wk2[k]
    wk = jnp.sqrt(jnp.abs(wk2))

    xk = dx * (k + 1) + x0
    xkm1 = xk - dx

    A = eigenmode_params.tk[k]
    B = eigenmode_params.tk[k + 1]

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
    a = radial_eigenmode_params.a
    b = radial_eigenmode_params.b
    x = x_of_r(r, a, b)
    return eval_eigenmode(x, radial_eigenmode_params.eigenmode_params) / jnp.sqrt(
        b * r + a * r**2
    )
