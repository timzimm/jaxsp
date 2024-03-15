from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import while_loop


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def lambertw(z: jnp.ndarray, tol: float = 1e-6, max_iter: int = 100) -> jnp.ndarray:
    """
    Lambert W function. Taken from
    https://github.com/ott-jax/ott/blob/main/src/ott/math/utils.py#L240
    """

    def initial_iacono(x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.sqrt(1.0 + jnp.e * x)
        num = 1.0 + 1.14956131 * y
        denom = 1.0 + 0.45495740 * jnp.log1p(y)
        return -1.0 + 2.036 * jnp.log(num / denom)

    def _initial_winitzki(z: jnp.ndarray) -> jnp.ndarray:
        log1p_z = jnp.log1p(z)
        return log1p_z * (1.0 - jnp.log1p(log1p_z) / (2.0 - log1p_z))

    def cond_fun(cont):
        it, converged, _ = cont
        return jnp.logical_and(jnp.any(~converged), it < max_iter)

    def hailley_iteration(cont):
        it, _, w = cont

        f = w - z * jnp.exp(-w)
        delta = f / (w + 1.0 - 0.5 * (w + 2.0) * f / (w + 1.0))
        w_next = w - delta

        not_converged = jnp.abs(delta) <= tol * jnp.abs(w_next)
        return it + 1, not_converged, w_next

    w0 = initial_iacono(z)
    converged = jnp.zeros_like(w0, dtype=bool)

    _, _, w = while_loop(
        cond_fun=cond_fun, body_fun=hailley_iteration, init_val=(0, converged, w0)
    )
    return w


@lambertw.defjvp
def lambertw_jvp(tol: float, max_iter: int, primals, tangents):
    (z,) = primals
    (dz,) = tangents
    w = lambertw(z, tol=tol, max_iter=max_iter)
    pz = jnp.where(z == 0, 1.0, w / ((1.0 + w) * z))
    return w, pz * dz
