import numpy as np
import jax.numpy as jnp

_glx, _glw = np.polynomial.legendre.leggauss(12)
_glx = jnp.asarray(0.5 * (_glx + 1))
_glw = jnp.asarray(0.5 * _glw)


def quad(f, a, b):
    """
    Fixed order (order=12) Gauss-Legendre quadrature for integration of f(x)
    from x=a to x=b
    """
    x_i = a + (b - a) * _glx
    return (b - a) * f(x_i) @ _glw
