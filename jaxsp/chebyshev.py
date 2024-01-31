import jax.numpy as jnp
import numpy as np


def chebyshev_pts(N):
    """
    Maxima of Chebyshev polynomials
    """
    x = jnp.sin(jnp.pi * ((N - 1) - 2 * jnp.linspace(N - 1, 0, N)) / (2 * (N - 1)))
    return x[::-1]


def clenshaw_curtis_weights(n):
    """
    Integration weights for Clenshar-Curtis quadrature
    TODO: JAX implementation, without in-place array ops

    Stolen from:
        https://people.math.sc.edu/Burkardt/py_src/quadrule/clenshaw_curtis_compute.py
    """
    i = np.arange(n)
    theta = (n - 1 - i) * np.pi / (n - 1)

    w = np.zeros(n)

    for i in range(0, n):
        w[i] = 1.0

        jhi = (n - 1) // 2

        for j in range(0, jhi):
            if 2 * (j + 1) == (n - 1):
                b = 1.0
            else:
                b = 2.0

            w[i] = w[i] - b * np.cos(2.0 * float(j + 1) * theta[i]) / float(
                4 * j * (j + 2) + 3
            )

    w[0] = w[0] / float(n - 1)
    for i in range(1, n - 1):
        w[i] = 2.0 * w[i] / float(n - 1)
    w[n - 1] = w[n - 1] / float(n - 1)

    return jnp.asarray(w[::-1])
