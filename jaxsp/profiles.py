import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from functools import partial

from .constants import GN


class MassProfile:
    def __init__(self, data):
        self.rgrid = data[:, 0]
        self.rmin = self.rgrid[0]
        self.rmax = self.rgrid[-1]
        self.Mgrid = data[:, 1]
        self.dMgrid = jnp.maximum(data[:, 2] - data[:, 1], data[:, 3] - data[:, 1])
        self.log10rgrid = jnp.log10(self.rgrid)
        self.log10Mgrid = jnp.log10(self.Mgrid)
        self.log10dMgrid = jnp.log10(self.dMgrid)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, rkpc):
        return (
            self.Mgrid[0] * jnp.heaviside(self.rmin - rkpc, 0.5)
            + jnp.heaviside(rkpc - self.rmin, 0.5)
            * self._mass_interpolation(rkpc, self.log10rgrid, self.log10Mgrid)
            * jnp.heaviside(self.rmax - rkpc, 0.5)
            + self.Mgrid[-1] * jnp.heaviside(rkpc - self.rmax, 0.5),
            self.dMgrid[0] * jnp.heaviside(self.rmin - rkpc, 0.5)
            + jnp.heaviside(rkpc - self.rmin, 0.5)
            * self._mass_interpolation(rkpc, self.log10rgrid, self.log10dMgrid)
            * jnp.heaviside(self.rmax - rkpc, 0.5)
            + self.dMgrid[-1] * jnp.heaviside(rkpc - self.rmax, 0.5),
        )

    def _mass_interpolation(self, rkpc, log10rgrid, log10Mgrid):
        return 10 ** jnp.interp(jnp.log10(rkpc), log10rgrid, log10Mgrid)

    def __repr__(self):
        return f"MassProfile(rmin={self.rmin} kpc, max={self.rmax} kpc, npoints={len(self.rgrid)})"

    def plot(self):
        import matplotlib.pyplot as plt

        plt.loglog(self.rgrid, self.Mgrid, c="k", lw=1.0)
        plt.xlabel(r"$r\,\mathrm{[kpc]}$")
        plt.ylabel(r"$M(r)\,\mathrm{[M_\odot]}$")
        plt.xlim(self.rmin, self.rmax)
        plt.show()


class DensityProfile:
    def __init__(self, mass):
        self.mass = mass
        self.rmin = mass.rmin
        self.rmax = mass.rmax
        self.rgrid = mass.rgrid

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, rkpc):
        return self._dm_dr(rkpc) / (4.0 * jnp.pi * rkpc**2)

    @partial(jax.jit, static_argnums=(0,))
    def _log10M(self, log10rkpc):
        return jnp.log10(self.mass(10**log10rkpc)[0])

    @partial(jax.jit, static_argnums=(0,))
    def _dlog10Mdlog10r(self, log10rkpc):
        return jax.vmap(jax.grad(self._log10M))(log10rkpc)

    @partial(jax.jit, static_argnums=(0,))
    def _dm_dr(self, rkpc):
        return self.mass(rkpc)[0] * self._dlog10Mdlog10r(jnp.log10(rkpc)) / rkpc

    def __repr__(self):
        return f"DensityProfile(rmin={self.rmin} kpc, max={self.rmax} kpc, mass={self.mass})"

    def plot(self):
        import matplotlib.pyplot as plt

        plt.loglog(self.rgrid, self(self.rgrid), c="k", lw=1.0)
        plt.xlabel(r"$r\,\mathrm{[kpc]}$")
        plt.ylabel(r"$\rho(r)\,\mathrm{[M_\odot kpc}^{-3}\mathrm{]}$")
        plt.xlim(self.rmin, self.rmax)
        plt.show()


class GravPotential:
    def __init__(self, mass, rpts=1000, precache=True):
        self.mass = mass
        self.mtot = self.mass(self.mass.rmax)[0]
        self.rmin = self.mass.rmin
        self.rmax = self.mass.rmax
        self.rpts = rpts
        self.GN = GN()
        self._integrator = self.get_integrator(self.rpts)
        self._cached = False
        self._potential_interpolation = jnp.vectorize(lambda x: 0.0)
        if precache:
            self._cache()

    def __call__(self, rkpc):
        return (
            jnp.heaviside(self.rmin - rkpc, 0.5)
            * jax.lax.cond(
                self._cached,
                self._potential_interpolation,
                self._potential_inside,
                jnp.array([self.rmin]),
            )
            + jnp.heaviside(rkpc - self.rmin, 0.5)
            * jax.lax.cond(
                self._cached,
                self._potential_interpolation,
                self._potential_inside,
                rkpc,
            )
            * jnp.heaviside(self.rmax - rkpc, 0.5)
            + self._potential_outside(rkpc) * jnp.heaviside(rkpc - self.mass.rmax, 0.5)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _potential_inside(self, rkpc):
        return (
            -self.GN.value * self._integrator(rkpc, self.mass)
            - self.GN.value * self.mtot / self.mass.rmax
        )

    @partial(jax.jit, static_argnums=(0,))
    def _potential_outside(self, rkpc):
        return -self.GN.value * self.mtot / rkpc

    def get_integrator(self, rpts):
        def _int(rkpc, mass):
            rint = jnp.linspace(rkpc, mass.rmax, rpts)
            return trapezoid(mass(rint)[0] / rint**2, rint)

        _int_vm = jax.vmap(_int, in_axes=(0, None))
        return _int_vm

    def _cache(self):
        self.rgrid = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.rpts)
        self.phigrid = self(self.rgrid)

        def _potential_interpolation(rkpc):
            return -(
                10
                ** jnp.interp(
                    jnp.log10(rkpc), jnp.log10(self.rgrid), jnp.log10(-self.phigrid)
                )
            )

        self._potential_interpolation = jax.jit(_potential_interpolation)
        self.__call__ = jax.jit(self.__call__)
        self._cached = True

    def __repr__(self):
        return f"GravPotential(\n\trmin={self.rmin} kpc, \n\trmax={self.rmax} kpc, \n\tnpoints={self.rpts}, \n\tcached={self._cached}, \n\tmass={self.mass})\n)"

    def plot(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(2, 1, 1)
        plt.loglog(self.mass.rgrid, self.mass.Mgrid, c="k", lw=1.0)
        plt.xlabel(r"$r\,\mathrm{[kpc]}$")
        plt.ylabel(r"$M(r)\,\mathrm{[M_\odot]}$")
        plt.xlim(self.rmin, self.rmax)
        ax = plt.subplot(2, 1, 2)
        if self._cached:
            plt.semilogx(self.rgrid, -self.phigrid, c="k", lw=1.0)
        else:
            plt.semilogx(
                self.rgrid,
                -self(jnp.linspace(self.rmin, self.rmax, 1000)),
                c="k",
                lw=1.0,
            )
        plt.xlabel(r"$r\,\mathrm{[kpc]}$")
        plt.ylabel(r"$-V(r)\,\mathrm{[kpc}^2 \, \mathrm{Gyr}^{-2}\,\mathrm{]}$")
        plt.xlim(self.rmin, self.rmax)
        plt.tight_layout()
        plt.show()
