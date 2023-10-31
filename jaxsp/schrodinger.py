import jax
import jax.numpy as jnp
from functools import partial
import warnings
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    PIDController,
    Dopri5,
)

from .constants import mah_factor, maV_factor, to_kpc_factor


class Schrodinger:
    def __init__(self, ma, potential):
        self.potential = potential
        self.ma = ma
        self._solver = Dopri5()
        self.maV_factor = maV_factor()
        self.mah_factor = mah_factor()
        self.to_kpc_factor = to_kpc_factor()
        if not self.potential._cached:
            self.potential._cache()
        self.Emin = (
            self.maV_factor.value
            * self.ma
            * self.potential(jnp.array([self.potential.rmin]))[0]
        )
        self.Emax = (
            self.maV_factor.value
            * self.ma
            * self.potential(jnp.array([self.potential.rmax]))[0]
        )

    def schrodinger_solver(
        self,
        l,
        Enl,
        rmax=None,
        rtol=1e-8,
        atol=1e-8,
        max_steps=16**3,
        maxstep_warnings=False,
    ):
        try:
            return self._schrodinger_solver(l, Enl, rmax, rtol, atol, max_steps)
        except Exception as e:
            if "max_steps" in str(e):
                max_steps = max_steps * 16
                if maxstep_warnings:
                    warnings.warn(
                        f"Mass integrator had to increase max_steps in the integration to {max_steps}. Consider using a larger value."
                    )
                return self.schrodinger_solver(
                    l, Enl, rmax, rtol, atol, max_steps, maxstep_warnings
                )
            # reraise the exception if it is not due to max-steps being hit
            raise e

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "max_steps",
        ),
    )
    def _schrodinger_solver(
        self, l, Enl, rmax=None, rtol=1e-2, atol=1e-2, max_steps=16**3
    ):
        U0 = jnp.array(
            [self.potential.rmin ** (l + 1), (l + 1) * self.potential.rmin**l]
        )
        args = {"l": l, "Enl": Enl}
        term = ODETerm(self._schrodinger_derivative)
        solver = self._solver
        saveat = SaveAt(t0=False, t1=True, ts=None, dense=True)
        stepsize_controller = PIDController(rtol=rtol, atol=atol)
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=self.potential.rmin,
            t1=rmax if rmax is not None else self.potential.rmax,
            y0=U0,
            dt0=None,
            args=args,
            max_steps=max_steps,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        return solution

    @partial(jax.jit, static_argnums=(0,))
    def _schrodinger_derivative(self, rkpc, U, args):
        l, Enl = args["l"], args["Enl"]
        return jnp.array(
            [
                U[1],
                (
                    U[0]
                    * (
                        l * (l + 1) / rkpc**2
                        + 2
                        * (self.mah_factor.value * self.to_kpc_factor.value)
                        * self.ma
                        * (
                            self.maV_factor.value
                            * self.ma
                            * self.potential(jnp.array([rkpc]))[0]
                            - Enl
                        )
                    )
                ),
            ]
        )

    def __repr__(self):
        return f"Schrodinger(\n\tpotential={self.potential}, \n\tma={self.ma} eV, \n\tEmin={self.Emin} kg nm^2 / Gyr^2, \n\tEmax={self.Emax} kg nm^2 / Gyr^2\n)"
