from typing import NamedTuple
import hashlib

import jax
import jax.numpy as jnp
from jaxopt import Broyden

from .constants import Delta, om
from .io_utils import hash_to_int32


class core_NFW_tides_params(NamedTuple):
    """
    Parameters specifying the cored-NFW-tides (cNFWt) density model introduced
    in arxiv:1805.06934. This is the internal dark matter halo model used by GravSphere
    """

    M200: float
    c: float
    n: float
    rc: float
    rt: float
    delta: float
    name: int

    @property
    def rs(self):
        return (3.0 * self.M200 * om.value / (4 * jnp.pi * Delta.value)) ** (
            1.0 / 3.0
        ) / self.c

    @property
    def g(self):
        return 1.0 / (jnp.log(1.0 + self.c) - self.c / (1.0 + self.c))

    @property
    def rho0(self):
        return 1.0 / om.value * Delta.value * self.c**3.0 * self.g / 3.0

    @classmethod
    def compute_name(cls, sample):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.array(sample[..., 0])).digest())
        combined.update(hashlib.md5(jnp.array(sample[..., 1])).digest())
        combined.update(hashlib.md5(jnp.array(sample[..., 2])).digest())
        combined.update(hashlib.md5(jnp.array(sample[..., 3])).digest())
        combined.update(hashlib.md5(jnp.array(sample[..., 4])).digest())
        combined.update(hashlib.md5(jnp.array(sample[..., 5])).digest())
        return hash_to_int32(combined.hexdigest())

    def __repr__(self):
        return (
            f"core_NFW_tides_params:\n\tM200={self.M200},"
            f"\n\tc={self.c},"
            f"\n\tn={self.n},"
            f"\n\trc={self.rc},"
            f"\n\trt={self.rt},"
            f"\n\tdelta={self.delta},"
            f"\n\tDerived:"
            f"\n\trs={self.rs},"
            f"\n\tg={self.g},"
            f"\n\trho0={self.rho0},"
        )


def init_core_NFW_tides_params_from_sample(sample):
    """
    Initialise cNFWt model from GravSphere chain sample
    """
    result_shape = jax.ShapeDtypeStruct((), jnp.int32)
    name = jax.pure_callback(core_NFW_tides_params.compute_name, result_shape, sample)
    return core_NFW_tides_params(
        M200=sample[0],
        c=sample[1],
        n=sample[2],
        rc=sample[3],
        rt=sample[4],
        delta=sample[5],
        name=name,
    )


def _M_NFW(r, M200, g, rs):
    """Enclosed mass of NFW density profile"""
    return M200 * g * (jnp.log(1.0 + r / rs) - r / rs / (1.0 + r / rs))


def _M_cNFW(r, n, rc, M_NFW_r):
    """Enclosed mass of the cored NFW density profile"""
    f = jnp.tanh(r / rc)
    return f**n * M_NFW_r


def _rho_cNFW(r, n, rc, rho_NFW_r, M_NFW_r):
    """Cored NFW density profile"""
    f = jnp.tanh(r / rc)
    return (
        f**n * rho_NFW_r
        + n * f ** (n - 1) * (1 - f**2) / (4 * jnp.pi * r**2 * rc) * M_NFW_r
    )


def _rho_NFW(r, rho0, rs):
    """NFW density profile"""
    return rho0 / ((r / rs) * (1.0 + (r / rs)) ** 2.0)


def core_nfw_tides_rho(r, params):
    """Density of the cNFWt profile"""
    M200 = params.M200
    g = params.g
    rc = params.rc
    n = params.n
    rt = params.rt
    delta = params.delta
    rs = params.rs
    rho0 = params.rho0

    return jnp.piecewise(
        r,
        [r < rt],
        [
            lambda r: _rho_cNFW(
                r, n, rc, _rho_NFW(r, rho0, rs), _M_NFW(r, M200, g, rs)
            ),
            lambda r: _rho_cNFW(
                rt, n, rc, _rho_NFW(rt, rho0, rs), _M_NFW(rt, M200, g, rs)
            )
            * (r / rt) ** (-delta),
        ],
    )


def core_nfw_tides_M(r, params):
    """Enclosed mass of the cNFWt profile"""
    M200 = params.M200
    rc = jnp.abs(params.rc)
    n = params.n
    rt = params.rt
    delta = params.delta
    rs = params.rs
    rho0 = params.rho0
    g = params.g

    return jnp.piecewise(
        r,
        [r < rt],
        [
            lambda r: _M_cNFW(r, n, rc, _M_NFW(r, M200, g, rs)),
            lambda r: _M_cNFW(rt, n, rc, _M_NFW(rt, M200, g, rs))
            + 4.0
            * jnp.pi
            * _rho_cNFW(rt, n, rc, _rho_NFW(rt, rho0, rs), _M_NFW(rt, M200, g, rs))
            * rt**3.0
            / (3.0 - delta)
            * ((r / rt) ** (3.0 - delta) - 1.0),
        ],
    )


def total_mass(params):
    """Total mass of the cNFWt profile"""
    return core_nfw_tides_M(jnp.inf, params)


def enclosing_radius(mass_fraction, params):
    """Radius enclosing mass_fraction % of the total cNFWt mass"""
    M = total_mass(params)

    @jax.jit
    def objective(r):
        return core_nfw_tides_M(r, params) / M - mass_fraction

    broyden = Broyden(fun=objective)
    return broyden.run(jnp.array(1.0)).params


enclosed_mass = core_nfw_tides_M
rho = core_nfw_tides_rho
