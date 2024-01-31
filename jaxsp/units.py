import jax.numpy as jnp

from .constants import h, om, hbar, Msun, GN, c, m22


class Units:
    """
    A very simple unit conversion helper class providing conversion factors
    from/to code units
    """

    cm = 1
    g = 1
    s = 1
    pc = 3.0856775815e18  # parsec in cm
    yr = 60 * 60 * 24 * 365.25  # year in seconds
    kms = 1.0e5  # velocity in km/s
    Kpc = 1.0e3 * pc  # kiloparsec
    Mpc = 1.0e6 * pc  # megaparsec
    Myr = 1.0e6 * yr  # megayear
    Gyr = 1.0e9 * yr  # gigayear
    Kpc_kms = Kpc * kms  # angular momentum
    Gev_per_cm3 = 1.782662e-24  # volume density in Gev/cm^3
    eV = 1.60218e-12  # erg

    def __init__(self, length_unit_in_cm, mass_unit_in_g, time_unit_in_s):
        self.__length_unit = length_unit_in_cm
        self.__mass_unit = mass_unit_in_g
        self.__time_unit = time_unit_in_s
        self.__action_unit = (
            self.__mass_unit * self.__length_unit**2 / self.__time_unit
        )
        self.from_cm = Units.cm / self.__length_unit
        self.from_g = Units.g / self.__mass_unit
        self.from_s = Units.s / self.__time_unit
        self.from_pc = Units.pc / self.__length_unit
        self.from_Msun = Msun.value / self.__mass_unit
        self.from_Kpc = Units.Kpc / self.__length_unit
        self.from_Mpc = Units.Mpc / self.__length_unit
        self.from_yr = Units.yr / self.__time_unit
        self.from_Myr = Units.Myr / self.__time_unit
        self.from_Gyr = Units.Gyr / self.__time_unit
        self.from_kms = Units.kms / (self.__length_unit / self.__time_unit)
        self.from_Kpc_kms = Units.Kpc_kms / (
            self.__length_unit * self.__length_unit / self.__time_unit
        )
        self.from_hbar = hbar.value / self.__action_unit
        self.from_m22 = m22.value / self.__mass_unit

        self.to_cm = 1.0 / self.from_cm
        self.to_g = 1.0 / self.from_g
        self.to_s = 1.0 / self.from_s

        self.to_Msun = 1.0 / self.from_Msun
        self.to_m22 = 1.0 / self.from_m22
        self.to_pc = 1.0 / self.from_pc
        self.to_Kpc = 1.0 / self.from_Kpc
        self.to_Mpc = 1.0 / self.from_Mpc
        self.to_yr = 1.0 / self.from_yr
        self.to_Myr = 1.0 / self.from_Myr
        self.to_Gyr = 1.0 / self.from_Gyr
        self.to_kms = 1.0 / self.from_kms
        self.to_Kpc_kms = 1.0 / self.from_Kpc_kms
        self.to_hbar = 1.0 / self.from_hbar


def set_schroedinger_units(m):
    """
    Factory function for Schroedinger code units (our main convention)
    """
    H0 = 100 * h.value * Units.kms / Units.Mpc
    rho_m = om.value * 3 * H0**2 / (8 * jnp.pi * GN.value)

    m = m * m22.value

    T = (3 / 2 * om.value * H0**2) ** (-1 / 2)
    L = jnp.sqrt(hbar.value / m) * (3 / 2 * om.value * H0**2) ** (-1 / 4)
    M = rho_m * L**3

    return Units(L, M, T)
