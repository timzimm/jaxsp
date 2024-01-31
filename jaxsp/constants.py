import chex
import jax.numpy as jnp


@chex.dataclass
class Delta:
    """Virial overdensity parameter"""

    value: float = 200

    @property
    def unit(self):
        return ""

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"Delta: {self.value} {self.unit}"


@chex.dataclass
class rho_crit:
    """Critical overdensity parameter"""

    value: float = 135.05

    @property
    def unit(self):
        return "M_sun/kpc^3"

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"rho_crit: {self.value} {self.unit}"


@chex.dataclass
class GN:
    """Gravitational constant in kpc3 Msun-1 Gyr-2"""

    value: float = 4.498502151469553e-6

    @property
    def unit(self):
        return "kpc3 Msun-1 Gyr-2"

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"GN: {self.value} {self.unit}"


@chex.dataclass
class maV_factor:
    """Conversion factor from eV/c^2 kpc^2/Gyr^2 to kg nm^2/Gyr^2"""

    value: float = 1.6973448160638636e21

    @property
    def unit(self):
        return "kg nm^2/Gyr^2"

    def __repr__(self):
        return f"maV_factor: {self.value} {self.unit}"


@chex.dataclass
class mah_factor:
    """Conversion factor from eV/c^2 / ħ^2 to s^2 / kg / nm^4"""

    value: float = 0.00016029377826686628

    @property
    def unit(self):
        return "s^2 / kg / nm^4"

    def __repr__(self):
        return f"mah_factor: {self.value} {self.unit}"


@chex.dataclass
class to_kpc_factor:
    """Conversion factor from (s/Gyr)^2 / nm^2 to 1/kpc^2"""

    value: float = 9.560776199153959e23

    @property
    def unit(self):
        return "1/kpc^2"

    def __repr__(self):
        return f"to_kpc_factor: {self.value} {self.unit}"


@chex.dataclass
class c:
    """Speed of light in kpc/Gyr"""

    value: float = 2.99792458e8

    @property
    def unit(self):
        return "m/s"

    def __repr__(self):
        return f"c: {self.value} {self.unit}"


@chex.dataclass
class Gyr_to_s:
    """Conversion factor from Gyr to s"""

    value: float = 3.15576e16

    @property
    def unit(self):
        return "s"

    def __repr__(self):
        return f"Gyr_to_s: {self.value} {self.unit}"


@chex.dataclass
class kpc_to_m:
    """Conversion factor from kpc to m"""

    value: float = 3.08567758e19

    @property
    def unit(self):
        return "m"

    def __repr__(self):
        return f"kpc_to_m: {self.value} {self.unit}"


@chex.dataclass
class ma:
    """Axion mass in eV/c^2"""

    value: float = 1e-22

    @property
    def unit(self):
        return "eV/c^2"

    @property
    def compton_wavelength(self):
        return length(
            value=1.2398419738620804e-6 / self.value / kpc_to_m().value, unit="kpc"
        )

    def length_scale(self, potential_kpc_Gyr):
        velocity = (
            jnp.sqrt(jnp.abs(potential_kpc_Gyr)) * kpc_to_m().value / Gyr_to_s().value
        )
        return length(
            value=self.compton_wavelength.value * c().value / velocity,
            unit=self.compton_wavelength.unit,
        )

    def __repr__(self):
        return f"ma: {self.value} {self.unit}"


@chex.dataclass
class length:
    value: float = 1.0
    unit: str = "kpc"

    def __repr__(self):
        return f"{self.value} {self.unit}"


@chex.dataclass
class time:
    value: float = 1.0
    unit: str = "Gyr"

    def __repr__(self):
        return f"{self.value} {self.unit}"


@chex.dataclass
class mass:
    value: float = 1.0
    unit: str = "Msol"

    def __repr__(self):
        return f"{self.value} {self.unit}"
