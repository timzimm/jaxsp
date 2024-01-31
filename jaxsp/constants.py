import chex


@chex.dataclass
class om:
    """DM mass density at z=0"""

    value: float = 0.245

    @property
    def unit(self):
        return ""

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"h: {self.value} {self.unit}"


@chex.dataclass
class h:
    """Little h"""

    value: float = 0.7

    @property
    def unit(self):
        return ""

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"h: {self.value} {self.unit}"


@chex.dataclass
class m22:
    """Axion mass scale"""

    value: float = 1.78266192e-55

    @property
    def unit(self):
        return "g"

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"m22: {self.value} {self.unit}"


@chex.dataclass
class Msun:
    """Solar mass"""

    value: float = 1.988409871e33

    @property
    def unit(self):
        return "g"

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"Msun: {self.value} {self.unit}"


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
    """Critical overdensity of the universe"""

    value: float = 9.20387392292102e-30

    @property
    def unit(self):
        return "g/cm^3"

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"rho_crit: {self.value} {self.unit}"


@chex.dataclass
class hbar:
    """Reduced Planck constant"""

    value: float = 1.0545919e-27

    @property
    def unit(self):
        return "erg s-1"

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"hbar: {self.value} {self.unit}"


@chex.dataclass
class GN:
    """Gravitational constant"""

    value: float = 6.6743e-8

    @property
    def unit(self):
        return "cm3 g-1 s-2"

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"GN: {self.value} {self.unit}"


@chex.dataclass
class c:
    """Speed of light"""

    value: float = 2.99792458e10

    @property
    def unit(self):
        return "cm s-1"

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"c: {self.value} {self.unit}"
