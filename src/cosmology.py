# cosmology.py
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import hyp2f1


@dataclass(frozen=True)
class Cosmology:
    """
    Minimal flat ΛCDM cosmology wrapper.

    Parameters
    ----------
    h : float
        Dimensionless Hubble parameter, H0 = 100 h km/s/Mpc.
    Om0 : float
        Matter density parameter at z=0.
    Ode0 : float
        Dark energy density parameter at z=0.
        For flat models, Ode0 = 1 - Om0 - Ok0.
    ns : float
        Scalar spectral index of the primordial power spectrum.
    sigma8 : float
        RMS linear mass fluctuation in 8 Mpc/h spheres at z=0.

    Notes
    -----
    - This is intentionally minimal: you can extend it later
      (radiation, w(z), etc.) if needed.
    - Units:
        * H0 in km/s/Mpc
        * rho_crit0 in Msun / Mpc^3
    """
    h: float = 0.6777
    Om0: float = 0.307
    Ode0: float = 0.693
    ns: float = 0.9611
    sigma8: float = 0.8288

    # ------------------------------------------------------------------
    # Alternative constructors for common cosmologies
    # ------------------------------------------------------------------

    @classmethod
    def planck13(cls) -> "Cosmology":
        """
        Planck Collaboration, 2014, A&A, 571, A16 (Paper XVI),
        Table 2 (Planck + lensing) & Table 5 (Planck + WP + highL + BAO)
        """
        return cls(
            h=0.6777, 
            Om0=0.307, 
            Ode0=0.693, 
            ns=0.9611, 
            sigma8=0.8288
        )

    @classmethod
    def planck15(cls) -> "Cosmology":
        """
        Planck Collaboration, 2016, A&A, 594, A13 (Paper XIII), 
        Table 4 (TT, TE, EE + lowP + lensing + ext)
        """
        return cls(
            h=0.6774, 
            Om0=0.3089, 
            Ode0=0.6911,
            ns=0.9667,
            sigma8=0.8159
        )

    @classmethod
    def planck18(cls) -> "Cosmology":
        """
        Planck Collaboration, 2020, A&A, 641, A6 (Paper VI), 
        Table 2 (TT, TE, EE + lowE + lensing + BAO)
        """
        return cls(
            h=0.6766,
            Om0=0.3111,
            Ode0=0.6889,
            ns=0.9665,
            sigma8=0.8102
        )

    # ------------------------------------------------------------------
    # Basic derived quantities
    # ------------------------------------------------------------------

    @property
    def H0(self) -> float:
        """Hubble constant at z=0 in km/s/Mpc."""
        return 100.0 * self.h

    @property
    def Ok0(self) -> float:
        """Curvature density parameter at z=0."""
        return 1.0 - self.Om0 - self.Ode0

    # ------------------------------------------------------------------
    # Expansion factor and growth
    # ------------------------------------------------------------------

    def E(self, z: float) -> float:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.

        For flat ΛCDM with optional curvature:

            E(z)^2 = Om0 (1+z)^3 + Ok0 (1+z)^2 + Ode0
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zp1 = 1.0 + z
            return np.sqrt(self.Om0 * np.power(zp1, 3) + self.Ok0 * np.power(zp1, 2) + self.Ode0)

    def H(self, z: float) -> float:
        """Hubble parameter H(z) in km/s/Mpc."""
        return self.H0 * self.E(z)

    # ------------------------------------------------------------------
    # Linear growth factor D(z), normalized to D(0) = 1
    # ------------------------------------------------------------------

    def growth_factor(self, z: float) -> float:
        """
        Linear growth factor D(z) normalized such that D(0) = 1.

        Uses the integral expression (Carroll+ 1992 style):

            D(a) ∝ E(a)^(-3) ∫^a da' [a'^3 / E(a')^3]

        and then normalized by D(a=1).

        This is a simple numeric implementation, sufficiently accurate
        for most HMF applications.
        """

        # Unnormalized growth factor D(a)
        def _D_of_a(self, a, l, p=1.0):
            a3 = np.power(a, 3)
            return np.power(a, p) * np.sqrt(1 + l * a3) * \
                   hyp2f1((2. * p + 7.) / 6., (2. * p + 3.) / 6., (4. * p + 7.) / 6., -l * a3)

        a = 1.0 / (1.0 + z)
        l = self.Ode0 / self.Om0

        # no neutrino, features for later versions
        f_nu = 0.0  
        p = np.sqrt(25. - 24. * f_nu) / 4. - 0.25

        # e.g., self.growth_factor(z = 1) ~ 0.609 for Planck15, reciprocal to N-Genic (1.641)
        return _D_of_a(self, a, l, p) / _D_of_a(self, 1.0, l, p)


    # ------------------------------------------------------------------
    # Critical density today
    # ------------------------------------------------------------------
    @property
    def rho_crit0(self) -> float:
        """
        Critical density at z=0 in units of Msun / Mpc^3.

        Uses:
            rho_crit0 = 3 H0^2 / (8πG)

        with G in (km/s)^2 * Mpc / Msun, so that units work out.
        """
        # Gravitational constant in (km/s)^2 * Mpc / Msun
        G = 4.30091e-9  # (km/s)^2 * Mpc / Msun

        H0 = self.H0  # km/s/Mpc
        rho_crit0 = 3.0 * H0 * H0 / (8.0 * np.pi * G)  # Msun / Mpc^3
        return rho_crit0

    @property
    def rho_m0(self) -> float:
        """
        Mean matter density at z=0 in units of Msun / Mpc^3:

            rho_m0 = Om0 * rho_crit0
        """
        return self.Om0 * self.rho_crit0
    
    def rho_crit(self, z: float = 0.0) -> float:
        # use E(z) to compute rho_crit at redshift z
        return self.rho_crit0 * np.power(self.E(z), 2)
    
    def rho_m(self, z: float = 0.0) -> float:
        # use rho_crit(z) to compute rho_m at redshift z
        return self.rho_m0 * np.power(1 + z, 3)