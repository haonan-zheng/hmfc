# public/models/base.py
from __future__ import annotations

import numpy as np
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import warnings

class HMFModel:
    """
    Base class for all halo mass function models.

    Each concrete model must implement:
        - f_nu(self, nu): multiplicity function f(nu)

    This base class provides:
        - nu(M,z) from sigma(M,z)
        - dn/dlog10M via universal relation
        - hmf(M,z, quantity=...) as the main entry point
    """

    def __init__(self, cosmo, powerspec, delta_c=1.68647, **kwargs):
        """
        Parameters
        ----------
        cosmo : Cosmology
            Cosmology object.
        powerspec : PowerSpectrum
            Power spectrum object providing sigma_M(M,z).
        **kwargs
            Extra model-specific parameters.
        """
        self.cosmo = cosmo
        self.ps = powerspec
        self.delta_c = delta_c
        self.kwargs = kwargs

    # -----------------------------------------------------------
    # nu(M,z) and f(nu)
    # -----------------------------------------------------------

    def nu(self, m, z, scheme="default", overdensity=0.0, mass_region=1e99):
        """
        Peak height nu(M,z) = delta_c / sigma(M,z). 
        Parameters
        ----------
        m : array_like
            Mass in (converted) M200m.
        z : float
            Redshift.
        overdensity : float
            Overdensity of the region.
        mass_region : float
            Mass of the region in which the HMF is computed.
        scheme : str
            Scheme for computing nu. 
            "default": return nu
            "diff": return nu and dlnnu/dlnsigma 
        """

        warnings.simplefilter("ignore", RuntimeWarning)

        m = np.asarray(m)
        d1 = self.delta_c / self.cosmo.growth_factor(z)
        sigma = self.ps.sigma_M(m)

        if overdensity == 0.0 and mass_region >= 1e99:
            if scheme == "default":
                return d1 / sigma
            else:
                return d1 / sigma, np.ones_like(m)
        
        d0 = self.d0_dnl0(overdensity) / self.cosmo.growth_factor(z)

        s1 = sigma * sigma
        s0 = self.ps.s_M(mass_region)

        dd = d1 - d0
        ds = s1 - s0

        if scheme == "default":
            return dd * np.power(ds, -0.5)
        else:
            return dd * np.power(ds, -0.5), np.abs(s1 / ds)

    def d0_dnl0(self, dnl0):
        d0 = self.delta_c
        d0 = d0 / 1.68647 * (1.68647 - 1.35 * np.power(1 + dnl0, -2. / 3.) -
                            1.12431 * np.power(1 + dnl0, -0.5) + 0.78785 * np.power(1 + dnl0, -0.58661))
        x = np.min([0, np.log(1 + dnl0)])
        C_correct = 1 - 0.0053977 * x + 0.00184835 * x ** 2 + 0.00011834 * x ** 3 
        d0 = d0 * C_correct
        return d0

    def f_nu(self, nu):
        """
        Multiplicity function f(nu). Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement f_nu(nu).")

    # -----------------------------------------------------------
    # dn/dlog10M
    # -----------------------------------------------------------

    def dn_dlog10M(self, m, z, overdensity=0.0, mass_region=1e99):
        """
        Differential mass function:

            dn/dlog10M = (rho_m / M) * f(nu) * dlognu/dlogM

        where rho_m is the mean matter density at z=0.
        """
        m = np.asarray(m)
        rho_m = self.cosmo.rho_m0

        nu, dlnnu_dlnsigma = self.nu(m, z, scheme="diff", overdensity=overdensity, mass_region=mass_region)
        fnu = self.f_nu(nu, z = z)

        dlnsigma_dlogm = -self.ps.dlnsigma_dlogM(m)
        dnu_dlogm = dlnnu_dlnsigma * dlnsigma_dlogm

        return (rho_m * (1 + self.overdensity) / m) * fnu * dnu_dlogm

    # -----------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------

    def hmf(self, m, z, quantity="dn_dlog10M", overdensity=0.0, mass_region=1e99, jacobian=1.0, **kwargs):
        """
        Compute a chosen HMF-related quantity.

        Parameters
        ----------
        m : array_like
            Mass in the working mass definition (e.g. M200m).
        z : float
            Redshift.
        quantity : {'dn_dlog10M', 'f_nu'}, optional
            What to return. Default is 'dn_dlog10M'.
        overdensity, mass_region : float
            Currently unused here, but passed through for API compatibility.

        Returns
        -------
        ndarray
        """

        self.overdensity = overdensity
        self.mass_region = mass_region

        if quantity == "dn_dlog10M":
            # dn_dlog10M_work -> dn_dlog10M_input
            return self.dn_dlog10M(m, z, overdensity=overdensity, mass_region=mass_region) * jacobian
        if quantity == "dn_dM":
            # dn_dlog10M_work -> dn_dlog10M_input -> dn_dM_input
            return self.dn_dlog10M(m, z, overdensity=overdensity, mass_region=mass_region) * jacobian * (kwargs.get('m_input', m) * np.log(10.))
        if quantity == "n(>M)" or quantity == 'n':
            # \sum dn_dlog10M_work dlog10M_work, integral number density above M
            log10M_grid = self.ps.log10M_grid
            dn_dlog10M_grid = self.dn_dlog10M(10**log10M_grid, z, overdensity=overdensity, mass_region=mass_region)

            dn_dlog10M_grid = np.where(np.isnan(dn_dlog10M_grid), 0.0, dn_dlog10M_grid)
            n_gt_M_grid = -cumtrapz(dn_dlog10M_grid, log10M_grid, initial=0)

            return interp1d(log10M_grid, n_gt_M_grid)(np.log10(m))
        elif quantity == "f_nu" or quantity == "f(nu)":
            return self.f_nu(self.nu(m, z, overdensity=overdensity, mass_region=mass_region), z = z)
        elif quantity == 'class':
            return self
        else:
            raise ValueError(f"Unknown quantity='{quantity}'")