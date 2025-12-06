from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import pdb
import numpy as np
from scipy.interpolate import interp1d

from .cosmology import Cosmology

# ------------------------------------------------------------------
# only support top-hat window type for now
# you may extend _tophat_window() to sharp-k or Gaussian if needed
# ------------------------------------------------------------------
WindowType = Literal["TopHat"]


@dataclass
class PowerSpectrum:
    """
    Power spectrum wrapper.

    Supports:
    - "bbks" P(k) generated from cosmology with bbks formula. 
    - "table" P(k) based on user-provided (k, Pk), optionally normalized
      to match cosmology.sigma8.

    Units convention
    ----------------
    - k       : in Mpc^{-1}
    - P(k)    : in Mpc^{3}
    - R       : in Mpc (comoving)
    - sigma8  : RMS in an 8 Mpc top-hat sphere.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology object containing e.g. h, Om0, Ode0, ns, sigma8, and
        a growth_factor(z) method.
    scheme : {"cosmo", "table"}, optional
        How to interpret the power spectrum. "cosmo" means generated
        from cosmology; "table" means use a user-provided (k, Pk).
    kmin, kmax : float
        Internal k-grid bounds in Mpc^{-1}.
    nk : int, optional
        Number of k-grid points. If provided, it overrides dlnk.
    dlnk : float, optional
        Logarithmic spacing in ln(k). If nk is None, dlnk determines
        the number of k points. If dlnk is also None, defaults to 0.01.
    """
    cosmo: Cosmology
    scheme: str = "table"
    kmin: float = 1e-6
    kmax: float = 1e8
    nk: int | None = None
    dlnk: float | None = None

    # Only used when scheme == "table"
    _k_table: np.ndarray | None = field(default=None, repr=False)
    _pk_table: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Classmethod: build from a user-supplied (k, Pk) table
    # ------------------------------------------------------------------

    @classmethod
    def from_table(
        cls,
        cosmo: Cosmology,
        k: np.ndarray,
        pk: np.ndarray,
        normalize_to_cosmo: bool = True
    ) -> "PowerSpectrum":
        """
        Build a PowerSpectrum from a user-supplied (k, Pk) table.

        Parameters
        ----------
        cosmo : Cosmology
            Cosmology used for growth factor and target sigma8.
        k, pk : array_like
            1D arrays of wavenumber and power spectrum P(k) at z=0.
            Assumed units:
              - k  in Mpc^{-1}
              - Pk in Mpc^{3}
        normalize_to_cosmo : bool, optional
            If True, rescale P(k) so that sigma8 computed from the table
            matches cosmo.sigma8.

        Returns
        -------
        PowerSpectrum
            A PowerSpectrum instance using this table.
        """
        k = np.asarray(k)
        pk = np.asarray(pk)

        if k.ndim != 1 or pk.ndim != 1 or k.size != pk.size:
            raise ValueError("k and pk must be 1D arrays of the same length.")

        # Ensure k is sorted
        idx = np.argsort(k)
        k = k[idx]
        pk = pk[idx]

        ps = cls(cosmo=cosmo, scheme="table")
        ps._k_table = k
        ps._pk_table = pk

        if normalize_to_cosmo:
            sigma8_table = ps._sigma_R_from_table(R = 8.0 / cosmo.h)
            if sigma8_table <= 0:
                raise ValueError("Computed sigma8 from table is non-positive.")
            factor = np.power(cosmo.sigma8 / sigma8_table, 2)
            ps._pk_table = pk * factor

        return ps

    # ------------------------------------------------------------------
    @classmethod
    def from_formula(
        cls,
        cosmo: Cosmology,
        model: str = "bbks",
        kmin: float | None = None,
        kmax: float | None = None,
        nk: int | None = None,
        normalize_to_cosmo: bool = True
    ) -> "PowerSpectrum":
        """
        Build a PowerSpectrum from an analytic formula (e.g. BBKS).

        This constructs an internal k-grid (like a normal PowerSpectrum),
        evaluates P(k) on that grid, and then stores it as a 'table'
        spectrum (scheme='table') so that the rest of the API can reuse
        the same machinery as from_table().

        Parameters
        ----------
        cosmo : Cosmology
        model : {'bbks'}, optional
            Analytic shape to use. Currently only 'bbks' is implemented.
        kmin, kmax : float, optional
            If given, override the default kmin/kmax of cls.
        nk : int, optional
            If given, override the default nk of cls.
        normalize_to_cosmo : bool, optional
            If True, renormalise to match cosmo.sigma8.

        Returns
        -------
        PowerSpectrum
        """
        # Create an instance to define k_grid
        ps = cls(
            cosmo=cosmo,
            scheme="table",               
            kmin=kmin if kmin is not None else cls.kmin,
            kmax=kmax if kmax is not None else cls.kmax,
            nk=nk if nk is not None else cls.nk,
            dlnk=None 
        )

        k = ps.k_grid 

        if model == 'bbks':
            Om = cosmo.Om0
            h  = cosmo.h
            ns = cosmo.ns

            # Shape parameter Gamma = omega_m h (no baryon correction here)
            Gamma = Om * h

            # k is in Mpc^{-1}. BBKS uses k in h Mpc^{-1}.
            # => k_h = k / h
            # => q = k_h / Gamma = k / (Gamma * h) = k / (Ω_m h^2)
            q = k / (Gamma * h)

            T = np.log(1.0 + 2.34 * q) / (2.34 * q) * \
                np.power(1.0 + 3.89 * q + np.power(16.1 * q, 2)
                            + np.power(5.46 * q, 3) + np.power(6.71 * q, 4), -0.25)

            # Primordial power = k^ns
            pk = np.power(k, ns) * T * T
        else:
            raise NotImplementedError(f"Formula model '{model}' not implemented.")
        
        ps = cls(cosmo=cosmo, scheme=model)
        ps._k_table = k
        ps._pk_table = pk
        
        if normalize_to_cosmo:
            sigma8_table = ps._sigma_R_from_table(R = 8.0 / cosmo.h)
            if sigma8_table <= 0:
                raise ValueError("Computed sigma8 from table is non-positive.")
            factor = np.power(cosmo.sigma8 / sigma8_table, 2)
            ps._pk_table = pk * factor

        return ps

    # ------------------------------------------------------------------
    # Basic k-grid and P(k,z=0) on that grid
    # ------------------------------------------------------------------

    @cached_property
    def ln_k_grid(self) -> np.ndarray:
        """
        Internal log(k) grid for derived quantities like sigma(M).

        - If nk is provided, use nk points.
        - Else, use dlnk (or default dlnk=0.01) to determine the number
          of k points.
        """
        ln_kmin = np.log(self.kmin)
        ln_kmax = np.log(self.kmax)

        if self.nk is not None:
            # nk overrides dlnk
            return np.linspace(ln_kmin, ln_kmax, self.nk)

        # Use dlnk to determine nk
        dlnk = self.dlnk if self.dlnk is not None else 0.01
        span = ln_kmax - ln_kmin
        # number of intervals ~ span/dlnk → number of points = intervals + 1
        n_intervals = max(1, int(np.round(span / dlnk)))
        n_points = n_intervals + 1

        return np.linspace(ln_kmin, ln_kmax, n_points)

    @cached_property
    def k_grid(self) -> np.ndarray:
        """k-grid in Mpc^{-1}."""
        return np.exp(self.ln_k_grid)

    @cached_property
    def Pk0_grid(self) -> np.ndarray:
        """
        P(k, z=0) evaluated on the internal k-grid.

        - If scheme == "table": interpolate from the user-provided table.
        - If scheme == "formula": use bbks-like analytic form, renormalized

        Units: P(k) in Mpc^{3}.
        """
        k = self.k_grid

        if self.scheme == "table" or self.scheme == "bbks":
            if self._k_table is None or self._pk_table is None:
                raise RuntimeError(
                    "scheme='table' or 'bbks' but no _k_table and _pk_table table has been set."
                )
            interp = interp1d(
                np.log(self._k_table),
                np.log(self._pk_table),
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate"
            )
            return np.exp(interp(np.log(k)))
        else:
            raise NotImplementedError(f"Power spectrum scheme '{self.scheme}' not implemented.")

    # ------------------------------------------------------------------
    # Public interface: P(k,z)
    # ------------------------------------------------------------------

    def Pk(self, k: np.ndarray | float, z: float = 0.0) -> np.ndarray:
        """
        Matter power spectrum P(k, z).

        Uses the internal representation at z=0 and scales with the
        linear growth factor: P(k,z) = P(k,0) * [D(z)/D(0)]^2.

        Parameters
        ----------
        k : array_like or float
            Wavenumber in Mpc^{-1}.
        z : float
            Redshift.

        Returns
        -------
        P(k,z) in Mpc^{3}.
        """
        k = np.asarray(k)
        interp = interp1d(
            np.log(self.k_grid),
            np.log(self.Pk0_grid),
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
        ln_pk0 = interp(np.log(k))
        pk0 = np.exp(ln_pk0)

        Dz = self.cosmo.growth_factor(z)
        D0 = self.cosmo.growth_factor(0.0)
        return pk0 * np.power(Dz / D0, 2)

    # ------------------------------------------------------------------
    # sigma(R) helpers (used for sigma8 renormalization & later sigma(M))
    # ------------------------------------------------------------------

    @staticmethod
    def _tophat_window(kR: np.ndarray) -> np.ndarray:
        """
        Real-space top-hat window in Fourier space:

        W(kR) = 3 [sin(kR) - kR cos(kR)] / (kR)^3
        """
        kR = np.asarray(kR)
        w = np.ones_like(kR)

        mask = kR != 0
        kR = kR[mask]
        w[mask] = 3.0 * (np.sin(kR) - kR * np.cos(kR)) / np.power(kR, 3)

        return w

    @classmethod
    def _sigma_R_from_Pk(
        cls,
        k: np.ndarray,
        pk: np.ndarray,
        R: float
    ) -> float:
        """
        Compute sigma(R) from a given P(k) via:

            sigma^2(R) = 1/(2π^2) ∫ dk k^2 P(k) W^2(kR)

        Units assumed:
            - k   in Mpc^{-1}
            - P(k) in Mpc^{3}
            - R   in Mpc
        """
        k = np.asarray(k)
        pk = np.asarray(pk)

        kR = k * R
        W = cls._tophat_window(kR)
        integrand = pk * np.power(k * W, 2)

        integrand = integrand * k  # change dk to dlnk: dk = k dlnk
        ln_k = np.log(k)
        dlnk = np.diff(ln_k)
        avg = 0.5 * (integrand[1:] + integrand[:-1])
        integral = np.sum(avg * dlnk)

        sigma2 = integral * (4 * np.pi)
        return np.sqrt(sigma2)

    def _sigma_R_from_table(self, R: float) -> float:
        """
        Compute sigma(R) using the current table P(k), regardless of the
        internal k_grid.
        """
        if self._k_table is None or self._pk_table is None:
            raise RuntimeError("No (k, Pk) table set for 'table' scheme.")

        return self._sigma_R_from_Pk(self._k_table, self._pk_table, R=R)

    # ------------------------------------------------------------------
    # sigma(M) (cached)
    # ------------------------------------------------------------------

    @cached_property
    def log10M_grid(self) -> np.ndarray:
        """Log10 mass grid corresponding to the internal k-grid."""
        return np.log10(self.M_grid)

    @cached_property
    def M_grid(self) -> np.ndarray:
        r_grid = 2.0 * np.pi / self.k_grid  # Mpc
        return 4.0 / 3.0 * np.pi * np.power(r_grid, 3) * self.cosmo.rho_m0  # Msun

    @cached_property
    def logM_grid(self) -> np.ndarray:
        return np.log10(self.M_grid)

    @cached_property
    def sigma0_grid(self) -> np.ndarray:
        """sigma(M, z=0) on the internal mass grid."""
        r_grid = 2.0 * np.pi / self.k_grid
        return np.array([
            self._sigma_R_from_Pk(self.k_grid, self.Pk0_grid, R=r)
            for r in r_grid
        ])
    
    @cached_property
    def s0_grid(self) -> np.ndarray:
        return np.power(self.sigma0_grid, 2)

    @cached_property
    def _sigma0_interp(self):
        """Interpolator for ln sigma(M,0) vs log10 M."""
        return interp1d(
            self.log10M_grid,
            np.log(self.sigma0_grid),
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
    
    @cached_property
    def _M_interp(self):
        """Interpolator for M(sigma0,0) vs sigma0."""
        return interp1d(
            np.log(self.sigma0_grid),
            self.log10M_grid,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    def sigma_M(self, M: np.ndarray | float, z: float = 0.0) -> np.ndarray:
        """
        Return sigma(M,z) using cached sigma(M,0) and the growth factor:

            sigma(M,z) = sigma(M,0) * D(z) / D(0)
        """
        M = np.asarray(M)
        logM = np.log10(M)

        Mmax = np.max(self.M_grid)
        mask = (M < Mmax)

        ln_sigma0_mask = self._sigma0_interp(logM[mask])
        sigma0 = np.exp(ln_sigma0_mask)

        # Enforce sigma(M > Mmax) = 1e-99
        sigma0_full = np.zeros_like(M)
        sigma0_full[mask] = sigma0

        Dz = self.cosmo.growth_factor(z)
        D0 = self.cosmo.growth_factor(0.0)
        return sigma0_full * (Dz / D0)
    
    def s_M(self, M: np.ndarray | float, z: float = 0.0) -> np.ndarray:
        return np.power(self.sigma_M(M, z), 2)
    

    # ------------------------------------------------------------------
    # d ln sigma / d log10 M  (cached)
    # ------------------------------------------------------------------

    @cached_property
    def dlnsigma_dlogM_grid(self) -> np.ndarray:
        """
        Cached derivative of ln sigma0(M) with respect to log10(M).
        Computed on the internal mass grid.
        """

        # ln sigma(M) on the grid
        ln_sigma = np.log(self.sigma0_grid)   # shape (N,)

        # derivative d ln sigma / d log10 M
        # Use numpy.gradient which handles uneven grids (y and x arrays)
        dlnsigma_dlogM = np.gradient(ln_sigma, self.logM_grid)

        return dlnsigma_dlogM

    @cached_property
    def _dlnsigma_dlogM_interp(self):
        """Interpolator for d ln sigma / d log10 M."""
        return interp1d(
            self.logM_grid,
            self.dlnsigma_dlogM_grid,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    def dlnsigma_dlogM(self, M: np.ndarray | float, z: float = 0.0) -> np.ndarray:
        """
        Public method: compute d ln sigma / d log10 M.

        Masks values outside interpolation range by setting them to 0.
        """
        M = np.asarray(M)
        logM = np.log10(M)

        Mmax = np.max(self.M_grid)
        mask = (M < Mmax)

        val = self._dlnsigma_dlogM_interp(logM[mask])

        # Enforce dlnsigma/dlogM(M > Mmax) = 0
        val_full = np.zeros_like(M)
        val_full[mask] = val

        return val_full