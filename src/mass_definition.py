from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.special import erfc, gammainc
from scipy.optimize import fsolve

from .cosmology import Cosmology
from .power_spectrum import PowerSpectrum

RefType = Literal["crit", "mean"]
ProfileType = Literal["einasto"]
ConcModelType = Literal["ludlow"]


# -------------------------------------------------------------
# Concentration models
# -------------------------------------------------------------

def _alpha(M: float | np.ndarray) -> np.ndarray:
    # Wang+20 
    # M in Msun
    alpha = 0.16 + 0.0238 * np.power(M / 1.14e14, 1./3.)
    return alpha


def _c_model_Ludlow(
    M: np.ndarray,
    z: float,
    cosmo: Cosmology,
    powerspec: PowerSpectrum,
    f_Ludlow: float = 0.02,
    C_Ludlow: float = 650.0,
    alpha: float | np.ndarray = 0.18
) -> np.ndarray:
    """
    Ludlow+16 analytic concentration model c(M,z).

    Parameters
    ----------
    M : array_like
        Mass in Msun, interpreted as M200c.
    z : float
        Redshift.
    cosmo : Cosmology
        Cosmology object providing rho_m(z), rho_crit(z), H(z) and growth_factor(z).
    powerspec : PowerSpectrum
        Power spectrum object providing s_M(M).
    f_Ludlow, C_Ludlow : float
        Ludlow+14 parameters.
    alpha : float or array_like
        Einasto shape parameter; can be mass-dependent if desired.

    Returns
    -------
    c_Ludlow : ndarray
        Concentration c200c(M, z).
    """
    M = np.atleast_1d(M).astype(float)
    alpha_arr = np.atleast_1d(alpha)
    if alpha_arr.size == 1 and M.size > 1:
        alpha_arr = np.full_like(M, alpha_arr[0])

    # convenience: growth factor from Cosmology
    def growth_factor(z: float) -> float:
        return cosmo.growth_factor(z)

    # Einasto mass and density ratios
    def M_enclose_ratio(r, rs, c, alpha):
        x = r / rs
        return (gammainc(3.0 / alpha, 2.0 / alpha * np.power(x, alpha)) / 
                gammainc(3.0 / alpha, 2.0 / alpha * np.power(c, alpha)))

    def rho_enclose_ratio(r, rs, c, alpha):
        x = r / rs
        return M_enclose_ratio(r, rs, c, alpha) / np.power(x / c, 3)

    def eq7_left(c, alpha):
        # M_enclosed fraction at r=rs (x=1) relative to r=c*rs
        return M_enclose_ratio(1.0, 1.0, c, alpha=alpha)

    def eq7_right(M, z_m2):
        delta_c = 1.68647

        D_zm2 = growth_factor(z_m2)
        D_z   = growth_factor(z)
        dd = delta_c * (1. / D_zm2 - 1. / D_z)

        s1 = powerspec.s_M(M * f_Ludlow)
        s2 = powerspec.s_M(M)
        ds = s1 - s2
        return erfc(dd / np.sqrt(2.0 * ds))

    def H(z):
        # Assume Cosmology has H(z) in km/s/Mpc; if not, use your own formula
        return cosmo.H(z)

    def eq6_left(c, alpha):
        # ρ̄(<rs) / ρ_crit(z) * 200?  
        return rho_enclose_ratio(1.0, 1.0, c, alpha) * 200.0

    def eq6_right(z_m2):
        return C_Ludlow * (H(z_m2) / H(z)) ** 2

    # system of two equations for (c, z_m2) at each M
    def equations_M(p, M_val, alpha_val):
        c, z_m2 = p
        return (eq7_left(c, alpha_val) - eq7_right(M_val, z_m2),
                eq6_left(c, alpha_val) - eq6_right(z_m2), )

    c_Ludlow_list = []

    for Mi, alpha_i in zip(M, alpha_arr):
        # initial guess (you can tune this)
        c_guess = 50.0
        z_m2_guess = z
        c_sol, z_m2_sol = fsolve(equations_M, (c_guess, z_m2_guess), args=(Mi, alpha_i))
        c_Ludlow_list.append(c_sol)

    return np.array(c_Ludlow_list) if M.size > 1 else np.array(c_Ludlow_list[0])



@dataclass
class MassConcentration:
    """
    Precomputed mass–concentration table c(M,z) for a given
    cosmology, redshift and concentration model.

    Parameters
    ----------
    cosmo : Cosmology
    powerspec : PowerSpectrum
        Only strictly needed for 'ludlow', but harmless to keep for all.
    z : float
    model : {'powerlaw', 'ludlow'}
    logM_min, logM_max : float
        Bounds of log10(M/Msun) for the internal grid.
    nM : int
        Number of mass sampling points.
    """
    cosmo: Cosmology
    powerspec: PowerSpectrum
    z: float = 0.0
    model: ConcModelType = "ludlow"
    logM_min: float = -10.0
    logM_max: float = 20.0
    nM: int = 1201

    @cached_property
    def logM_grid(self) -> np.ndarray:
        return np.linspace(self.logM_min, self.logM_max, self.nM)

    @cached_property
    def M_grid(self) -> np.ndarray:
        return 10.0 ** self.logM_grid

    @cached_property
    def c_grid(self) -> np.ndarray:
        """
        c(M,z) evaluated on M_grid.
        """
        if self.model == "ludlow":
            # Zheng+25 default Ludlow+16 parameters for better fit to Wang+20
            return _c_model_Ludlow(self.M_grid, self.z, self.cosmo, self.powerspec, \
                                   f_Ludlow = 0.06, C_Ludlow = 1200, alpha=_alpha(self.M_grid))
        else:
            raise ValueError(f"Unknown concentration model '{self.model}'")

    @cached_property
    def _c_interp(self):
        return interp1d(
            self.logM_grid,
            self.c_grid,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    def c_M(self, M: float | np.ndarray) -> np.ndarray:
        """
        Return c(M,z) by interpolation from the internal table.
        """
        M = np.asarray(M, dtype=float)
        logM = np.log10(M)
        return self._c_interp(logM)


# -------------------------------------------------------------
# Cached conversion grid builder
# -------------------------------------------------------------

def _f_einasto(x: np.ndarray | float, alpha: float) -> np.ndarray:
    """
    Einasto cumulative mass factor:

        f(x; alpha) \prop gamma(3/alpha, 2 x^alpha / alpha)

    M(<r) \prop f(r/r_s; alpha). 
    Any overall normalization cancels in ratios like f(c y) / f(c).
    """
    x = np.asarray(x, dtype=float)
    a = float(alpha)
    # lower incomplete gamma
    return gammainc(3.0 / a, 2.0 / a * x**a)


def _solve_y_einasto(
    c200c: float,
    alpha: float,
    Delta_from: float,
    rho_from: float,
    Delta_to: float,
    rho_to: float,
    y_min: float = 1e-2,
    y_max: float = 1e2
) -> float:
    """
    For an Einasto halo defined by {M_from, Delta_from, rho_from, c_from = c200c},
    find y = r_to / r_from such that:

        M(< r_to) / M_from = f(c y; alpha) / f(c; alpha)
                           = (Delta_to * rho_to / (Delta_from * rho_from)) * y^3
    """
    c = float(c200c)
    a = float(alpha)
    A = Delta_to * rho_to / (Delta_from * rho_from)
    f_c = _f_einasto(c, a)

    def func(y: float) -> float:
        return _f_einasto(c * y, a) / f_c - A * y**3

    return brentq(func, y_min, y_max)


def _build_m200c_conversion_grid(
    cosmo: Cosmology,
    z: float,
    conc: MassConcentration,
    Delta_target: float,
    ref_target: RefType,
    profile: ProfileType = "einasto"
) -> dict[str, np.ndarray]:
    """
    Build a conversion table at redshift z for:

        M200c  ->  M_target(Delta_target, ref_target),

    using Einasto + concentration table.

    Parameters
    ----------
    cosmo : Cosmology
    z : float
    conc : Concentration
        Mass - concentration table at (cosmo,z). Its M_grid is interpreted
        as M200c, and c_grid as c200c(M200c,z).
    Delta_target : float
        Overdensity of target definition (e.g. 200, 500, 340, ...).
    ref_target : {'crit', 'mean'}
        Critical or mean density for the target definition.
    profile : {'einasto', 'nfw'}
        Only 'einasto' is implemented internally; 'nfw' kept for API.

    Returns
    -------
    dict with keys:
        'logM200c_grid'  : log10(M200c)
        'logMtarget_grid': log10(M_target)
    """
    if profile != "einasto":
        raise NotImplementedError("Currently only Einasto profile is implemented.")

    # Pivot M200c & concentrations
    logM200c = conc.logM_grid          # log10(M200c)
    M200c = conc.M_grid                # Msun (interpreted as M200c)
    c200c = conc.c_grid                # c200c(M200c,z)

    # Einasto alpha(M)
    alpha_arr = _alpha(M200c)          # you already have this

    # densities at z
    rho_crit = cosmo.rho_crit(z)       # Msun / Mpc^3
    rho_mean = cosmo.rho_m(z)          # Msun / Mpc^3

    # from-definition: 200c
    Delta_200c = 200.0
    rho_from = rho_crit

    # target density
    rho_target = rho_crit if ref_target == "crit" else rho_mean

    Mtarget = np.empty_like(M200c)

    for i, (M_c, c_c, a_c) in enumerate(zip(M200c, c200c, alpha_arr)):
        f_c = _f_einasto(c_c, a_c)

        y_t = _solve_y_einasto(
            c200c=c_c,
            alpha=a_c,
            Delta_from=Delta_200c,
            rho_from=rho_from,
            Delta_to=Delta_target,
            rho_to=rho_target
        )
        f_cy = _f_einasto(c_c * y_t, a_c)
        Mtarget[i] = M_c * f_cy / f_c

    logMtarget = np.log10(Mtarget)

    return {"logM200c_grid": logM200c,
            "logMtarget_grid": logMtarget}


@dataclass
class MassDefinition:
    """
    Spherical overdensity mass definition:

        name      : e.g. 'M200c', 'M200m', 'M500c'
        Delta     : overdensity value
        reference : 'crit' or 'mean'
        profile   : shape model; currently only 'einasto' implemented.
    """
    name: str
    Delta: float
    reference: RefType
    profile: ProfileType = "einasto"

    @classmethod
    def from_string(
        cls,
        name: str,
        profile: ProfileType = "einasto"
    ) -> "MassDefinition":
        s = name.strip().lower()

        if s == "m200c":
            return cls("M200c", 200.0, "crit", profile=profile)
        elif s == "m200m":
            return cls("M200m", 200.0, "mean", profile=profile)
        elif s == "m500c":
            return cls("M500c", 500.0, "crit", profile=profile)
        # add more here as needed
        else:
            raise ValueError(
                f"Unknown mass definition '{name}'. "
                "Extend MassDefinition.from_string() for more options."
            )
        
    def convert_mass(
        self,
        M: float | np.ndarray,
        z: float,
        cosmo: Cosmology,
        conc: MassConcentration,
        target: str = "M200m",
        return_interp: bool = False,
        return_jacobian: bool = False
    ):
        """
        Convert:

            M (self.definition) → M (target.definition)

        using Einasto + concentration table + M200c as pivot (implicitly).

        Parameters
        ----------
        M : float or array_like
            Input mass in 'self' definition.
        z : float
        cosmo : Cosmology
        conc : MassConcentration
            c(M200c, z) table.
        target : str, default 'M200m'
            Target mass definition.
        return_interp1d : bool, default False
            If True, also return the interp1d objects used for
            the conversion (in the log10 space). 
        return_jacobian : bool, default False
            If True, also return d log10(M_target) / d log10(M_self).

        Returns
        -------
        M_target : ndarray
        jac      : ndarray (only if return_jacobian=True)
        """
        M = np.asarray(M, dtype=float)

        if self.profile != "einasto":
            raise NotImplementedError("Only Einasto profile is implemented.")

        target_def = MassDefinition.from_string(target, profile=self.profile)

        # identity
        if self.name.lower() == target_def.name.lower():
            if return_jacobian:
                if not return_interp:
                    return M, np.ones_like(M)
                else:
                    return M, np.ones_like(M), lambda x: x, lambda x: np.ones_like(x)
            else:
                if not return_interp:
                    return M
                else:
                    return M, lambda x: x

        # --- Build grid for self definition: M_self(M200c) ---
        grid_self = _build_m200c_conversion_grid(
            cosmo=cosmo,
            z=z,
            conc=conc,
            Delta_target=self.Delta,
            ref_target=self.reference,
            profile=self.profile
        )
        logMself_grid = grid_self["logMtarget_grid"]  # M_target here is M_self

        # --- Build grid for target definition: M_target(M200c) ---
        grid_tgt = _build_m200c_conversion_grid(
            cosmo=cosmo,
            z=z,
            conc=conc,
            Delta_target=target_def.Delta,
            ref_target=target_def.reference,
            profile=self.profile
        )
        logMtarget_grid = grid_tgt["logMtarget_grid"]

        # Now treat logMtarget_grid as a function of logMself_grid
        interp_self_to_target = interp1d(
            logMself_grid,
            logMtarget_grid,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        interp_self_to_target_linspace = lambda M_vals: np.power(10, interp_self_to_target(np.log10(M_vals)))
        M_target = interp_self_to_target_linspace(M)

        if not return_jacobian:
            if not return_interp:
                return M_target
            else:
                return M_target, interp_self_to_target_linspace

        # Jacobian d logM_target / d logM_self on the grid:
        dlogMtgt_dlogMself_grid = np.gradient(logMtarget_grid, logMself_grid)

        interp_jac = interp1d(
            logMself_grid,
            dlogMtgt_dlogMself_grid,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

        interp_jac_linspace = lambda M_vals: interp_jac(np.log10(M_vals))
        jac = interp_jac_linspace(M)

        if not return_interp:
            return M_target, jac
        else:
            return M_target, jac, interp_self_to_target_linspace, interp_jac_linspace
        