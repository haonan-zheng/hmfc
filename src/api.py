from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import pdb

from .cosmology import Cosmology
from .mass_definition import MassDefinition, MassConcentration
from .models import get_model
from .power_spectrum import PowerSpectrum


# -------------------------------------------------------------------
# Cosmology → cache key helper
# -------------------------------------------------------------------

def _cosmo_params(cosmo: Cosmology) -> tuple[float, float, float, float, float]:
    """
    Build a hashable tuple of cosmology parameters for caching.

    Adjust this if your Cosmology class uses a different parameter set.
    """
    return (cosmo.h, cosmo.Om0, cosmo.Ode0, cosmo.ns, cosmo.sigma8)


# -------------------------------------------------------------------
# Cached PowerSpectrum builder for "generated from cosmology" cases
# -------------------------------------------------------------------

@lru_cache(maxsize=32)
def _get_power_spectrum(
    cosmo_params_with_label: tuple[float, float, float, float, float, str]
) -> PowerSpectrum:
    """
    Return a cached PowerSpectrum instance for a given cosmology + label.

    `label` can encode different generation schemes, e.g. 'linear',
    'nonlinear', 'eh98', etc. Internally you can branch on it in
    PowerSpectrum if needed.
    """
    h, Om0, Ode0, ns, sigma8, label = cosmo_params_with_label
    cosmo = Cosmology(h=h, Om0=Om0, Ode0=Ode0, ns=ns, sigma8=sigma8)
    return PowerSpectrum(cosmo=cosmo, scheme=label)


# -------------------------------------------------------------------
# General PowerSpectrum builder for the public API
# -------------------------------------------------------------------

def _build_powerspectrum(
    cosmo: Cosmology,
    powerspec: str | PowerSpectrum | tuple[np.ndarray, np.ndarray] | None
) -> PowerSpectrum:
    """
    Normalize and/or construct a PowerSpectrum object given cosmology
    and a flexible `powerspec` argument.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology used both to generate or normalize the power spectrum.
    powerspec : {None, str, PowerSpectrum, (k, Pk)}
        - None:      load from './powerspec' file by default
        - str:       named scheme (e.g. 'linear', 'nonlinear', 'eh98')
        - PowerSpectrum: use as-is
        - (k, Pk):   user-provided table, normalized to `cosmo`

    Returns
    -------
    PowerSpectrum
        A ready-to-use PowerSpectrum instance.
    """

    # load from './powerspec' by default
    if powerspec is None:
        powerspec = './powerspec'

    # Case 1: already a PowerSpectrum instance
    if isinstance(powerspec, PowerSpectrum):
        return powerspec

    # Case 2: None or string label → generate from cosmology with caching
    if isinstance(powerspec, str):
        if powerspec.lower() in {"bbks"}:
            label = powerspec or "cosmo"  # default scheme name
            cosmo_params = _cosmo_params(cosmo) + (label,)
            return PowerSpectrum.from_formula(
                cosmo=cosmo, 
                model=powerspec, 
                normalize_to_cosmo=True
            )
        else:
            # Loading from file powerspec with log10(k / h Mpc^-1) and log10(P(k) * k^3)
            powerspec_k_table, powerspec_P_table = np.loadtxt(powerspec, unpack=True)
            powerspec_k_table = powerspec_k_table + np.log10(0.6777)
            powerspec_k_table = np.power(10, powerspec_k_table)
            powerspec_P_table = np.power(10, powerspec_P_table) / np.power(powerspec_k_table, 3)

            powerspec = (powerspec_k_table, powerspec_P_table)

    # Case 3: table: (k, Pk)
    if isinstance(powerspec, tuple) and len(powerspec) == 2:
        k, pk = powerspec
        k = np.asarray(k)
        pk = np.asarray(pk)
        # We assume PowerSpectrum has a classmethod from_table(...)
        # that optionally normalizes to sigma8 of `cosmo`.
        return PowerSpectrum.from_table(
            cosmo=cosmo,
            k=k,
            pk=pk,
            normalize_to_cosmo=True
        )

    raise TypeError(
        "powerspec must be one of: str, PowerSpectrum, or (k, Pk) tuple"
    )


# -------------------------------------------------------------------
# Public halo mass function API
# -------------------------------------------------------------------

def hmf(
    m: Sequence[float] | np.ndarray,
    z: float,
    model: str = "Zheng25",
    cosmo: Cosmology | None = None,
    powerspec: str | PowerSpectrum | tuple[np.ndarray, np.ndarray] | None = None,
    mass_definition: str = "M200m",
    overdensity: float = 0.0,
    mass_region: float = 1e99,
    quantity: str = "dn_dlog10M",
    **model_kwargs
):
    """
    Parameters of hmf
    ----------
    m : array_like
        Halo mass in units of Msun, in the chosen `mass_definition`.
    z : float
        Redshift.
    model : str, optional
        The halo mass function model to use. Default is "Zheng25".
    cosmo : Cosmology, optional
        Cosmological parameters. If None, uses default cosmology (Planck13).
    powerspec : {None, str, PowerSpectrum, (k, Pk)}, optional
        - None:        build default linear P(k) from `cosmo` (cached)
        - str:         named scheme for P(k) generation, e.g. 'linear'
        - PowerSpectrum: use this object directly
        - (k, Pk):     user-provided power spectrum table, need to be ormalized
                        to match `cosmo` (e.g. sigma_8) via PowerSpectrum.from_table
        # default is loading from './powerspec'
    mass_definition : str, optional
        Mass definition of the input masses, e.g. 'M200m' or 'M200c'.
        Default is 'M200m'.
    overdensity : float, optional
        Overdensity of the region. Default is 0.0 (mean density).
    mass_region : float, optional
        Mass of the region in which the HMF is computed.
        Default is 1e99 Msun (effectively infinite).
    quantity : {'dn_dlog10M', 'dn_dM', 'n(>M)', 'f_nu', 'class'}, optional
        Quantity to compute. 
        'dn_dlog10M' is the differential mass function per d log10 M; 
        'dn_dM' is the differential mass function per d M;
        'n(>M)' is the cumulative number density above mass M;
        'f_nu' is the mass (fraction) function f(nu), as defined in Reed+07 eq. 3. 
        'class' is returning the whole class
        Default is 'dn_dlog10M'.

    Returns
    -------
    hmf_values : ndarray
        Computed halo mass function values corresponding to input masses.
    """

    # 1. Cosmology
    if cosmo is None:
        cosmo = Cosmology.planck13()

    # 2. Power spectrum (build/normalize from cosmology + powerspec)
    # Loading from file powerspec with log10(k / h Mpc^-1) and log10(P(k) * k^3)
    powerspec_obj = _build_powerspectrum(cosmo, powerspec=powerspec)

    # 3. Mass definition: parse and convert input masses to working definition
    # c_obj: mass - concentration relation for conversions
    # m_input: input masses as ndarray
    # m_work: masses converted to M200m
    # jacobian: dlog10M_work / dlog10M_input
    c_obj = MassConcentration(cosmo=cosmo, powerspec=powerspec_obj)
    md_input = MassDefinition.from_string(mass_definition, profile='einasto')
    m_input = np.asarray(m)
    m_work, jacobian = md_input.convert_mass(m_input, z, cosmo, c_obj, target="M200m", return_jacobian=True)

    # 4. Get the HMF model
    ModelClass = get_model(model)
    hmf_model = ModelClass(
        cosmo=cosmo,
        powerspec=powerspec_obj,
        **model_kwargs
    )

    # 5. Compute the HMF
    hmf_values = hmf_model.hmf(
        m_work,
        z,
        overdensity=overdensity,
        mass_region=mass_region,
        quantity=quantity,
        m_input=m_input,
        jacobian=jacobian
    )
    return hmf_values

