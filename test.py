import numpy as np
from src import hmf, Cosmology
import pdb


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
    Cosmological parameters. If None, uses default cosmology (Planck14).
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
"""

m = np.logspace(-6, 15, 22)  # Mass array from 10^-6 to 10^15 Msun
z = 0
cosmo = Cosmology.planck13()

# examples
#hmf_prds = hmf(m, z, cosmo=cosmo, quantity="dn_dlog10M", powerspec='bbks')
#hmf_prds = hmf(m, z, cosmo=cosmo, quantity="n(>M)", powerspec='bbks')

#hmf_prds = hmf(m, z, cosmo=cosmo, quantity="f_nu")
#hmf_prds = hmf(m, z, cosmo=cosmo, quantity="f_nu", overdensity=-0.5, mass_region=5e15)

#hmf_prds = hmf(m, z, cosmo=cosmo, quantity="dn_dlog10M", mass_definition="M200m")
#hmf_prds = hmf(m, z, cosmo=cosmo, quantity="dn_dlog10M", mass_definition="M200c")

pdb.set_trace()
