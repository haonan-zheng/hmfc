#  Halo Mass Function Calculator (hmfc)

**hmfc** is a Python tool for calculating the **halo mass function (HMF)** across

- **All halo masses:** $10^{-6} - 10^{16}$ $\mathrm{M}_\odot$ 
- **All environments:** overdensity $\delta > -0.99$
- **All cosmic times:** $z = 0 - 30$

It provides a unified interface for cosmology, power spectrum, mass definition, mass conversion, halo mass function models, and measurements in mass and peak-height ($\nu$) space.

---

##  Features

###  Mass definitions

- $M_{200\mathrm{m}}$, $M_{200\mathrm{c}}$, and $M_{500\mathrm{c}}$
- Automatic conversion between mass definitions using Einasto profiles
- Infrastructure for custom mass definitions

###  Quantities

- $dn/d\log_{10}M$
- $dn/dM$
- $n(>M)$
- $f(\nu)$

###  Cosmology

- Built-in cosmologies: **Planck13**, **Planck15**, **Planck18**
- User-defined cosmological parameters
- Growth factor, critical/mean density

###  Power spectrum

- Analytic **BBKS** transfer function
- User-provided $(k,$ $P(k))$ tables
- Automatic $\sigma_8$ normalization
- Cached internal grids with logarithmic spacing

###  HMF models

Included models (will include more models in future versions): 

- **Press–Schechter**
- **Sheth–Tormen**
- **Reed et al., 2007**
- **Zheng et al., 2025**

Extendable model API for custom HMF fits.

---

##  Quick Start

```python
import numpy as np
from src import hmf, Cosmology

cosmo = Cosmology.planck13()
m = np.logspace(-6, 15, 22)

hmf_predict = hmf(m, z=0, cosmo=cosmo, quantity="dn_dlog10M", powerspec="bbks", mass_definition="M200c")
print(hmf_predict)
