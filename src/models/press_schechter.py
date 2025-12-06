import numpy as np
from .base import HMFModel

class PressSchechter(HMFModel):
    def f_nu(self, nu, **kwargs):
        # without cloud-in-cloud correction
        return np.sqrt(2.0 / np.pi) * nu * np.exp(-nu * nu / 2.0)