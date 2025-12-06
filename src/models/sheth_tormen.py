import numpy as np
from .base import HMFModel

class ShethTormen(HMFModel):
    def __init__(self, cosmo, powerspec, **kwargs):
        super().__init__(cosmo, powerspec, **kwargs)
        self.A_st = kwargs.get("A_st", 0.3222)
        self.a_st = kwargs.get("a_st", 0.707)
        self.p_st = kwargs.get("p_st", 0.3)
        

    def f_nu(self, nu, **kwargs):
        anu2 = self.a_st * nu * nu   
        return self.A_st * np.sqrt(2.0 * self.a_st / np.pi) * \
               (1 + np.power(anu2, -self.p_st)) * nu * np.exp(-anu2 / 2.0)
