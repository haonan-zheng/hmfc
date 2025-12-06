import numpy as np
import pdb
from .base import HMFModel

class Zheng25(HMFModel):
    def __init__(self, cosmo, powerspec, **kwargs):
        super().__init__(cosmo, powerspec, **kwargs)
        self.A_st = kwargs.get("A_st", 0.3222)
        self.a_st = kwargs.get("a_st", 0.707)
        self.p_st = kwargs.get("p_st", 0.3)
        self.c_R07 = kwargs.get("c_R07", 1.08) 
        
    def f_nu(self, nu, **kwargs):
        z = kwargs.get("z", 0.0)
        self.c_Zheng25 = np.min([1.03 + 0.05 * z, 1.08])
        self.theta_Zheng25 = np.exp(-(1 + z) / 10.)
        self.alpha_Zheng25 = 0.275 - 0.304 * self.theta_Zheng25
        self.beta_Zheng25 = 1.75 - 1.45 * self.theta_Zheng25
        self.Gamma_Zheng25 = 1 - self.alpha_Zheng25 + self.alpha_Zheng25 * np.tanh(5 * (np.log(nu) - self.beta_Zheng25))

        anu2 = self.a_st * nu * nu
        G1 = np.exp(-np.power(np.log(nu / self.delta_c) - 0.4, 2) / (2 * 0.6 * 0.6))

        f = self.A_st * np.sqrt(2.0 * self.a_st / np.pi) * \
            (1 + np.power(anu2, -self.p_st) + 0.2 * G1) * \
            self.Gamma_Zheng25 * nu * np.exp(-self.c_Zheng25 * anu2 / 2.0)

        return f