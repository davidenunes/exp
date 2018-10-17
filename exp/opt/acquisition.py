""" Acquisition functions

The basic idea being that acquisition functions can be configured with default parameters in the constructor
and calling eval will evaluate the current acquisition function.

The difference between calling eval and calling the function e.g.::


    ei = ExpectedImprovement(eps=1e-4)
    ei(...)
    # vs
    ei.eval(...)

is that eval will capture any unused arguments in the kwargs if the function does not use
the said arguments. This is used by the BOPT class so that evaluating an acquisition function has the same
signature every time

refactored from pygpgo:

https://github.com/hawk31/pyGPGO/blob/master/pyGPGO/acquisition.py

"""
import numpy as np
from scipy.stats import norm, t


class Acquisition:
    def __init__(self, **params):
        self.params = params

    def eval(self, tau, mean, std):
        return self(tau, mean, std, **self.params)


class ExpectedImprovement(Acquisition):
    """ Expected Improvement acquisition function.

        Args:

        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
    """

    def __call__(self, tau, mean, std, eps=1e-08, **kwargs):
        z = (mean - tau - eps) / (std + eps)
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)[0]

    """
     fmfn 
     mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
      
     GPyOPT   
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu
        
        def get_quantiles(acquisition_par, fmin, m, s):
            Quantiles of the Gaussian distribution useful to determine the acquisition function values
            :param acquisition_par: parameter of the acquisition function
            :param fmin: current minimum.
            :param m: vector of means.
            :param s: vector of standard deviations.
            '''
            if isinstance(s, np.ndarray):
                s[s<1e-10] = 1e-10
            elif s< 1e-10:
                s = 1e-10
            u = (fmin - m - acquisition_par)/s
            phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
            Phi = 0.5 * erfc(-u / np.sqrt(2))
            return (phi, Phi, u)


    """


class LowerConfidenceBound(Acquisition):
    """Upper-confidence bound acquisition function.

        Args:

        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        beta: float
            Hyperparameter controlling exploitation/exploration ratio.

        Returns
        -------
        float
            Upper confidence bound.
        """

    def __call__(self, mean, std, exploration_weight=1.5, **kwargs):
        return -mean + exploration_weight * std
