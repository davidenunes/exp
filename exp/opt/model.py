""" Surrogate Models for Bayesian Optimisation

"""

from scipy.linalg import cholesky, solve
import numpy as np
from collections import OrderedDict
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor


class GaussianProcess:
    def __init__(self, covfunc, optimize=False, usegrads=False, mprior=0):
        """
        Gaussian Process regressor class. Based on Rasmussen & Williams [1]_ algorithm 2.1.

        Parameters
        ----------
        covfunc: instance from a class of covfunc module
            Covariance function. An instance from a class in the `covfunc` module.
        optimize: bool:
            Whether to perform covariance function hyperparameter optimization.
        usegrads: bool
            Whether to use gradient information on hyperparameter optimization. Only used
            if `optimize=True`.

        Attributes
        ----------
        covfunc: object
            Internal covariance function.
        optimize: bool
            User chosen optimization configuration.
        usegrads: bool
            Gradient behavior
        mprior: float
            Explicit value for the mean function of the prior Gaussian Process.

        Notes
        -----
        [1] Rasmussen, C. E., & Williams, C. K. I. (2004). Gaussian processes for machine learning.
        International journal of neural systems (Vol. 14). http://doi.org/10.1142/S0129065704001899
        """
        self.covfunc = covfunc
        self.optimize = optimize
        self.usegrads = usegrads
        self.mprior = mprior
        self.x = []
        self.y = []

    def getcovparams(self):
        """
        Returns current covariance function hyperparameters

        Returns
        -------
        dict
            Dictionary containing covariance function hyperparameters
        """
        d = {}
        for param in self.covfunc.parameters:
            d[param] = self.covfunc.__dict__[param]
        return d

    def fit(self, x, y):
        """
        Fits a Gaussian Process regressor

        Parameters
        ----------
        x: np.ndarray, shape=(nsamples, nfeatures)
            Training instances to fit the GP.
        y: np.ndarray, shape=(nsamples,)
            Corresponding continuous target values to X.

        """
        self.x = x
        self.y = y
        self.nsamples = self.x.shape[0]
        if self.optimize:
            grads = None
            if self.usegrads:
                grads = self._grad
            self.optHyp(param_key=self.covfunc.parameters, param_bounds=self.covfunc.bounds, grads=grads)

        self.K = self.covfunc.K(self.x, self.x)
        self.L = cholesky(self.K).T
        self.alpha = solve(self.L.T, solve(self.L, y - self.mprior))
        self.logp = -.5 * np.dot(self.y, self.alpha) - np.sum(np.log(np.diag(self.L))) - self.nsamples / 2 * np.log(
            2 * np.pi)

    def param_grad(self, k_param):
        """
        Returns gradient over hyperparameters. It is recommended to use `self._grad` instead.

        Parameters
        ----------
        k_param: dict
            Dictionary with keys being hyperparameters and values their queried values.

        Returns
        -------
        np.ndarray
            Gradient corresponding to each hyperparameters. Order given by `k_param.keys()`
        """
        k_param_key = list(k_param.keys())
        covfunc = self.covfunc.__class__(**k_param)
        K = covfunc.K(self.x, self.x)
        L = cholesky(K).T
        alpha = solve(L.T, solve(L, self.y))
        inner = np.dot(np.atleast_2d(alpha).T, np.atleast_2d(alpha)) - np.linalg.inv(K)
        grads = []
        for param in k_param_key:
            gradK = covfunc.gradK(self.x, self.x, param=param)
            gradK = .5 * np.trace(np.dot(inner, gradK))
            grads.append(gradK)
        return np.array(grads)

    def _lmlik(self, param_vector, param_key):
        """
        Returns marginal negative log-likelihood for given covariance hyperparameters.

        Parameters
        ----------
        param_vector: list
            List of values corresponding to hyperparameters to query.
        param_key: list
            List of hyperparameter strings corresponding to `param_vector`.

        Returns
        -------
        float
            Negative log-marginal likelihood for chosen hyperparameters.

        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        self.covfunc = self.covfunc.__class__(**k_param)

        # This fixes recursion
        original_opt = self.optimize
        original_grad = self.usegrads
        self.optimize = False
        self.usegrads = False

        self.fit(self.x, self.y)

        self.optimize = original_opt
        self.usegrads = original_grad
        return (- self.logp)

    def _grad(self, param_vector, param_key):
        """
        Returns gradient for each hyperparameter, evaluated at a given point.

        Parameters
        ----------
        param_vector: list
            List of values corresponding to hyperparameters to query.
        param_key: list
            List of hyperparameter strings corresponding to `param_vector`.

        Returns
        -------
        np.ndarray
            Gradient for each evaluated hyperparameter.

        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        return - self.param_grad(k_param)

    def optHyp(self, param_key, param_bounds, grads=None, n_trials=5):
        """
        Optimizes the negative marginal log-likelihood for given hyperparameters and bounds.
        This is an empirical Bayes approach (or Type II maximum-likelihood).

        Parameters
        ----------
        param_key: list
            List of hyperparameters to optimize.
        param_bounds: list
            List containing tuples defining bounds for each hyperparameter to optimize over.

        """
        xs = [[1, 1, 1]]
        fs = [self._lmlik(xs[0], param_key)]
        for trial in range(n_trials):
            x0 = []
            for param, bound in zip(param_key, param_bounds):
                x0.append(np.random.uniform(bound[0], bound[1], 1)[0])
            if grads is None:
                res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=param_bounds)
            else:
                res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=param_bounds, jac=grads)
            xs.append(res.x)
            fs.append(res.fun)

        argmin = np.argmin(fs)
        opt_param = xs[argmin]
        k_param = OrderedDict()
        for k, x in zip(param_key, opt_param):
            k_param[k] = x
        self.covfunc = self.covfunc.__class__(**k_param)

    def predict(self, x_star, variance_only=False):
        """
        Returns mean and covariances for the posterior Gaussian Process.

        Parameters
        ----------
        x_star: np.ndarray, shape=((nsamples, nfeatures))
            Testing instances to predict.
        variance_only: bool
            Whether to return the standard deviation of the posterior process. Otherwise,
            it returns the whole covariance matrix of the posterior process.

        Returns
        -------
        np.ndarray
            Mean of the posterior process for testing instances.
        np.ndarray
            Covariance of the posterior process for testing instances.
        """
        x_star = np.atleast_2d(x_star)
        kstar = self.covfunc.K(self.x, x_star).T
        fmean = self.mprior + np.dot(kstar, self.alpha)
        v = solve(self.L, kstar.T)
        fcov = self.covfunc.K(x_star, x_star) - np.dot(v.T, v)
        if variance_only:
            fcov = np.diag(fcov)
        return fmean, fcov

    def update(self, x, y):
        """ Updates the internal model with new observations `x` and `y` instances.

        Args:

        x: np.ndarray, shape=((m, nfeatures))
            New training instances to update the model with.
        y: np.ndarray, shape=((m,))
            New training targets to update the model with.
        """
        y = np.concatenate((self.y, y), axis=0)
        x = np.concatenate((self.x, x), axis=0)
        self.fit(x, y)


class RandomForest:
    def __init__(self, **params):
        """
        Wrapper around sklearn's Random Forest implementation for pyGPGO.
        Random Forests can also be used for surrogate models in Bayesian Optimization.
        An estimate of 'posterior' variance can be obtained by using the `impurity`
        criterion value in each subtree.
        Parameters
        ----------
        params: tuple, optional
            Any parameters to pass to `RandomForestRegressor`. Defaults to sklearn's.
        """
        self.params = params

    def fit(self, x, y):
        """
        Fit a Random Forest model to data `X` and targets `y`.
        Parameters
        ----------
        x : array-like
            Input values.
        y: array-like
            Target values.
        """
        self.x = x
        self.y = y
        self.n = self.x.shape[0]
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(x, y)

    def predict(self, Xstar, return_std=True, eps=1e-6):
        """
        Predicts 'posterior' mean and variance for the RF model.
        Parameters
        ----------
        Xstar: array-like
            Input values.
        return_std: bool, optional
            Whether to return posterior variance estimates. Default is `True`.
        eps: float, optional
            Floating precision value for negative variance estimates. Default is `1e-6`
        Returns
        -------
        array-like:
            Posterior predicted mean.
        array-like:
            Posterior predicted std
        """
        Xstar = np.atleast_2d(Xstar)
        ymean = self.model.predict(Xstar)
        if return_std:
            std = np.zeros(len(Xstar))
            trees = self.model.estimators_

            for tree in trees:
                var_tree = tree.tree_.impurity[tree.apply(Xstar)]
                var_tree = np.clip(var_tree, eps, np.inf)
                mean_tree = tree.predict(Xstar)
                std += var_tree + mean_tree ** 2

            std /= len(trees)
            std -= ymean ** 2
            std = np.sqrt(np.clip(std, eps, np.inf))
            return ymean, std
        return ymean

    def update(self, xnew, ynew):
        """
        Updates the internal RF model with observations `xnew` and targets `ynew`.
        Parameters
        ----------
        xnew: array-like
            New observations.
        ynew: array-like
            New targets.
        """
        y = np.concatenate((self.y, ynew), axis=0)
        x = np.concatenate((self.x, xnew), axis=0)
        self.fit(x, y)
