from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize


class BayesianOptimization:
    """ Bayesian Optimization class.

            Args:
                surrogate_model: surrogate model instance (e.g. GaussianProcess)
                acquisition: acquisition function
                target_func: target function to maximize over parameters specified by `parameter_dict`.
                parameter_dict: dictionary specifying parameter, their type, bounds, and distributions.

            Attributes:
                parameter_key: list
                    Parameters to consider in optimization
                parameter_type: list
                    Parameter types.
                parameter_range: list
                    Parameter bounds during optimization
                history: list
                    Target values evaluated along the procedure.
                tau:
                    Best value found so far
    """

    def __init__(self, model, acquisition, param_space):

        self.model = model
        self.acquisition = acquisition
        self.param_space = param_space

        self.history = []

        # observations
        self.current_min = None
        self.x = None
        self.y = None

    def init(self, x: dict, y):
        """ initial setup is required before starting the process

        fits the surrogate model base on the initial data and
        records data about the first observations

        Note:
            which parameters to evaluate first should be decided outside the global
            bayesian optimiser (this can be done by sampling from a ParamSpace for instance)

        Args:
            x: list of dictionary with the parameters that we are optimizing
            y: list of loss values for each parameter configuration in x

        Returns:


        """
        self.x = np.array([list(xi.values()) for xi in x]),
        self.y = np.array(list(y))

        self.model.fit(self.x, self.y)

        self.current_min = np.min(self.y)
        self.history.append(self.current_min)

    def _eval_acquisition(self, x):
        """Evaluates the acquisition function on a point.

        Args:

        x_point: np.ndarray Point to evaluate the acquisition function on.

        Returns:
            (float) acquisition function value for x

        """
        """
        f_acqu = self._compute_acq(x)
        cost_x, _ = self.cost_withGradients(x)
        return -(f_acqu * self.space.indicator_constraints(x)) / cost_x
        
        
        def indicator_constraints(self,x):
        
            #Returns array of ones and zeros indicating if x is within the constraints
        
        x = np.atleast_2d(x)
        I_x = np.ones((x.shape[0],1))
        if self.constraints is not None:
            for d in self.constraints:
                try:
                    exec('constraint = lambda x:' + d['constraint'], globals())
                    ind_x = (constraint(x)<0)*1
                    I_x *= ind_x.reshape(x.shape[0],1)
                except:
                    print('Fail to compile the constraint: ' + str(d))
                    raise
        return I_x
        
        from GpyOpt
        the authors include a weighing by cost, the cost of evaluating the target function
        I don't want that because we're using GPUs which are pretty fast anyway, and we will 
        evaluate only on a fixed number of epochs
        """

        mean, variance = self.model.predict(x, variance_only=True)
        variance = np.clip(variance, 1e-10, np.inf)
        # We can take the square root because variance is just a diagonal of the matrix of variances
        std = np.sqrt(variance)
        return -self.acquisition.eval(tau=self.current_min, mean=mean, std=std)

    def _optimize_acquisition(self, method='L-BFGS-B', n_start=100):
        """Optimizes the acquisition function using a multistart approach.

        Returns the minimum value for the acquisition function according to the surrogate model (gaussian process)
        of the target function.

        Args:
            method: str. optimization method to be used.
                    Default 'L-BFGS-B'.
                    Any `scipy.optimize` method that admits bounds and gradients.


            n_start: int. Number of starting points for the optimization procedure. Default is 100.

        """
        start_points_dict = [self._sample_params() for i in range(n_start)]
        start_points_arr = np.array([list(s.values()) for s in start_points_dict])
        x_best = np.empty((n_start, len(self.parameter_key)))
        f_best = np.empty((n_start,))

        for index, start_point in enumerate(start_points_arr):
            res = minimize(self._eval_acquisition, x0=start_point, method=method,
                           bounds=self.param_space.bounds())
            x_best[index], f_best[index] = res.x, np.atleast_1d(res.fun)[0]

        self.best = x_best[np.argmin(f_best)]

    def update_surrogate(self):
        """ Updates the surrogate model with the next acquired point and its evaluation.

        updates self.history with tau: the best found model according to the model
        """
        kw = {param: self.best[i] for i, param in enumerate(self.parameter_key)}
        f_new = self.target_func(**kw)
        self.model.update(np.atleast_2d(self.best), np.atleast_1d(f_new))
        self.tau = np.max(self.model.y)
        self.history.append(self.tau)

    def getResult(self):
        """
        Prints best result in the Bayesian Optimization procedure.

        Returns
        -------
        OrderedDict
            Point yielding best evaluation in the procedure.
        float
            Best function evaluation.

        """
        argtau = np.argmax(self.model.y)
        opt_x = self.model.X[argtau]
        res_d = OrderedDict()
        for i, key in enumerate(self.parameter_key):
            res_d[key] = opt_x[i]
        return res_d, self.tau

    def run(self, init_evals=3):
        """Runs one step of the Bayesian Optimisation Procedure

        Args:
            init_evals (int): Initial function evaluations before fitting a GP. Default is 3.

        """
        if self.tau is None:
            self.init_evals = init_evals
            self._firstRun(self.init_evals)

        self._optimize_acquisition()
        self.update_surrogate()
