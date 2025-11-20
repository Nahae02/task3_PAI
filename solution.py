"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        # Data storage
        self.X = np.empty((0,1))  # shape (n_samples, n_features)
        self.y_f = np.empty((0,)) 
        self.y_v = np.empty((0,))

        # Gaussian Process for f
        obj_kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
            length_scale=1.0, 
            length_scale_bounds=(0.1, 5.0),
            nu=2.5
        )

        # Noise std from task: sigma_f = 0.15  -> variance = 0.15**2
        self.obj_gp = GaussianProcessRegressor(
            kernel=obj_kernel,
            alpha=0.15**2,
            normalize_y=True,
            n_restarts_optimizer=3
        )

        # Gaussian Process for v
        constr_kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
            length_scale=1.0, 
            length_scale_bounds=(0.1, 5.0),
            nu=2.5
        )

        # Noise std from task: sigma_v = 0.0001 -> variance = 1e-8
        self.constr_gp = GaussianProcessRegressor(
            kernel=constr_kernel,
            alpha=0.0001**2,
            normalize_y=True,
            n_restarts_optimizer=3
        )

        # UCB parameters
        self.beta_obj = 2.0  # exploration-exploitation trade-off parameter
        self.beta_constr = 3.0  # safety trade-off parameter

        self.n_grid_points = 500  # number of grid points for safe set estimation

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        #raise NotImplementedError
        # If we have no data yet, return the center of the domain
        if self.X.shape[0] == 0:
            x0 = 0.5 * (DOMAIN[0, 0] + DOMAIN[0, 1])
            return np.atleast_2d([x0])
        
        x_next = self.optimize_acquisition_function()
        return np.atleast_2d(x_next)

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        if self.X.shape[0] == 0:
            return 0.0
        
        x = np.atleast_2d(x).reshape(-1, 1)

        mu_v, std_v = self.constr_gp.predict(x, return_std=True)
        ucb_v = mu_v + self.beta_constr * std_v

        if ucb_v[0] >= SAFETY_THRESHOLD:
            return -1e6

        mu_f, std_f = self.obj_gp.predict(x, return_std=True)
        ucb_f = mu_f[0] + self.beta_obj * std_f[0]

        return ucb_f
        # TODO: Implement the acquisition function you want to optimize.
        #raise NotImplementedError

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        #raise NotImplementedError

        # Make x a scalar float within DOMAIN
        x_scalar = float(np.squeeze(x))
        x_scalar = float(np.clip(x_scalar, DOMAIN[0, 0], DOMAIN[0, 1]))

        # Append to data
        self.X = np.vstack((self.X, [[x_scalar]]))
        self.y_f = np.append(self.y_f, f)
        self.y_v = np.append(self.y_v, v)

        # Refit GPs
        if self.X.shape[0] >= 1:
            self.obj_gp.fit(self.X, self.y_f)
            self.constr_gp.fit(self.X, self.y_v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        #raise NotImplementedError

        if self.X.shape[0] == 0:
            x0 = 0.5 * (DOMAIN[0, 0] + DOMAIN[0, 1])
            return np.atleast_2d([x0])
        
        #use GP posterior mean to find best safe point
        x_grid = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], self.n_grid_points)[:, None]

        mu_v, std_v = self.constr_gp.predict(x_grid, return_std=True)
        ucb_v = mu_v + self.beta_constr * std_v
        safe_mask = ucb_v < SAFETY_THRESHOLD
        safe_points = x_grid[safe_mask]

        if safe_points.shape[0] == 0:
            safe_obs_mask = self.y_v < SAFETY_THRESHOLD
            if np.any(safe_obs_mask):
                x_best_safe = self.X[safe_obs_mask][np.argmax(self.y_f[safe_obs_mask])]
            else:
                x_best_safe = 0.5 * (DOMAIN[0, 0] + DOMAIN[0, 1])
            return np.atleast_2d(x_best_safe)
        
        # maximize posterior mean of f over safe points
        mu_f_safe, _ = self.obj_gp.predict(safe_points, return_std=True)
        idx_best = np.argmax(mu_f_safe)
        x_opt = safe_points[idx_best]

        return np.atleast_2d(x_opt)

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
