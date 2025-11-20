"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, DotProduct
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

def get_initial_safe_point(domain=DOMAIN):
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    # assume v(x) = 2 for initial safe points
    c_val = np.full_like(x_domain, 2.0)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    return x_valid[0].reshape(1, -1)

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # data storage
        self.X = np.zeros((0, DOMAIN.shape[0]))
        self.y_f = np.zeros((0,))
        self.y_v = np.zeros((0,))

        # constants
        self.prior_mean_v = 4.0
        self.kappa = SAFETY_THRESHOLD
        self.domain = DOMAIN

        # noise levels
        self.sigma_f = 0.15
        self.sigma_v = 1e-4
        self.alpha_f = self.sigma_f ** 2
        self.alpha_v = self.sigma_v ** 2

        # GPs and kernels
        self.gp_f = None
        self.gp_v = None
        self.kern_f = ConstantKernel(0.5, (1e-3, 1e3)) * Matern(length_scale=1.0,
                                                                length_scale_bounds=(1e-2, 1e2),
                                                                nu=2.5)
        self.kern_v = DotProduct(sigma_0=1.0) + Matern(length_scale=1.0,
                                                       length_scale_bounds=(1e-2, 1e2),
                                                       nu=2.5)

        # GP fitting options
        self.n_restarts_optimizer = 5
        self.normalize_y = True

        # acquisition/safety hyperparams
        self.xi = 0.01  # for EI
        self.beta_v = 3.0  # UCB multiplier for conservative safety
        self.N_unsafe_max = 3
        self.N_unsafe_used = 0

    def _fit_gps(self):
        if self.X.shape[0] == 0:
            return

        # fit objective GP
        self.gp_f = GaussianProcessRegressor(
            kernel=self.kern_f,
            alpha=self.alpha_f,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer
        )
        self.gp_f.fit(self.X, self.y_f)

        # fit constraint GP
        self.gp_v = GaussianProcessRegressor(
            kernel=self.kern_v,
            alpha=self.alpha_v,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer
        )
        self.gp_v.fit(self.X, self.y_v)

    def _find_safest_point(self):
        """Return the safest point according to constraint GP"""
        grid = np.linspace(self.domain[0,0], self.domain[0,1], 400).reshape(-1,1)
        mu_v, sigma_v = self.gp_v.predict(grid, return_std=True)
        mu_v = mu_v + self.prior_mean_v
        best_idx = np.argmin(mu_v)
        return grid[best_idx].reshape(1, -1)

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

        if self.X.shape[0] == 0:
            return get_initial_safe_point()

        self._fit_gps()
        x_opt = self.optimize_acquisition_function()
        x_opt = np.atleast_2d(x_opt).reshape(1, -1)

        mu_v, sigma_v = self.gp_v.predict(x_opt, return_std=True)
        mu_v = mu_v + self.prior_mean_v
        ucb_v = mu_v + self.beta_v * sigma_v

        if ucb_v > self.kappa:
            return self._find_safest_point()

        return x_opt
    
    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x).item()

        f_values = []
        x_values = []

        for _ in range(20):
            x0 = DOMAIN[:,0] + (DOMAIN[:,1]-DOMAIN[:,0])*np.random.rand(DOMAIN.shape[0])
            res = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN, approx_grad=True)
            x_values.append(np.clip(res[0], *DOMAIN[0]))
            f_values.append(-res[1])

        best_idx = np.argmax(f_values)
        return np.array([[x_values[best_idx]]])

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
        x = np.atleast_2d(x)

        mu_f, sigma_f = self.gp_f.predict(x, return_std=True)
        sigma_f = np.maximum(sigma_f.reshape(-1,1), 1e-9)

        f_best = np.max(self.y_f)
        Z = (mu_f - f_best - self.xi)/sigma_f
        ei = (mu_f - f_best - self.xi) * norm.cdf(Z) + sigma_f * norm.pdf(Z)

        mu_v, sigma_v = self.gp_v.predict(x, return_std=True)
        mu_v = mu_v.reshape(-1,1) + self.prior_mean_v
        sigma_v = np.maximum(sigma_v.reshape(-1,1), 1e-9)
        Z_safe = (self.kappa - mu_v)/sigma_v
        p_safe = norm.cdf(Z_safe)

        return (ei * p_safe).reshape(-1,1)

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
        # ensure x is 2D with shape (1, D)
        x = np.atleast_2d(x).reshape(1, -1)

        self.X = np.vstack([self.X, x])
        self.y_f = np.append(self.y_f, f)
        self.y_v = np.append(self.y_v, v - self.prior_mean_v)
        if v > self.kappa:
            self.N_unsafe_used += 1


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        if self.X.shape[0] == 0:
            return np.array([[self.domain[0,0]]])

        self._fit_gps()
        grid = np.linspace(self.domain[0,0], self.domain[0,1], 400).reshape(-1,1)
        mu_f, _ = self.gp_f.predict(grid, return_std=True)
        mu_v, _ = self.gp_v.predict(grid, return_std=True)
        mu_v = mu_v.reshape(-1,1) + self.prior_mean_v

        safe_mask = (mu_v < self.kappa).flatten()
        if np.sum(safe_mask) == 0:
            safe_obs_mask = (self.y_v + self.prior_mean_v < self.kappa)
            if np.sum(safe_obs_mask) > 0:
                best_idx = np.argmax(self.y_f[safe_obs_mask])
                return self.X[safe_obs_mask][best_idx].reshape(1,-1)
            return np.array([[self.domain[0,0]]])

        safe_grid = grid[safe_mask]
        safe_mu_f = mu_f[safe_mask]
        best_idx = np.argmax(safe_mu_f)
        return safe_grid[best_idx].reshape(1,-1)

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
