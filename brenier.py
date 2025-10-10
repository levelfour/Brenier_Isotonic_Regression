import numpy as np
import ot
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import label_binarize, LabelEncoder


def uniform_simplex_sampling(n, d):
    """
    Uniformly sample n points over d-simplex.
    cf. https://math.stackexchange.com/a/502598
    """
    r = -np.log(np.random.uniform(size=(n, d)))
    r /= r.sum(axis=1, keepdims=True)
    return r

def uniform_cube_sampling(n, d, low=-1., high=1.):
    """
    Uniformly sample n points over d-dimensional hypercube.
    """
    r = np.random.uniform(low=low, high=high, size=(n, d))
    return r

def dist_matrix(x, y):
    if x.ndim == 1 and y.ndim == 1:
        return ot.dist(x.reshape(-1, 1), y.reshape(-1, 1), metric='sqeuclidean', p=2)
    else:
        return ot.dist(x, y, metric='sqeuclidean', p=2)

def get_transport_map(x_source, x_target, potential=False):
    """
    Compute the discrete optimal transport matrix between two point clouds.
    """
    cost = dist_matrix(x_source, x_target)
    marginal_source = ot.utils.unif(x_source.shape[0])
    marginal_target = ot.utils.unif(x_target.shape[0])
    P, result = ot.emd(marginal_source, marginal_target, cost, log=True)
    potential_source, potential_target = result["u"], result["v"]

    if potential:
        return P, potential_source, potential_target
    else:
        return P

def get_laguerre_assignment(query, u, potential):
    """
    Compute Laguerre cell assignment for query points.

    Parameters:
    -----------
    query: np.array of shape (m, dim)
        Points to assign to Laguerre cells
    u: np.array of shape (n, dim)
        Intermediate target support points obtained by Brenier isotonic regression
    potential: np.array of shape (n,)
        Laguerre weights

    Returns:
    --------
    assignments : np.array of shape (m,)
        Index of assigned target point for each query point
    """
    distances = dist_matrix(query, u)
    weighted_distances = distances - potential[np.newaxis, :]

    # Assign to closest target (minimum weighted distance)
    assignments = np.argmin(weighted_distances, axis=1)
    return assignments

def _get_u_constraints(n, dim):
    # equality constraints (for dim > 1): each row sums to 1
    cons = []
    if dim > 1:
        for i in range(n):
            cons.append({
                'type': 'eq',
                'fun': lambda u, row=i: np.sum(u.reshape(n, dim)[row, :]) - 1
            })

    # inequality constraints
    bds = [(0, 1)] * (n * dim)

    return cons, bds

def _brenier_isotonic_regression_sqp(x, y, n_bins=None, maxiter=50, tol=1e-6, verbose=False):
    """
    Parameters
    ----------
    x (numpy.ndarray): input covariates (n, dim)
    y (numpy.ndarray): vector labels (n, dim)
    """
    n = x.shape[0]
    dim = y.shape[1]
    n_bins = n if n_bins == None else n_bins

    cons, bds = _get_u_constraints(n_bins, dim)

    def objective(u_flat):
        u = u_flat.reshape(n_bins, dim)
        P = get_transport_map(x, u)
        fitted = n * (P @ u)
        return np.sum((y - fitted) ** 2) / n

    # initial guess
    if dim > 1:
        u0 = uniform_simplex_sampling(n_bins, dim)
    elif dim == 1:
        u0 = uniform_cube_sampling(n_bins, dim, low=0., high=1.)
    u0 = u0.flatten()

    result = minimize(objective,
                      u0,
                      bounds=bds, constraints=cons,
                      method='SLSQP',
                      options={'maxiter': maxiter, 'ftol': tol, 'disp': verbose})
    u = result.x.reshape(n_bins, dim)
    P = get_transport_map(x, u)

    return u, P, []

def brenier_isotonic_regression(x, y, args, n_bins=None, maxiter=50, tol=1e-6):
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape')

    if args.verbose:
        print(f"\n-------------------- [begin optimization] --------------------")

    u, P, fvals = _brenier_isotonic_regression_sqp(x, y, n_bins,
                                                   maxiter=maxiter, tol=tol,
                                                   verbose=args.verbose)

    if args.verbose:
        print(f"-------------------- [finish optimization] --------------------\n")

    return u, P, fvals


class BrenierIsotonicRegression(BaseEstimator, RegressorMixin):
    def __init__(self, n_bins=None, maxiter=100, tol=1e-6, verbose=False):
        """
        Parameters
        ----------
        n_bins: int
            How many Laguerre cells. If None, you'll have n_sample cells.
        maxiter: int
            How many iterations before terminating the outer optimization.
        tol: float
            Budget for the outer optimization.
        """
        self.n_bins = n_bins
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose
        self.fitted = False

    def fit(self, scores, y, *args, **kwargs):
        """
        Parameters
        ----------
        scores: array-like, shape = [n_samples, n_classes]
            Data.
        y : array-like, shape = [n_samples, ]
            Labels.

        Returns
        -------
        self
        """

        if "X_val" in kwargs and "y_val" in kwargs:
            X_val = kwargs["X_val"]
            y_val = kwargs["y_val"]
            scores = np.concatenate([scores, X_val])
            y = np.concatenate([y, y_val])

        self.classes = np.unique(y)
        target = label_binarize(y, classes=self.classes)

        u, P, _ = _brenier_isotonic_regression_sqp(scores, target,
                                                   self.n_bins,
                                                   maxiter=self.maxiter,
                                                   tol=self.tol,
                                                   verbose=self.verbose)

        self.predictions = u / u.sum(axis=1, keepdims=True)
        self.P = P

        _, _, potential = get_transport_map(scores, u, potential=True)
        self.potential = potential

        self.fitted = True

        return self

    def predict_proba(self, scores, *args, **kwargs):
        laguerre_assignment = get_laguerre_assignment(
            scores, self.predictions, self.potential
        )
        return self.predictions[laguerre_assignment]
