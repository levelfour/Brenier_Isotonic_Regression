import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, softmax
from sklearn.isotonic import IsotonicRegression

from brenier import brenier_isotonic_regression
from brenier import get_transport_map, get_laguerre_assignment
from brenier import uniform_cube_sampling


def visualize_2d(x, y_true, y_pred, low=-3., high=3.):
    # create a grid for plotting the true function surface
    x_range = np.linspace(low, high, 30)
    y_range = np.linspace(low, high, 30)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_grid = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
    truefunc_grid = y_true(Z_grid)
    truefunc_grid_1 = truefunc_grid[:, 0].reshape(X_grid.shape)
    truefunc_grid_2 = truefunc_grid[:, 1].reshape(X_grid.shape)
    
    fig = plt.figure(figsize=(12, 6))
    
    # plot 1: first coordinate
    ax1 = fig.add_subplot(121, projection='3d')
    
    # plot fitted points (y_i first coordinate)
    scatter1 = ax1.scatter(x[:, 0], x[:, 1], y_pred[:, 0],
                           c='red', s=50, alpha=0.7, label='Fitted y₁')
    
    # plot true function surface (first coordinate)
    surf1 = ax1.plot_wireframe(X_grid, Y_grid, truefunc_grid_1,
                               alpha=0.3, cmap='viridis', label='True y₁')
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_zlabel('First coordinate value')
    ax1.set_title('First Coordinate: Fitted vs True Function')
    ax1.legend()
    
    # plot 2: second coordinate
    ax2 = fig.add_subplot(122, projection='3d')
    
    # plot fitted points (y_i second coordinate)
    scatter2 = ax2.scatter(x[:, 0], x[:, 1], y_pred[:, 1],
                           c='red', s=50, alpha=0.7, label='Fitted y₂')
    
    # Plot true function surface (second coordinate)
    surf2 = ax2.plot_wireframe(X_grid, Y_grid, truefunc_grid_2,
                               alpha=0.3, cmap='plasma', label='True y₂')
    
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_zlabel('Second coordinate value')
    ax2.set_title('Second Coordinate: Fitted vs True Function')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def visualize_1d(x, y_true, y_obs, y_pred, u_opt, P_opt):
    y_ir = IsotonicRegression().fit_transform(x, y_obs)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3))
    x_smooth = np.linspace(-3., 3., 100)
    xlim = (-3, 3)
    ylim = (-0.01, 1.01)
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

    y_true = y_true(x_smooth)
    ax1.scatter(x, y_obs, color='blue', s=30, marker='x')
    ax1.plot(x_smooth, y_true, 'b--', lw=3, alpha=0.5)
    ax1.set_xlabel(r'$z$', fontsize=14)
    ax1.set_ylabel(r'$y$', fontsize=14)
    ax1.set_title('Observations', fontsize=14)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_yticks(yticks)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(x, y_ir, color='gray', s=30)
    ax2.set_xlabel(r'$x$', fontsize=14)
    ax2.set_title('Isotonic Regression', fontsize=14)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_yticks(yticks, labels=[])
    ax2.grid(True, alpha=0.3)

    ax3.scatter(x, y_pred, color='green', s=30)

    def laguerre_interpolation(query):
        _, _, potential = get_transport_map(x, u_opt, potential=True)
        laguerre_assignment = get_laguerre_assignment(query, u_opt, potential)
        return u_opt[laguerre_assignment]

    ax3.plot(x_smooth, laguerre_interpolation(x_smooth), 'g-', lw=3, alpha=0.5)
    ax3.set_xlabel(r'$x$', fontsize=14)
    ax3.set_title('Brenier IR', fontsize=14)
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_yticks(yticks, labels=[])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bins', default=None, type=int, help='number of bins')
    parser.add_argument('-n', default=50, type=int, help='number of samples')
    parser.add_argument('-d', '--dim', default=1, type=int, help='dimension')
    parser.add_argument('-s', '--seed', default=-1, type=int, help='random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

    parser.add_argument('--reduced', action='store_true', help='reduced visualization for paper')

    args = parser.parse_args()

    if args.seed != -1:
        np.random.seed(args.seed)

    n = args.n
    dim = args.dim
    
    x = uniform_cube_sampling(n, dim, low=-3., high=3.)

    # true function of y
    if dim > 1:
        truefunc = lambda x: softmax(x, axis=1)
    elif dim == 1:
        truefunc = lambda x: expit(x)

    # observation of y
    noise = np.random.normal(0, 0.1, (n, dim))
    y = truefunc(x) + noise
    
    n_bins = args.bins
    u_opt, P_opt, fvals = brenier_isotonic_regression(x, y, args, n_bins, maxiter=100)
    fitted_y = n * (P_opt @ u_opt)
    
    print(f"MSE between true and Brenier: {np.sum((y - fitted_y)**2)/n:.6f}")
    
    if dim == 2:
        visualize_2d(x, truefunc, fitted_y)
    elif dim == 1:
        visualize_1d(x[:, 0], truefunc, y[:, 0], fitted_y[:, 0], u_opt[:, 0], P_opt)
