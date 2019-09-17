from scipy.stats import entropy
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_prediction_uncertainty(net, points):
    points_torch = torch.from_numpy(points).float()
    # Given toy dataset
    mean, alpha, precision = net(points_torch)

    N = np.int_(np.sqrt(points.shape[0]))
    max_prod = np.max(mean.detach().numpy(), 1).reshape((N, N))
    x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), N)
    y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), N)

    plt.figure()
    plt.contourf(x, y, max_prod, levels=20, cmap='Blues')
    plt.suptitle('Max class probability (higher more certain)')


def plot_contour(points, z, title, levels=10, ax=None, fig=None):
    N = np.int_(np.sqrt(points.shape[0]))
    z = z.reshape((N, N))
    x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), N)
    y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), N)

    if ax is None or fig is None:
        plt.figure()
        cs = plt.contourf(x, y, z, levels=levels, cmap='Blues')
        cbar = plt.colorbar(cs)
        plt.suptitle(title)
    else:
        cs = ax.contourf(x, y, z, levels=levels, cmap='Blues')
        fig.colorbar(cs, ax=ax)
        ax.title.set_text(title)
        # return cs


def get_entropy(mean):

    entropys = []
    for row in mean:
        ent = entropy(row.detach())
        entropys.append(ent)
    return np.array(entropys)


def to_torch(points):
    points_torch = torch.from_numpy(points).float()
    return points_torch


def plot_entropys(ent, points):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=ent)
    plt.suptitle('Entropy distribution (lower more certain)')


def calc_dirichlet_differential_entropy(alphas, epsilon=1e-8):
    # Calculate Expected Entropy of categorical distribution under dirichlet Prior.
    # Higher means more uncertain
    alphas = np.asarray(alphas, dtype=np.float64) + epsilon
    diff_entropy = np.asarray([dirichlet(alpha).entropy() for alpha in alphas])
    return diff_entropy
