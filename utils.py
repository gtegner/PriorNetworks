import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def accuracy(mean, y_train):
    acc = torch.sum(torch.argmax(mean, 1) == y_train.squeeze()
                    ).numpy() / float(len(mean))

    print("Accuracy {0:.2f}%".format(acc * 100))
    # return acc


def generate_toy_dataset(N, sigma, use_torch=False):
    mu1 = np.array([0, 0])*3
    mu2 = np.array([2, 0])*3
    mu3 = np.array([1, np.sqrt(3)])*3

    def cov(_alpha): return np.array([[1, 0], [0, 1]]) * _alpha

    alpha = sigma
    alpha1 = cov(alpha)
    alpha2 = cov(alpha)
    alpha3 = cov(alpha)

    def gen_normal(mu, alpha):
        return np.random.multivariate_normal(
            mu, alpha, size=N)

    x1 = gen_normal(mu1, alpha1)
    x2 = gen_normal(mu2, alpha2)
    x3 = gen_normal(mu3, alpha3)

    y1 = np.zeros((N, 1))
    y2 = np.ones((N, 1))
    y3 = np.ones((N, 1))*2

    x1 = np.concatenate((x1, y1), 1)
    x2 = np.concatenate((x2, y2), 1)
    x3 = np.concatenate((x3, y3), 1)

    X = np.vstack((x1, x2, x3))
    np.random.shuffle(X)

    X_ood = generate_ood(X.shape[0], X[:, 0:2], 5)

    if use_torch:
        return torch.from_numpy(X).float()
    return X, X_ood


def generate_points(x_train, y_train, N=100, r=10):
    x_range = (x_train[:, 0].min()-r, x_train[:, 0].max()+r)
    y_range = (x_train[:, 1].min()-r, x_train[:, 1].max()+r)

    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    coord = np.array([[(i, j) for i in x] for j in y]).reshape(-1, 2)
    return coord


def generate_ood(N, x_train, dist):
    r = dist
    x_range = (x_train[:, 0].min()-r, x_train[:, 0].max()+r)
    v = np.random.uniform(x_range[0], x_range[1], size=(N, 2))
    return v


def plot_x(x, ax):
    if ax is None:
        return plt.scatter(x[:, 0], x[:, 1])
    return ax.scatter(x[:, 0], x[:, 1])


def plot_data(*args, ax=None):
    plt.figure()
    for arg in args:
        plot_x(arg, ax)


def plot_train(train, ytrain, ax):
    x1_ind = np.where(ytrain == 0)[0]
    x1 = train[x1_ind]

    x2_ind = np.where(ytrain == 1)[0]
    x2 = train[x2_ind]

    x3_ind = np.where(ytrain == 2)[0]
    x3 = train[x3_ind]

    plot_data(x1, x2, x3, ax=ax)
