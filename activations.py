import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z, get_derivative=False):
    if get_derivative:
        return sigmoid(z)*(1-sigmoid(z))
    else:
        return 1.0/(1+np.exp(-z))


def tanh(z, get_derivative=False):
    if get_derivative:
        return 1 - np.power(np.tanh(z), 2)
    else:
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def linear(z, get_derivative=False):
    if get_derivative:
        return np.ones(z.shape)
    else:
        return z


def relu(z, get_derivative=False):
    if get_derivative:
        return np.where(z > 0, 1, 0)
    else:
        return np.maximum(0, z)


def leaky_relu(z, get_derivative=False, alpha=0.1):
    if get_derivative:
        return np.where(z > 0, 1, alpha)
    else:
        return np.maximum(alpha * z, z)


def elu(z, get_derivative=False, alpha=0.1):
    if get_derivative:
        return np.where(z > 0, 1, alpha * np.exp(z))
    else:
        return np.where(z >= 0, z, alpha * (np.exp(z) - 1))



def hinton(matrix, max_weight=None, ax=None):
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()



