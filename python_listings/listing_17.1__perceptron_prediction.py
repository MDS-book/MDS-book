import numpy as np


def predict(x, weights, bias, threshold):
    # Evaluate the transfer function: we're using numpy's scalar product `dot`
    a = bias + np.dot(x, weights)

    # Activation function: check if the perceptron fires
    # (For 1 data record a is a scalar and we could simply use
    # `y = int(a >= threshold)`. For multiple data records, a is a 1D array
    # --> use numpy' `where(condition, x, y)` function which returns x for
    # those elements where the condition holds and for all others it returns y.)
    y = np.where(a >= threshold, 1, 0)

    return y