import numpy as np
from abc import ABC


def sigmoidal_function(x):
    return 1 / (1 + np.exp(-x))


def sigmoidal_derivative(x):
    return sigmoidal_function(x)*(1.0 - sigmoidal_function(x))


def tanh_function(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu_function(x):
    return np.maximum(0, x)


def relu_derivative(x):
    der = np.ones_like(x)
    der[relu_function(x) < 0] = 0
    return der


def leakyrelu_function(x, alpha):
    return np.maximum(x*alpha, x)


def leakyrelu_derivative(x, alpha):
    der = np.ones_like(x)
    der[leakyrelu_function(x, alpha) < 0] = alpha
    return der


# This is the same base class as the one developed in section 18.5.1
class NNLayer(ABC):
    """Abstract base class which serves as a "template" for fully connected
    layers and for activation layers.

    Each derived class has the input x and output y both of which are numpy arrays.
    Also, each derived class should implement the two methods below (with the
    same signature)
    """
    def __init__(self):
        self.x = np.empty()
        self.y = np.empty()

    def feed_forward(self, x):
        """Perform a forward step: compute output for given input
        
        This method additionally is supposed to store the input and output vectors
        
        :param x: numpy array of all inputs for one record
        :returns: numpy array of all output for one record
        """
        raise NotImplementedError("Any layer needs to implement the feed_forward method.")

    def backward_propagation(self, dJdy, learning_rate):
        """Backpropagate the error sensitivity to the input
        
        :returns: numpy array of dJ/dy of the previous layer
        """
        raise NotImplementedError("Any layer needs to implement the backward_propagation method.")