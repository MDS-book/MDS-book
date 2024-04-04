import numpy as np


""" === These are the used parameters and variables: ==============
X: input data,                  shape: (n_samples,      n_features)
W1: weights of hidden layer,    shape: (n_features + 1, n_units)
W2: weights of output node,     shape: (n_units + 1, 1)
theta1: threshold hidden layer, shape: (n_units) -> a 1d array
theta1: threshold output node,  shape: (1)       -> a 1d array
"""

n_samples = X.shape[0]   # number of samples/data records

# prepend a "1"-column for bias of hidden layer neurons
X1 = np.concatenate((np.ones((n_samples, 1)), X), axis=1)

# copy the row vector theta 1 n_samples-times
THETA1 = np.repeat(theta1, n_samples, axis=0)
#
# evaluate the neurons of the hidden lyer
T1 = np.where(X1 @ W1 >= THETA1, 1, -1)

# prepend a "1"-column for bias of output nodes
T1 = np.concatenate((np.ones((n_samples, 1)), T1), axis=1)

# compute results of ouput nodes
y = np.where(T1 @ W2 >= theta2, True, False)