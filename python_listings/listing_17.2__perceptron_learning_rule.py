import numpy as np


def predict(x, weights, bias, threshold=0.):
    a = bias + np.dot(x, weights)
    y = np.where(a > threshold, 1, 0)
    return y


def train_1_epoch(X_train, y_train, weights, bias, threshold, learning_rate):
    n_samples, n_features = X_train.shape
    y_pred = np.zeros_like(y_train)

    for j in range(n_samples):
        # feed forward direction
        y_pred[j] = predict(X_train[j], weights, bias, threshold)

        # backwards direction ("perceptron rule")
        weights += learning_rate * (y_train[j] - y_pred[j]) * X_train[j]
        bias += learning_rate * (y_train[j] - y_pred[j])

    return weights, bias


def train(X_train, y_train, n_epochs, learning_rate, threshold=0.):
    n_features = X_train.shape[1]  # the features are the columns of X_train
    weights = np.zeros(n_features)
    bias = 0.

    for i in range(n_epochs):
        weights, bias = train_1_epoch(X_train, y_train, weights, bias,
                                      threshold, learning_rate)
        y_pred = predict(X_train, weights, bias, threshold)

        if np.sum(np.abs(y_pred - y_train)) == 0:  # check if error is zero
             break
    
    return weights, bias