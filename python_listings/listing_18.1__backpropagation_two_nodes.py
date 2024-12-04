import numpy as np


def sigmoidal_function(x):
    return 1 / (1 + np.exp(-x))


def forward_pass(W, B, X1, X2):
    (w1, w2, w3, w4, w5, w6),  (b1, b2, b3) = W, B
    
    a1 = b1 + X1 * w1 + X2 * w2
    a2 = b2 + X1 * w3 + X2 * w4
    y1 = sigmoidal_function(a1)
    y2 = sigmoidal_function(a2)
    a3 = y1 * w5 + y2 * w6 + b3
    return a1, a2, y1, y2, a3


def predict(W, B, X1, X2):
    _, _, _, _, Y_pred = forward_pass(W, B, X1, X2)
    return Y_pred


def backward_pass(W, B, X1, X2, Y):
    w1, w2, w3, w4, w5, w6 = W
    a1, a2, y1, y2, a3 = forward_pass(W, B, X1, X2)

    dJdb1 = -(Y - a3) * w5 * y1 * (1 - y1)
    dJdw1 = dJdb1 * X1
    dJdw2 = dJdb1 * X2
    dJdb2 = -(Y - a3) * w6 * y2 * (1 - y2)
    dJdw3 = dJdb2 * X1
    dJdw4 = dJdb2 * X2
    dJdb3 = -(Y - a3)
    dJdw5 = -(Y - a3) * y1
    dJdw6 = -(Y - a3) * y2

    dJdW, dJdB = (dJdw1, dJdw2, dJdw3, dJdw4, dJdw5, dJdw6), (dJdb1, dJdb2, dJdb3)
    return dJdW, dJdB


def update_weights(W, B, dJdW, dJdB, learning_rate):
    (w1, w2, w3, w4, w5, w6), (b1, b2, b3) = W, B
    (dJdw1, dJdw2, dJdw3, dJdw4, dJdw5, dJdw6), (dJdb1, dJdb2, dJdb3) = dJdW, dJdB
        
    w1 -= learning_rate * np.mean(dJdw1)
    w2 -= learning_rate * np.mean(dJdw2)
    w3 -= learning_rate * np.mean(dJdw3)
    w4 -= learning_rate * np.mean(dJdw4)
    w5 -= learning_rate * np.mean(dJdw5)
    w6 -= learning_rate * np.mean(dJdw6)
    b1 -= learning_rate * np.mean(dJdb1)
    b2 -= learning_rate * np.mean(dJdb2)
    b3 -= learning_rate * np.mean(dJdb3)

    W_new, B_new = (w1, w2, w3, w4, w5, w6), (b1, b2, b3)
    return W_new, B_new


def mean_loss(Y, Y_pred):
    return 0.5 * np.mean((Y - Y_pred) ** 2)


def train(W, B, X1, X2, Y, epochs, learning_rate):
    mean_losses = []
    
    for epoch in range(epochs):
        Y_pred = predict(W, B, X1, X2)
        mean_losses.append(mean_loss(Y, Y_pred))
        
        dJdW, dJdB = backward_pass(W, B, X1, X2, Y)
        W, B = update_weights(W, B, dJdW, dJdB, learning_rate)

    # also add the updated last value
    Y_pred = predict(W, B, X1, X2)
    mean_losses.append(mean_loss(Y, Y_pred))
    
    return W, B, mean_losses