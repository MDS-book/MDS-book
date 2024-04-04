import numpy as np

def RBF(x, mu, alpha=1):
    """Evaluates the Gaussian RBF at positions x. 
    
    The location and shape is given by mu and alpha.
    """
    return np.exp(-(mu - x)**2 / alpha)


def feature_matrix(X, mu_rbf, alpha):
    """Assemble the feature/design matrix from the RBFs.
    
    mu_rbf is a list of locations of the RBFs, 
    alpha is 'spread' of the Gaussians.
    """
    IX = np.ones((X.size, len(mu_rbf) + 1))
    for j, mu in enumerate(mu_rbf, start=1):
        IX[:,j] = RBF(X, mu, alpha)
    return IX