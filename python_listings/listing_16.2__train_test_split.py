import numpy as np

# This is mostly the same as the listing 12.1 but has additional assertions.

def train_test_split(X, y, split=0.6, seed=None):
    """Splits a dataset given by X and y into training and testing data
    
    This function uses numpy's function `shuffle` and preserves the order of elements in X by
    randomly picking elements and "moving" those to the test data set. If you also want the
    two data sets themselves to be shuffled you have to do this afterwards.
    
    :param X: feature data (a 1D numpy array)
    :param y: target data (a 1D numpy array)
    :param split: percentage of training data (e.g., 0.6: 60% of the DS is used as training data)
    :param seed: either None or an integer used as seed from the random number generator
    :returns: X_train, y_train, X_test, y_test
    """
    assert (X.ndim == 1) and (y.ndim == 1)

    # total number of records and number of points for training
    n_records = y.size   
    n_training_records = int(np.round(split * n_records))

    # get random indices for the data used for learning
    mask = np.empty(n_records, dtype=bool)
    mask[:n_training_records] = True
    mask[n_training_records:] = False
    rng = np.random.default_rng(seed)
    rng.shuffle(mask)

    # get training data
    X_train = X[mask]
    y_train = y[mask]

    # ... and what's left is used for testing
    X_test = X[~mask]
    y_test = y[~mask]
    
    return X_train, y_train, X_test, y_test