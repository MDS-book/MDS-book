from numpy import ndarray, empty, argsort, argmax, bincount
from scipy.spatial import distance_matrix


def predict(X_train: ndarray, y_train: ndarray, X: ndarray, k=1) -> ndarray:
    """ 
    Returns the k-nearest neighbors for all instances in `X` and given training data,
    X and X_train are 2D numpy arrays, y_train is a 1D array.
    """
    # row i of dist contains the distances for vector X[i] to every
    # other vector of the training dataset
    dist = distance_matrix(X, X_train)

    # Go through all rows of dist, i.e. the distance of vector X[i] to all training 
    # points: get the `k` closest points and check which class has the most entries
    y = empty(X.shape[0])
    for i, d in enumerate(dist):

        # get the indices "that would sort the 1D array `d`" from nearest to farest 
        sorting_indices = argsort(d)
        kn_y = y_train[sorting_indices][:k].ravel() 

        # `bincount` returns the number of occurrences of each value of the array --
        # therefore, the position of the maximum IS the class with the maximum count
        count = bincount(kn_y)
        y[i] = argmax(count)

    return y