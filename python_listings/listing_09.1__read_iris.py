import numpy as np

def import_iris_dataset():
    """Read the file `iris.csv` and prepare data.

    Returns the feature matrix X and target vector y.
    """
    data = np.loadtxt('iris.csv', dtype='str', delimiter=',')
    X = np.array(data[:, :4], dtype=float)
    y = np.array(data[:, 4], dtype=str)

    return X, y