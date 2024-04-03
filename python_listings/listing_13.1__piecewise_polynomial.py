from numpy import linspace
from numpy.polynomial import Polynomial


def piecewise_fit(x_data, y_data, knots_x, degree, n_interpol_points=10):
    """Divides data into intervals and fits a polynomial to each of them

    :param x_data, y_data: data points (numpy arrays)
    :param knots_x: list of the horizontal positions of the support points
                    (= (n_knots-1) intervals)
    :param degree: polynomial degree
    :param n_interpol_points: number of equally spaced points in each interval
    :returns: lists with the approximated data for each interval
    """
    piecewise_data = []
    for x0, x1 in zip(knots_x[:-1], knots_x[1:]):
        idx = (x_data >= x0) * (x_data < x1)
        poly = Polynomial.fit(x=x_data[idx], y=y_data[idx], deg=degree)
        interp_x = linspace(x0, x1, n_interpol_points)
        piecewise_data.append((interp_x, poly(interp_x)))

    return piecewise_data
