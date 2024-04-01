import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x = np.linspace(-8, 8, 200)
mu, sigma = -2, 1
rv = stats.norm(loc=mu, scale=sigma)

plt.plot(x, rv.pdf(x), label='PDF')
plt.plot(x, rv.cdf(x), label='CDF')
plt.legend()
plt.show()