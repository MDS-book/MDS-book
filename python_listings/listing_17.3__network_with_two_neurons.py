import numpy as np


b1, w1, w2, theta1 = 0.5, -2.0, 2.0, -1.0  # TLU 1
b2, w3, w4, theta2 = 0.0, 1.5, -2.1, -1.0  # TLU 2
b3, w5, w6, theta3 = 0.0, -1.8, -2.2, -2.5  # output node

# the 4 points for the XOR problem: (0, 0), (1, 0), (0, 1), (1, 1)
x1 = np.array([0, 1, 0, 1])
x2 = np.array([0, 0, 1, 1])

# TLU 1
a1 = w1 * x1 + w2 * x2 + b1          # weighted sum
y1 = np.where(a1 >= theta1, +1, -1)  # evaluate activation functions

# TLU 2
a2 = w3 * x1 + w4 * x2 + b2           # weighted sum
y2 = np.where(a2 >= theta2, +1, -1)   # evaluate activation functions

# Output node
a3 = w5 * y1 + w6 * y2 + b3
y = np.where(a3 >= theta3, False, True)
print(f'x1 = {x1}, x2 = {x2}')
print(f' y = {y}')
# [output]: x1 = [0 1 0 1], x2 = [0 0 1 1]
# [output]: y = [ True False False True]