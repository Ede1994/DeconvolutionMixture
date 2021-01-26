# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:49:00 2021

@author: Eric
"""

import numpy as np
from scipy.optimize import minimize

# Observation A has the following measurements:
A = np.array([0, 4.1, 5.6, 8.9, 4.3])
# How similar is A to ideal groups identified by the following:
group1 = np.array([1, 3, 5, 10, 3])
group2 = np.array([6, 3, 2, 1, 10])
group3 = np.array([3, 3, 4, 2, 1])

# Define the objective function
# x is the array containing your wanted coefficients
def obj_fun(x, A, g1, g2, g3):
    y = x[0] * g1 + x[1] * g2 + x[2] * g3
    return np.sum((y-A)**2)

# Bounds for the coefficients
bnds = [(0, 1), (0, 1), (0, 1)]
# Constraint: x[0] + x[1] + x[2] - 1 = 0
cons = [{"type": "eq", "fun": lambda x: x[0] + x[1] + x[2] - 1}]

# Initial guess
xinit = np.array([1, 1, 1])
res = minimize(fun=obj_fun, args=(A, group1, group2, group3), x0=xinit, bounds=bnds, constraints=cons)
print(res.x)