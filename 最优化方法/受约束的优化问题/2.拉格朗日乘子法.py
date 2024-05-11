import sympy
from scipy import *
import matplotlib.pyplot as plt
import numpy as np


x = x1, x2, x3, l = sympy.symbols("x_1, x_2, x_3, lambda")
f = x1 * x2 * x3
g = 2 * (x1 * x2 + x2 * x3 + x3 * x1) - 1
L = f + l * g
grad_L = [sympy.diff(L, x_) for x_ in x]
sols = sympy.solve(grad_L)
print(sols)

print(g.subs(sols[0]))
print(f.subs(sols[0]))

