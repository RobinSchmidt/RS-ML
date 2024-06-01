# ToDo: Move this file into the Libraries folder

import numpy as np

# Van der Pol oscillator with parameter mu:
def van_der_pol(v, t, mu):
    x, y = v[0], v[1]
    xd = y                             # x' = y
    yd = mu * (1 - x**2) * y - x       # y' = mu * (1 - x^2)*y - x
    return np.array([xd, yd])
# https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
# https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form
# ToDo: document properly