
# ToDo: Move this file into the Libraries folder

import numpy as np

# Van der Pol system:
def van_der_pol(v, t, mu):
    xd = v[1]                          # x' = y
    yd = mu * (1-v[0]**2)*v[1]-v[0]    # y' = mu * (1 - x^2)*y - x
    return np.array([xd, yd])
# ToDo: find better names for the variables. Maybe we should use only one and
# call it mu?
# https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
# https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form