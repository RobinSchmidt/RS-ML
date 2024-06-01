# ToDo: Move this file into the Libraries folder

import numpy as np

def van_der_pol(v, t, mu):
    '''
    Derivative calculation for a van der Pol oscillator. It is defined by the 
    autonomous system of two ordinary differential equations:
        
      x' = y                                   \n
      y' = mu * (1 - x^2) * y - x

    Parameters
    ----------
    v : np.array of 2 floats
        The current state of the oscillator. v = [x, y]
    t : float
        The current time instant. Not used but needed as dummy arg for 
        compatibility with odeint.
    mu : float
        The system parameter. Higher values make the oscillator more nonlinear.

    Returns
    -------
    np.array of 2 floats
        The derivative vector v' = [x', y'] 
    '''
    x, y = v[0], v[1]
    xd = y                             # x' = y
    yd = mu * (1 - x**2) * y - x       # y' = mu * (1 - x^2)*y - x
    return np.array([xd, yd])
# https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
# https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form
# ToDo: 
# -Document properly, make mu optional (maybe default to 1 or zero)
# -Factor out a function for the pure derivative calculation, i.e. without the
#  code x,y = ... and np.array(...) ...ah . no - wait - it already is in the 
#  correct format. We want it to be in the format that is consumed by odeint
