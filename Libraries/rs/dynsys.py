'''
This module contains...TBC...
'''

import numpy as np

def van_der_pol(v, t, mu):
    '''
    Derivative calculation for a Van der Pol oscillator. This is originally a 
    nonlinear, autonomous, second order, ordinary differential equation (ODE). 
    Its two-dimensional form, which we implement here, is given by the 
    following system of two first order ODEs:
        
      x' = y                                   \n
      y' = mu * (1 - x^2) * y - x
      
    The function can be used with scipy.integrate.odeint.

    Parameters
    ----------
    v : np.array of 2 floats
        The current state of the oscillator. v = [x, y]
    t : float
        The current time instant. Not used in the computation but needed as 
        dummy parameter for compatibility with odeint.
    mu : float
        The system parameter. Higher values make the oscillator more nonlinear.

    Returns
    -------
    v' : np.array of 2 floats
        The derivative vector v' = [x', y'] 
        
    References
    ----------
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form 
    '''
    x, y = v[0], v[1]
    xd = y                             # x' = y
    yd = mu * (1 - x**2) * y - x       # y' = mu * (1 - x^2)*y - x
    return np.array([xd, yd])
