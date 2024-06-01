'''
DynSys: Dynamical Systems
=========================

This module contains functions that implement the equations of various 
continuous nonlinear dynamical systems such as the Lorenz system, Van der Pol 
oscillator, Volterra-Lotka model etc. The functions have all the same signature 
and can be called like:

    derivs = f(state, time, ...)
  
where "state" is the current state vector of the system, time is a scalar time 
variable and after that, a bunch of parameters for the particular system may 
follow. The signature of the functions is compatible with the odeint routine 
from scipy.integrate which implements a numerical ODE solver. Functions that
implement the equations of autonomous systems will ignore the "time" parameter.
It is still there as dummy parameter for the compatibility with odeint, though.
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



'''
ToDo:
    
-Add Volterra-Lotka model,    

'''