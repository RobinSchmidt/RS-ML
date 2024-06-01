'''
DynSys: Dynamical Systems
=========================

This module contains functions that implement the equations of various 
continuous nonlinear dynamical systems such as the Lorenz system, Van der Pol 
oscillator, Volterra-Lotka model, Roessler attractor, etc. The functions have 
all the same signature and can be called like:

    derivs = f(state, time, ...)
  
where "state" is the current state vector of the system, "time" is a scalar 
time variable and after that, a bunch of parameters for the particular system 
may follow. The return value is the vector of derivative values at the given
time instant. Vectors are passed in and out as NumPy arrays.

The signature of the functions is compatible with the odeint routine 
from scipy.integrate which implements a numerical ODE solver. Functions that
implement the equations of autonomous systems will ignore the "time" parameter.
It is still there as dummy parameter for the compatibility with odeint, though.
'''

import numpy as np                     # I/O is done via numpy.array


def lorenz(v, t, sigma, rho, beta):
    '''
    Derivative calculation for the Lorenz system ...TBC...

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    References
    ----------
    https://en.wikipedia.org/wiki/Lorenz_system
    '''
    x, y, z = v[0], v[1], v[2]
    xd = sigma * (y - x)               # x' = sigma * (y - x)
    yd = x * (rho - z) - y             # y' = x * (rho - z) - y
    zd = x*y - beta*z                  # z' = x*y - beta*z
    return np.array([xd, yd, zd])


def van_der_pol(v, t, mu):
    '''
    Derivative calculation for the Van der Pol oscillator. This is originally a 
    nonlinear, autonomous, second order, ordinary differential equation (ODE). 
    Its two-dimensional form, which we implement here, is given by the 
    following system of two first order ODEs:
        
      x' = y                                   \n
      y' = mu * (1 - x^2) * y - x

    Parameters
    ----------
    v : np.array of 2 floats
        The current state of the oscillator v = [x, y].
    t : float
        The current time instant.
    mu : float
        The system parameter. Higher values make the oscillator more nonlinear.

    Returns
    -------
    v' : np.array of 2 floats
        The derivative vector v' = [x', y'].
        
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
    
-Add Volterra-Lotka model, Roessler attractor, and maybe more... 

https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor

'''