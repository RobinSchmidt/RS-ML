# Plots phase space trajectory and time series of Van der Pol system

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Configure libraries:
plt.style.use('dark_background')

# Define parameters:
a = 2.2 
b = 2.0   

# Define the function to compute the derivative of the Van der Pol system:
def f(v, t):
    xd = a * v[1]                   # x' = 10 * y
    yd = b * (1-v[0]**2)*v[1]-v[0]  # y' = 10 * (1 - x^2)*y - x
    return np.array([xd, yd])

# Produce the data for plotting:
t     = np.linspace(0.0, 10.0, 1001)  # Time axis
v0    = np.array([1.0, 1.0])          # Initial conditions
vt    = odeint(f, v0, t)              # Solution of the ODE
(x,y) = vt.T                          # Extract x(t), y(t) from vector v(t)

# Phase-space plot:
fig1 = plt.figure()
plt.plot(x, y)

# Time series plot:
fig2 = plt.figure()
plt.plot(t, x, color='lightsalmon')
plt.plot(t, y, color='skyblue')