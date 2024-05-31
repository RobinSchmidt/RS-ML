# Plots phase space trajectory and time series of Van der Pol system

# Import libraries:
import numpy as np    
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Configure libraries:
plt.style.use('dark_background')

# Define the function to compute the derivative of the Van der Pol system:
def f(v, t):
    xd = 10.0 * v[1]                   # x' = 10 * y
    yd = 10.0 * (1-v[0]**2)*v[1]-v[0]  # y' = 10 * (1 - x^2)*y - x
    return np.array([xd, yd])

# Produce the data for plotting:
t     = np.linspace(0.0, 10.0, 1001)  # Time axis
v0    = np.array([1.0, 1.0])          # Initial conditions
vt    = odeint(f, v0, t)              # Solution of the ODE
(x,y) = vt.T                          # Extract x(t), y(t) from vector v(t)

# Phase-space plot:
fig1 = plt.figure()
plt.plot(x, y)


# To set up colors:
# https://stackoverflow.com/questions/14088687/how-to-change-plot-background-color
# https://matplotlib.org/stable/gallery/style_sheets/dark_background.html

# Some simple example plots:
# https://matplotlib.org/stable/gallery/pyplots/index.html