# We generate and plot the phase space trajectory and time series of the Van 
# der Pol system defined by the (autonomous) system of ordinary differential 
# equations:
#
#   x' = a * y
#   y' = b * (1 - x^2) * y - x
#
# where x = x(t) and y = y(t) are two coupled variables that change with 
# respect to time and x' = dx/dt, y' = dy/dt are their derivatives with respect 
# to time. The system quickly settles into a (quasi?) periodic motion.

# Import libraries:
import numpy as np                    # For array data types
import matplotlib.pyplot as plt       # For plotting the results
from scipy.integrate import odeint    # The numerical ODE solver

# Configure libraries:
plt.style.use('dark_background')      # I really like dark mode!

# Define system parameters:
a = 2.2 
b = 2.0
# Using higher values for both increases the frequency and also seems to make 
# the x-curve steeper and the y-curve more spikey.

# Define initial conditions. We start close to zero:
x0 = +0.001                           # x(0)
y0 = -0.001                           # y(0)

# Define length of signal:
tMax = 30.0                           # Is the unit seconds? I guess so.

# Define the function to compute the derivative of the Van der Pol system:
def f(v, t):
    xd = a * v[1]                     # x' = a * y
    yd = b * (1-v[0]**2)*v[1]-v[0]    # y' = b * (1 - x^2)*y - x
    return np.array([xd, yd])

# Produce the data for plotting. The solution vt = v(t) is a vector-valued 
# time series:
t     = np.linspace(0.0, tMax, 1001)  # Time axis
v0    = np.array([x0, y0])            # Initial conditions
vt    = odeint(f, v0, t)              # Solution of the ODE
(x,y) = vt.T                          # Extract x(t), y(t) from vector v(t)

# Plot the trajectory in phase space:
fig1 = plt.figure()
plt.plot(x, y)

# Plot the two time series x(t), y(t):
fig2 = plt.figure()
plt.plot(t, x, color='lightsalmon')
plt.plot(t, y, color='skyblue')

# ToDo:
#
# Demonstrate the quiver function to plot a vector field and stream lines