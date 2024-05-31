# We generate and plot the phase space trajectory and time series of the Lorenz 
# system defined by the (autonomous) system of ordinary differential equations:
#
#   x' = a * (y - x)
#   y' = b*x - x*z - y
#   z' = x*y - c*z
#
# where x = x(t), y = y(t), z = z(t) are three coupled variables that change 
# with respect to time and x' = dx/dt, y' = dy/dt, z' = dz/dt are their 
# derivatives with respect to time. The system behaves chaotically ...TBC...

# Import libraries:
import numpy as np                    # For array data types
import matplotlib.pyplot as plt       # For plotting the results
from scipy.integrate import odeint    # The numerical ODE solver

# Configure libraries:
plt.style.use('dark_background')      # I really like dark mode!

# Define system parameters:
a = 10
b = 28
c = 8/3.0
# Using higher values for both increases the frequency and also seems to make 
# the x-curve steeper and the y-curve more spikey.

# Define initial conditions. We start close to zero:
x0 = 0                                # x(0)
y0 = 1                                # y(0)
z0 = 0                                # z(0)

# Define length of signal:
tMax = 80.0                           # Is the unit seconds? I guess so.
N    = 10001                          # Numer of samples

# Define the function to compute the derivative of the Van der Pol system:
def f(v, t):
    x  = v[0]
    y  = v[1]
    z  = v[2]
    xd = a * (y-x)                     # x' = ...
    yd = b*x - x*z - y                 # y' = ...
    zd = x*y - c*z                     # z' = ...
    return np.array([xd, yd, zd])

# Produce the data for plotting. The solution vt = v(t) is a vector-valued 
# time series:
t       = np.linspace(0.0, tMax, N)    # Time axis
v0      = np.array([x0, y0, z0])       # Initial conditions
vt      = odeint(f, v0, t)             # Solution of the ODE
(x,y,z) = vt.T                         # Extract x(t), y(t) from vector v(t)

# Plot a projection of the phase-space trajectory onto xz-plane:
fig1 = plt.figure()
plt.plot(x, z)

# Plot the three time series x(t), y(t), z(t):
fig2 = plt.figure()
plt.plot(t, x, color='lightsalmon')
plt.plot(t, y, color='skyblue')
plt.plot(t, z, color='wheat')

# Plot the trajectory in 3D (it's still a bit ugly, though):
fig3 = plt.figure()
ax   = fig3.add_subplot(projection='3d')
ax.plot(x, y, z)
plt.show()

# ToDo:
#
# https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html#sphx-glr-gallery-mplot3d-quiver3d-py

