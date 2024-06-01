'''
We generate and plot the phase space trajectory and time series of the Van 
der Pol system defined by the (autonomous) system of ordinary differential 
equations:
    
  x' = y  
  y' = mu * (1 - x^2) * y - x

where x = x(t) and y = y(t) are two coupled variables that change with 
respect to time and x' = dx/dt, y' = dy/dt are their derivatives with respect 
to time. The system quickly settles into a (quasi?) periodic motion.
'''

# Imports from standard libraries:
import numpy as np                    # For array data types
import matplotlib.pyplot as plt       # For plotting the results
from scipy.integrate import odeint    # The numerical ODE solver

# Imports from my own libraries:
import sys
sys.path.append("../../Libraries")
from rs.dynsys import van_der_pol     # Derivative calculation for the system

# Set up user parameters:
mu   =  1.0                           # System parameter
x0   = +0.001                         # Initial condition x(0)
y0   = -0.001                         # Initial condition y(0)
tMax =  50.0                          # Length (in seconds?)


# Produce the data for plotting. The solution vt = v(t) is a vector-valued 
# time series:
t  = np.linspace(0.0, tMax, 1001)     # Time axis
vt = odeint(van_der_pol,              # Solution to the ODE
            [x0, y0], t, args=(mu,))


# Plot the trajectory in phase space:
(x,y) = vt.T                          # Extract x(t), y(t) from vector v(t)    
plt.style.use('dark_background')      # I really like dark mode!
fig1 = plt.figure()
plt.plot(x, y)

# Plot the two time series x(t), y(t):
fig2 = plt.figure()
plt.plot(t, x, color='lightsalmon')
plt.plot(t, y, color='skyblue')

'''
Observations:
    
- Using mu = 0, x0 = 0, y0 = 1, we get a sine/cosine pair, i.e. x = sin, 
  y = cos. With x0 = 1, y0 = 0, we'll get x = cos, y = -sin.
  
- Using higher values for mu makes the x-curve steeper and the y-curve more 
  spikey. With 0.0, we get a sine/cosine

- Using small values for the initial conditions, we see a build-up phase.

- Uinsg mu = -0.1 gives (approximately?) a decaying sinusoid
 
 
ToDo: 

- Demonstrate the quiver function to plot a vector field and stream lines

- Try to import van_der_pol via relative imports. I'd like to write it in one 
  line like "from ...Libraries.rs.dynsys import va_der_pol" but that doesn't 
  work. Figure out if and how this can be made to work
 
'''