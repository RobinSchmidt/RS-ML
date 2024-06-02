"""

"""

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt    
from scipy.integrate import odeint                  # The numerical ODE solver
from sklearn.neural_network import MLPRegressor

# Imports from my own libraries:
import sys
sys.path.append("../../Libraries")
from rs.dynsys import van_der_pol
from rs.datatools import delayed_values, signal_ar_to_nn
  
# Create the signal:
tMax = 50
N    = 401                               # Number of samples
t    = np.linspace(0.0, tMax, N)         # Time axis
mu   = 1.0
x0   = 0
y0   = 1
vt   = odeint(van_der_pol,               # Solution to the ODE
              [x0, y0], t, args=(mu,))
s = vt[:,0]
#s = vt[:,1]  # Alternative

# Set up the delays to be used:
d = [1,2]
D = max(d)
# Set up more modeling parameters here - like number of hidden neurons, etc.

# Extract a bunch of input vectors and scalar target outputs for learning:
X, y = signal_ar_to_nn(s, d)

# Fit a multilayer perceptron regressor to the data and use it for prediction:
mlp = MLPRegressor(hidden_layer_sizes=(4,), activation="tanh",
                   max_iter=10000, tol=1.e-7, random_state = 0) 
mlp.fit(X, y)
p = mlp.predict(X);



# Under construction....
# 
# Now let's do a real autoregressive synthesis using the mlp model. It just 
# takes an initial section as input and continues it up to a given desired 
# length using the predictions of the mlp recursively:
L  = 200           # Desired length for prediction
qs = s[50:150]     # Initial section to be used
q  = np.zeros(L)   # Signal that we want to generate
Li = len(qs)       # Length of given initial section
q[0:Li] = qs       # Copy given initial section into our result

# Recursive prediction of the values q[n] for n >= Li
for n in range(Li, L):
    xn = delayed_values(q, n, d)
    yn = mlp.predict(xn.reshape(1, -1))
    q[n] = yn[0]

plt.plot(q)        # Preliminary for debugging
#
# End of "Under construction". Eventually, the code written here should go into
# a function synthesize_skl_mlp(mlp, qs, L)


    
# Plot reference and predicted signal:
#plt.style.use('dark_background') 
#plt.figure()    
#plt.plot(t,      s)                    # Input signal
#plt.plot(t[D:N], p)                    # Predicted signal

# Plot training loss curve:
#plt.figure()
#loss = mlp.loss_curve_
#plt.plot(loss)                         # The whole loss progression
#plt.figure()
#plt.plot(loss[3000:4000])              # A zoomed in view of the tail

"""
Observations:
    
- The predicted signal p actually looks too good to be true. I think, using 
  p = mlp.predict is not the right thing to do. We want to predict recursively, 
  i.e. use previous predictor outputs. Predicting from X uses the true input 
  signal values for prediction. Maybe we need to modify the file for the sine, 
  too. ..OK...done: we now also have the signal q, which should be what we 
  actually want. Some more tests are needed, though
  
- Let K be the number of neurons in the (single) hidden layer and let's pick 
  tanh as actiavtion function and S be the random see. The results are as 
  follows: 
    K =  2, S = 0: garbage
    K =  3, S = 0: good
    K =  3, S = 1: garbage
    K =  4, S = 0: good
    K =  5, S = 3: garbage
    K = 13, S = 0: garbage
    
    
Conclusions:

- The result depends crucially on the random seed. Maybe to find a global 
  optimum we should train a lot of networks of each architechture and pick
  the best among them.
  
- Maybe we could take several of the best ones and try to create even better 
  ones by means of evolutionary algorithms
  
  
ToDo:
    
- Do a more quantitative assesments of the different trained networks. 
  Currently, I just say good or garbage based on visual inspection. Maybe 
  compute a prediction error compare the values of the different networks.
  
"""