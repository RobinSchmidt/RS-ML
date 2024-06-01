"""

"""

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt    
from sklearn.neural_network import MLPRegressor

# Imports from my own libraries:
import sys
sys.path.append("../../Libraries")
#from rs.dynsys import van_der_pol
from rs.datatools import signal_ar_to_nn
  
# Create the training data. We synthesize a time series of a sinusoid:
w = 0.1                                # Normalized radian frequency
N = 201                                # Number of samples
t = np.linspace(0.0, N-1, N)           # Time axis
s = np.sin(w*t)                        # Our sine wave

# We now have the time series for the sine in s. From that signal, we now 
# extract a bunch of input vectors (of dimension 2) and scalar target outputs:
D = 2                                  # Maximum delay    
X = np.zeros((N-D, 2))
y = np.zeros( N-D)
for n in range(0, N-D):
    X[n,0] = s[n]
    X[n,1] = s[n+1]
    y[n]   = s[n+2]

X2, y2 = signal_ar_to_nn(s, [1,2])
# this looks wrong. Maybe we should implement a unit test


# Fit a multilayer perceptron regressor to the data and use it for prediction:
mlp = MLPRegressor(hidden_layer_sizes=(2,), activation="identity",
                   max_iter=4000, tol=1.e-7, random_state = 0) 
mlp.fit(X, y)
p = mlp.predict(X);

# Plot reference and predicted signal:
plt.style.use('dark_background') 
plt.figure()    
plt.plot(t,      s)                    # Input signal
plt.plot(t[D:N], p)                    # Predicted signal

# Plot training loss curve:
plt.figure()
loss = mlp.loss_curve_
plt.plot(loss)                         # The whole loss progression
plt.figure()
plt.plot(loss[3000:4000])              # A zoomed in view of the tail

"""
Observations:
"""