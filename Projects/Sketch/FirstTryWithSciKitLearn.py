"""
My very first attempt at creating a neutal network model using keras. I try to
model a sine wave using as target output the scalar x[n] and as input the 2D 
vector (x[n-1], x[n-2]). It should actually be possible to predict this signal 
perfectly using linear units. Let's see, if this does indeed work...

References:

  (1) https://scikit-learn.org/stable/modules/neural_networks_supervised.html#regression
  (2) https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
  
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt                # For plotting the results    
plt.style.use('dark_background')   


# Create the training data. We synthesize a time series of a sinusoid
w = 0.1                       # Normalized radian frequency
N = 201                       # Number of samples
t = np.linspace(0.0, N-1, N)  # Time axis
s = np.sin(w*t)               # Our sine wave

# We now have the time series for the sine in s. From that signal, we now 
# extract a bunch of input vectors (of dimension 2) and scalar target outputs:
D = 2                         # Maximum delay    
X = np.full((N-D, 2), 0.0)
y = np.full((N-D),    0.0)
for n in range(0, N-D):
    X[n,0] = s[n]
    X[n,1] = s[n+1]
    y[n]   = s[n+2]
    

#fig1 = plt.figure()
plt.plot(t, s)
#plt.plot(t[0:N-D], y)


# Create and set up a multilayer perceptron regressor:
mlp = MLPRegressor(hidden_layer_sizes=(2,), activation="tanh",
                   max_iter=900) 
mlp.fit(X, y)

# Use the trained MLP for prediction:
p = mlp.predict(X);
plt.plot(t[0:N-D], p)

#p = np.zeros(N)
#for n in range(2, n):
#    p[n] = mlp.predict(X)


# .fit, .predict

"""

ToDo:
    
  - Maybe try adding some noise to the data
  - Figure out, if we can use custom activation functions. If so, try
    f(x) = x / sqrt(1 + x^2). It's cheaper than the sigmoid and potentially 
    less prone to the vanishing gradients problem because it goes into 
    saturation more slowly (I think)
  - Figure out, how to produce multidimensional output
  - Figure out what mlp.loss_curve is - 
  
    

See also:
    
  https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py

"""