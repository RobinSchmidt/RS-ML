"""
My very first attempt at creating a neural network model using sklearn. I try 
to model a sine wave using as target output the scalar x[n] and as input the 2D 
vector (x[n-1], x[n-2]). So, we are doing an autoregressive model of the sine
signal. It should actually be possible to predict this signal perfectly using 
linear units because a sine can be created autoregressively. It can be done by
a 2nd order digital filter. Let's see, if this does indeed work...Yep!

References:

  (1) https://scikit-learn.org/stable/modules/neural_networks_supervised.html#regression
  (2) https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
  
"""

# Import and configure libraries:
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt    
plt.style.use('dark_background')   

# Create the training data. We synthesize a time series of a sinusoid:
w = 0.1                       # Normalized radian frequency
N = 201                       # Number of samples
t = np.linspace(0.0, N-1, N)  # Time axis
s = np.sin(w*t)               # Our sine wave

# We now have the time series for the sine in s. From that signal, we now 
# extract a bunch of input vectors (of dimension 2) and scalar target outputs:
D = 2                         # Maximum delay    
X = np.full((N-D, D), 0.0)
y = np.full((N-D),    0.0)
for n in range(0, N-D):
    X[n,0] = s[n]
    X[n,1] = s[n+1]
    y[n]   = s[n+2]    

# Fit a multilayer perceptron regressor to the data and use it for prediction:
mlp = MLPRegressor(hidden_layer_sizes=(2,), activation="identity",
                   max_iter=4000, tol=1.e-7, random_state = 0) 
mlp.fit(X, y)
p = mlp.predict(X);

# Plot reference and predicted signal:
plt.figure()    
plt.plot(t,      s)  # Input signal
plt.plot(t[D:N], p)  # Predicted signal

# Plot training loss curve:
plt.figure()
loss = mlp.loss_curve_
plt.plot(loss)
#plt.plot(loss[3000:4000])



"""
Observations:
    
- The loss curve drops quickly until around 1000 iterations. Then it looks 
  close to zero. However, when reducing max_iter, we get a warning that the 
  training didn't converge. OK - zooming in, it becomes apparent that the loss 
  continues to drop.



ToDo:
    
- Maybe try adding some noise to the data
- Figure out, if we can use custom activation functions. If so, try
  f(x) = x / sqrt(1 + x^2). It's cheaper than the sigmoid and potentially 
  less prone to the vanishing gradients problem because it goes into 
  saturation more slowly (I think)
- Figure out, how to produce multidimensional output
- Figure out what mlp.loss_curve is - 
- Maybe use array slicing to produce the X,y data arrays
- Maybe introduce variables for the network setup stuff (max_iter, tol, etc.)
  
    

See also:
    
  https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py

"""