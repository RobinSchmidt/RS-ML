"""
My very first attempt at creating a neural network model using sklearn. I 
create a linear autoregressive (AR) model for a sine wave signal using a 
multilayer perceptron (MLP). Autoregressive means that the network tries to 
predict the current sample from past samples. A single sine wave could be 
produced by a 2nd order digital LTI filter. We don't produce our sine that way 
here, though - but it is possible. That means that a neural network with just 
1 hidden layer containing 2 neurons with linear activation functions should be 
able to predict the signal perfectly.

If the sinusoidal signal is contained in a 1D array s[n], then the data for the 
network to learn looks like:

  X = [(s[0], s[1]), (s[1], s[2]), (s[2], s[3]), ...]
  y = [ s[2],         s[3],         s[4],        ...]

The target output is always s[n] and the input vector is always 
(s[n-1], s[n-2]) where n = 2,..,N. We start at n = 2, because that's the first
index for which we have 2 past samples available. If s is N samples long, i.e. 
n = 0,..,N-1, then y will we N-2 samples long and X will have a shape of (N,2).
"""

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt    
from sklearn.neural_network import MLPRegressor
  
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
# ToDo: Generalize and factor that out into a function timeSeriesToData(s, d) 
# where d is a vector of delays. Here, d = [1,2]. Call it here like:
#   X, y = timeSeriesToData(s, [1,2])
# Or maybe call the function signalToData

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

# Plot log of training loss curve:
plt.figure()
loss = mlp.loss_curve_
plt.plot(np.log10(loss))

"""
Observations:
    
- Using 2 hidden neurons in 1 layer with an identity actiavtions function, we 
  should theoretically be able to achieve a perfect match and that seems to 
  indeed work in practice
  
- When switching to the tanh activation function, there is some residual error.
  It can be reduced by increasing the number of hidden neurons to, say, 5.
  
- With relu, we also get good results with 5 hidden neurons.

- The loss curve drops quickly until around 1000 iterations. Then it looks 
  close to zero. However, when reducing max_iter, we get a warning that the 
  training didn't converge. OK - zooming in, it becomes apparent that the loss 
  continues to drop.
  
- The random_state = ... in the creation of the MLPRegressor object really 
  makes a difference. We get a good result with 0 but much less good with 2, 
  for example. 


ToDo:
    
- Currently, we only predict one sample into the future using delayed 
  *original* input samples. Ultimately, we are interested in recursive 
  prediction. Add that. See AR_MLP_sklearn_vand_der_pol. There, we already do 
  that. 
    
- Try to explain why with some random seeds, we do not seem to able to get good
  results. After all, with a linear network, the loss function should be a 
  hyperparaboloid with a single global minimum and no local minima, right?
  
- Maybe try adding some noise to the data

- Figure out, if we can use custom activation functions. If so, try
  f(x) = x / sqrt(1 + x^2). It's cheaper than the sigmoid and potentially 
  less prone to the vanishing gradients problem because it goes into 
  saturation more slowly (I think)
  
- Figure out, how to produce multidimensional output

- Maybe use array slicing to produce the X,y data arrays

- Maybe introduce variables for the network setup stuff (max_iter, tol, etc.)

- Apply an exponential decay. It should still be possible to model it perfecly
  with 2 hidden (linear) neurons.
  
- Add a 2nd sine (with amplitude and phase parameters). To model the signal,
  properly, we'll probably need to add 2 more hidden neurons.
  
- Try it with a Van der Pol system and with the Lorenz system

- Experiment with different seeds for the random initialization of the weights.

- Figure out, if we can manually initialize the weights. If so, start with a 
  given know set of weights, run the learning and the try to replicate the 
  results with my C++ implementation.  
  
- Write a utility function that takes the time series s and a delay vector d 
  and produces the training data X, y from that. In this example, the delay 
  vector would be [1, 2] because we use s[n-1] and s[n-2] for the prediction. 
  We want to able to use general delays like e.g. [1,2,3,4,5, 29,30,31] because
  that is want I want to try for autoregressive modeling of musical instrument
  samples. In particular, I want to use [1,2,3, ..., P-1,P,P+1] where P is the
  period of the signal. Assuming a periodic signal and using some delays 
  surrounding the period, it should be possible to perfectly predict the value 
  at n from the value at n-P alone - s[n] is just equal to s[n-P]. If P is 
  non-integer, some interpolation may need to take place, so we use not only 
  n-P, but also a couple of samples around it.

    
See:
    
- https://scikit-learn.org/stable/modules/neural_networks_supervised.html#regression
- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor   
- https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py

"""