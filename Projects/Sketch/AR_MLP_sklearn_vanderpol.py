"""
In this script, we create a nonlinear autoregressive model of the Van der Pol
oscillator using a multilayer perceptron from the Scikit-Learn library. Then, 
we evaluate the so created model by visually inspecting the output that it 
generates.

ToDo: Looking at plots is well and good but at some point, we should also 
compute some objective, quantitative error measures ...TBC...
"""

# Imports third party libraries:
import numpy             as np
import matplotlib.pyplot as plt    
from scipy.integrate        import odeint        # Numerical ODE solver
from sklearn.neural_network import MLPRegressor  # Multilayer perceptron

# Imports from my own libraries:
import sys
sys.path.append("../../Libraries")
from rs.dynsys     import van_der_pol
from rs.datatools  import signal_ar_to_nn
from rs.learntools import synthesize_skl_mlp
  
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

# Set up the modeling parameters:
delays  = [1,2,3]     # Delay times (in samples)
layers  = (3,)        # Numbers of neurons in the layers
actfun  = "tanh"      # Activation function (identity, tanh, logistic, relu)
seed    = 0           # Seed for PRNG
tol     = 1.e-12      # Tolerance for fitting
max_its = 10000       # Maximum number of training iterations (epochs?)

D = max(delays)

# Set up more modeling parameters here - like number of hidden neurons, etc.

# Extract a bunch of input vectors and scalar target outputs for learning:
X, y = signal_ar_to_nn(s, delays)

# Fit a multilayer perceptron regressor to the data and use it for prediction:
mlp = MLPRegressor(hidden_layer_sizes = layers, activation = actfun,
                   max_iter = max_its, tol = tol, random_state = seed) 
mlp.fit(X, y)
p = mlp.predict(X);


# Now let's do a real autoregressive synthesis using the mlp model. It just 
# takes an initial section as input and continues it up to a given desired 
# length using the predictions of the mlp recursively:
L  = 300                                  # Desired length for prediction
qs = s[50:100]                            # Initial section to be used
q  = synthesize_skl_mlp(mlp, delays, qs, L);   
plt.plot(q)                               # Preliminary for debugging

# ToDo:
#
# - Maybe let the initial section be of length D = max(d). Any data before is 
#   not used for the synthesis anyway. Maybe have variables 
#   syn_start (here 50...or 100?), syn_len (here 300)
# 
# - Make a plot that overlays the synthesized signal q with the original signal
#   s in those regions where they overlap. Maybe it should be 




    
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
    
  =============================================
  |Hidden Layers | Act. Func. | Seed | Result |
  |===========================================|
  | 3            | ReLU       |   0  | Bad    |
  | 13           | ReLU       |   0  | Good   |
  | 8,4,2        | ReLU       |   0  | Good   |
  | 8,4,2        | ReLU       |   1  | Good   |
  | 8,4,2        | ReLU       |   2  | Good   |
  --------------------------------------------|
  | 8,4,2        | tanh       |   0  | Bad    |  
  | 8,4,2        | tanh       |   1  | Trash  |
  | 8,4,2        | tanh       |   2  | Trash  |
  | 8,4,2        | tanh       |   3  | Trash  |
  | 4,2          | tanh       |   0  | OK     |
  | 6,3          | tanh       |   0  | Good   |     
  | 3            | tanh       |   0  | So-so  |
  =============================================
     
- With d = [1,2,3], HL = (13,), AF = ReLU, Seed = 0, the prediction gets 
  unstable! Use L = 300 to show this behavior. Using tanh tames the amplitude 
  but we get wirld oscillations at much higher freq than we should
  
- With d = [1,2,3], HL = (8,4,2), AF = ReLU, Seed = 0, we get all zeros

- With d = [1,2,3], HL = (4,2), AF = tanh, Seed = 0 - it looks quite good
    
- With mu = 0, we get a sine wave. When we try to model it with linear units,
  the model tends to introduce an undesired decay. But this decay tends to go 
  away when we reduce the tolerance in the learning/fitting stage, tol=1.e-7 
  shows string decay. Using 1.e-9, we get much less decay. At 1.e-12, it seems
  to have gone completely.
    
Conclusions:

- The result depends crucially on the random seed. Maybe to find a global 
  optimum we should train a lot of networks of each architechture and pick
  the best among them.
  
- Maybe we could take several of the best ones and try to create even better 
  ones by means of evolutionary algorithms
  
- With tanh, we seem to get good results only with 1 or 2 hidden layers. More 
  hidden layers tend to yield unstable systems. ..or well..maybe not generally.
  (5,4,3,2) seems to be stable
  
  
ToDo:
    
- Do a more quantitative assesments of the different trained networks. 
  Currently, I just say good or garbage based on visual inspection. Maybe 
  compute a prediction error compare the values of the different networks.
  
- Add the delay vector to the table of results, Maybe also the value of mu and
  which signal coordinate we model (x or y)
  
- Maybe for the training, we should remove the transient part of the signal to
  avoid fitting thoses parts of the signal that are not representative of the
  dynamics - or are they? ...not sure

- Try other learning algorithms. This here:
  https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html  
  says that for smaller data sets, 'lbfgs' may converge faster.
  
- Try custom activation functions. See:
  https://datascience.stackexchange.com/questions/18647/is-it-possible-to-customize-the-activation-function-in-scikit-learns-mlpclassif
  It's not possible with sklearn by default but the article says that one might
  add it to the library. See here:
  https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neural_network/_base.py
  But maybe use TensorFlow or PyTorch instead. Or better: use Keras which is
  a high-level interface that can use both of them as backend. To define custom
  activation functions in keras, see:
  https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
  
- I think, it would also be very nice, if we could specify the activation 
  function per layer rather than for the whole network at once. Ideally, I may
  at some stage be able to specify it per neuron. Maybe for certain tasks, it
  may make sense to have some tanh neurons, some linear neurons, some swish
  neurons etc. per layer.
  
"""