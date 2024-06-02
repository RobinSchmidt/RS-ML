"""
In this script, we create a nonlinear autoregressive model of the Van der Pol
oscillator using a multilayer perceptron from the Scikit-Learn library. Then, 
we evaluate the so created model by visually inspecting the output that it 
generates.

ToDo: Looking at plots is well and good but at some point, we should also 
compute some objective, quantitative error measures ...TBC...
"""

#==============================================================================
# Imports

# Imports from third party libraries:
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

#==============================================================================
# Setup
  
# Signal parameters:
tMax    = 50             # Maximum time value 
N       = 401            # Number of samples - maybe rename to in_len (input length)
mu      = 1.0            # Nonlinearity parameter
x0      = 0.0            # Initial condition x(0)
y0      = 1.0            # Initial condition y(0)
dim     = 0              # Dimension to use as time series. 0 or 1 -> x or y

# Modeling parameters:
delays  = [1,2,3]        # Delay times (in samples)
layers  = (3,)           # Numbers of neurons in the layers
act_fun = "tanh"         # Activation function (identity, tanh, logistic, relu)
seed    = 0              # Seed for PRNG
tol     = 1.e-12         # Tolerance for fitting
max_its = 10000          # Maximum number of training iterations (epochs?)

# Resynthesis parameters:
syn_len = 400            # Length of resynthesized signal
syn_beg = 150            # Beginning of resynthesis    

#==============================================================================
# Processing

# Create signal:
t  = np.linspace(0.0, tMax, N)         # Time axis    
vt = odeint(van_der_pol,               # Solution to the ODE
            [x0, y0], t, args=(mu,))
s = vt[:,dim]                          # Select one dimension for time series

#s = vt[:,0]
#s = vt[:,1]  # Alternative
# ToDo: have a signal parameter "dim" or similar that selects, which dimension
# we wnat to use as our time series. Then here, do s = vt[:,dim]

# Fit a multilayer perceptron regressor to the data and use it for prediction:
X, y = signal_ar_to_nn(s, delays)  # Extract input vectors and scalar outputs 
mlp  = MLPRegressor(hidden_layer_sizes = layers, activation = act_fun,
                    max_iter = max_its, tol = tol, random_state = seed) 
mlp.fit(X, y)
p = mlp.predict(X);

# Now let's do a real autoregressive synthesis using the mlp model. It just 
# takes an initial section as input and continues it up to a given desired 
# length using the predictions of the mlp recursively:
D  = max(delays)
qs = s[(syn_beg-D):syn_beg]            # Initial section to be used
q  = synthesize_skl_mlp(mlp, delays, qs, syn_len)

# Compute synthesis error signal for the region where input and synthesized 
# signals overlap:
s_chunk = s[syn_beg:N]
q_chunk = q[0:len(s_chunk)]
error   = s_chunk - q_chunk

#==============================================================================
# Visualization

# Create shifted time axis for resynthesized signal:
tr = np.linspace(syn_beg, syn_beg+syn_len, syn_len)
tr = tr * (tMax / (N-1))  # Yes - we need to divide by N-1. Look at t and t2.

# Plot reference and predicted signal:
plt.style.use('dark_background') 
plt.figure()    
plt.plot(t,      s)                    # Input signal
plt.plot(t[D:N], p)                    # Predicted signal
plt.plot(tr,     q)                    # Synthesized signal

# Plot original, synthesized and error signal for the region where they 
# overlap:
plt.figure()
plt.plot(s_chunk)
plt.plot(q_chunk)
plt.plot(error)

# Plot training loss curve:
plt.figure()
loss = mlp.loss_curve_
plt.plot(loss)                         # The whole loss progression
plt.figure()
#plt.plot(loss[3000:4000])              # A zoomed in view of the tail

#==============================================================================
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
    
- Do a more quantitative evaluations of the different trained networks. 
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
  
- Try other model types, for example:
  https://scikit-learn.org/stable/modules/svm.html#regression
  Maybe with SVMs, try to use the "kernel-trick". I think, it consists of 
  forming nonlinear combinations of the delayed values such as products. 
  
- Try a linear MLP (i.e. using the identity activation function) together with
  input vectors that contain nonlinear combinations of delayed input values. I 
  think, when using products for these nonlinear combination, we essentially do 
  something like a Volterra kernel model.
  
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
  
- Eventually, the goal is to apply it to musical instrument samples as 
  explained in the paper "Neural Network Modeling of Speech and Music Signals" 
  by Axel Roebel, see: https://hal.science/hal-02911718  
  
- Try to model both time series x[n], y[n] simultaneously using an MLP with 2
  output neurons. I'm not sure, if that's possible with sklearn. If not, look 
  into keras for that
  
"""