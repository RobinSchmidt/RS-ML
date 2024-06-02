"""
In this script, we create a nonlinear autoregressive model of the Van der Pol
oscillator using a multilayer perceptron from the Scikit-Learn library. Then, 
we evaluate the so created model by visually inspecting the output that it 
generates.

ToDo: Looking at plots is well and good but at some point, we should also 
compute some objective, quantitative error measures ...TBC...
"""

#==============================================================================
# Imports and Config

# Imports from third party libraries:
from scipy.integrate        import odeint        # Numerical ODE solver
from sklearn.neural_network import MLPRegressor  # Multilayer perceptron    
import numpy             as np
import matplotlib.pyplot as plt
import sys

# Configuration:
plt.style.use('dark_background')                 # Plot in dark mode
sys.path.append("../../Libraries")               # Make available for import

# Imports from my own libraries:
from rs.dynsys     import van_der_pol            # To generate the input data
from rs.datatools  import signal_ar_to_nn        # For data reformatting
from rs.learntools import synthesize_skl_mlp     # Resynthesize via MLP model

#==============================================================================
# Setup
  
# Signal parameters:
t_max   = 50             # Maximum time value 
in_len  = 401            # Number of input samples
mu      = 1.0            # Nonlinearity parameter
x0      = 0.0            # Initial condition x(0)
y0      = 1.0            # Initial condition y(0)
dim     = 0              # Dimension to use as time series. 0 or 1 -> x or y

# Modeling parameters:
delays  = [1,2,3,4]      # Delay times (in samples)
layers  = (10,)          # Numbers of neurons in the layers
act_fun = "relu"         # Activation function (identity, tanh, logistic, relu)
seed    = 0              # Seed for PRNG
fit_tol = 1.e-16         # Tolerance for fitting
max_its = 10000          # Maximum number of training iterations (epochs?)

# Resynthesis parameters:
syn_len = 400            # Length of resynthesized signal
syn_beg = 150            # Beginning of resynthesis    

#==============================================================================
# Processing

# Create signal:
t  = np.linspace(0.0, t_max, in_len)   # Time axis    
vt = odeint(van_der_pol,               # Solution to the ODE
            [x0, y0], t, args=(mu,))
s = vt[:, dim]                         # Select one dimension for time series

# Fit a multilayer perceptron regressor to the data and use it for prediction:
X, y = signal_ar_to_nn(s, delays)  # Extract input vectors and scalar outputs 
mlp  = MLPRegressor(hidden_layer_sizes = layers, activation = act_fun,
                    max_iter = max_its, tol = fit_tol, random_state = seed) 
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
#s_chunk = s[syn_beg:in_len]
#s_chunk = s[syn_beg-2:in_len-2]
s_chunk = s[syn_beg-D:in_len-D]
q_chunk = q[0:len(s_chunk)]
error   = s_chunk - q_chunk
max_err = max(error) / max(s_chunk)    #  Maximum relative error
# It looks like q is shifted with respect to s. Check, if we have an off-by-one
# error somewhere - in the synthesis or in the extraction of the chunks etc.
# Maybe make unit test with an extremely simple signal - a straight line
# ...ah! I see I have an off-by-D error. The synthesis procedure copies the 
# initial section into the synthesized signal which is D samples long and 
# prepended to the actually synthesized section
# Maybe wrap the model evaluation into a class ModelEvaluator_skl_mlp. Maybe
# write a class ModelExplorer which programmatically searches the design space
# for good models.

#==============================================================================
# Visualization

# Create shifted time axis for resynthesized signal:
#tr = np.linspace(syn_beg, syn_beg+syn_len, syn_len)
tr = np.linspace(syn_beg-D, syn_beg-D+syn_len, syn_len)
tr = tr * (t_max / (in_len-1))
# I think, this might still be wrong. It looks misaligned. It may have to do
# with a shift by D ...OK - seems to be fixed.

# Plot reference and predicted signal:
plt.figure()    
plt.plot(t, s)                         # Input signal
plt.plot(t[D:in_len], p)               # Predicted signal
plt.plot(tr, q)                        # Synthesized signal

# Plot original, synthesized and error signal for the region where they 
# overlap:
plt.figure()
plt.plot(s_chunk)
plt.plot(q_chunk)
plt.plot(error)
print("Max error: ", max_err)

# Plot training loss curve:
plt.figure()
loss = mlp.loss_curve_
plt.plot(loss)                         # The whole loss progression
plt.figure()
#plt.plot(loss[3000:4000])              # A zoomed in view of the tail

#==============================================================================
"""
Observations:
 
Results for     

MaxErr still wrong - I had a bug in the aligment between original and 
synthesized    

======================================================================
|    Delays    | Hidden Layers | ActFunc | Seed | MaxErr |  Look     |
|====================================================================|
| 1,2,3        | 3             | tanh    |  0   | 0.5084 | Good      |
| 1,2,3        | 3             | tanh    |  1   | 1.7368 | Trash     |
| 1,2,3        | 3             | tanh    |  2   | 2.0051 | Trash     |
| 1,2,3        | 3             | tanh    |  5   | 2.1488 | Unstable  |
| 1,2,3,4      | 6,3           | tanh    |  0   | 2.7652 | Unstable  |
| 1,2,3,4      | 10            | tanh    |  0   | 0.0572 | Very Good |
| 1,2          | 3             | tanh    |  0   | 1.6047 | Slow      |
======================================================================


  
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
  think, when using products for these nonlinear combinations, we essentially 
  do something like a Volterra kernel model.
  
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
  
- Create a notebook from this - this can be used for a portfolio in a job 
  application
  
"""