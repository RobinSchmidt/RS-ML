"""
In this script, we create a nonlinear autoregressive model of the Van der Pol
oscillator using a multilayer perceptron from the Scikit-Learn library. Then, 
we evaluate the so created model by visually inspecting the output that it 
generates. In the bottom section is a table where I noted down my visual 
impression of different models using different random seeds for the weight
initialization. I have also computed a crude preliminary error measure which is
noted down into the table.
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
from rs.dynsys     import van_der_pol, lorenz    # To generate the input data
from rs.datatools  import signal_ar_to_nn        # For data reformatting
from rs.learntools import synthesize_skl_mlp     # Resynthesize via MLP model

#==============================================================================
# Setup
  
# Signal parameters:
ode     = 'lorenz'       # Select the ODE system
p1      = 10.0           # 1st parameter
p2      = 28.0           # 2nd ...
p3      = 8.0/3          # 3rd ...
x0      = -1.81          # x(0) initial condition
y0      = -0.89          # y(0) ...
z0      = 21.38          # z(0) ...
dim     = 1              # Dimension to use as signal. 0 is x, 1 is y, 2 is z
t_max   = 200             # Maximum time value 
in_len  = 10001           # Number of input samples

# Modeling parameters:
delays  = 2*[1,2,3,5,6,7]  # Delay times (in samples)
layers  = (50)           # Numbers of neurons in the layers
act_fun = "tanh"         # Activation function (identity, tanh, logistic, relu)
seed    = 5              # Seed for PRNG
fit_tol = 1.e-16         # Tolerance for fitting
max_its = 10000          # Maximum number of training iterations (epochs?)

# Resynthesis parameters:
syn_len =  10000          # Length of resynthesized signal
syn_beg =   100          # Beginning of resynthesis

#==============================================================================
# Processing

# Create signal:
t  = np.linspace(0.0, t_max, in_len)   # Time axis    

#vt = odeint(van_der_pol,               # Solution to the ODE
#            [x0, y0], t, args=(mu,))

if(ode == 'van_der_pol'): 
    vt = odeint(van_der_pol, [x0,y0],    t, args=(p1,))
elif(ode == 'lorenz'):
    vt = odeint(lorenz,      [x0,y0,z0], t, args=(p1,p2,p3))
else:
    vt = np.zeros(len(t))
    # ToDo: Maybe throw a warning "Unknown ODE string" or something
    
# ToDo:           
# 
# Maybe use a "match"-statement, see:
# https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/        

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
s_chunk = s[syn_beg-D:in_len-D]
q_chunk = q[0:len(s_chunk)]
error   = s_chunk - q_chunk
max_err = max(abs(error)) / max(abs(s_chunk))    #  Maximum relative error

#max_err = max(error) / max(s_chunk)    #  Maximum relative error
# !!! BUG !!! We need to do  max(abs(error)) / max(abs(s_chunk))
# Then, the table below needs to be re-evaluated. We expect 50% of its content 
# to be wrong!

#==============================================================================
# Visualization

# Create shifted time axis for resynthesized signal:
tr = np.linspace(syn_beg-D, syn_beg-D+syn_len, syn_len)
tr = tr * (t_max / (in_len-1))

# Plot reference and predicted signal:
#plt.figure()    
#plt.plot(t, s)                         # Input signal
#plt.plot(t[D:in_len], p)               # Predicted signal
#plt.plot(tr, q)                        # Synthesized signal

# Throwaway code for looking at Lorenz system modling output
plotMax = 2000
plt.figure()
plt.plot(s[0:plotMax])
plt.plot(p[0:plotMax])
plt.figure()
plt.plot(q[0:plotMax]) 

# Plot original, synthesized and error signal for the region where they 
# overlap:
#plt.figure()
#plt.plot(s_chunk)
#plt.plot(q_chunk)
#plt.plot(error)
print("Max error: ", max_err)

# Plot log of training loss curve:
#plt.figure()
#loss = mlp.loss_curve_
#plt.plot(np.log10(loss))

#==============================================================================
"""
Observations:
 
The table below shows results for the following setup:

# Signal parameters:
ode     = 'van_der_pol'  # Select the ODE system
p1      = 1.0            # 1st parameter
p2      = 0.0            # 2nd ...
p3      = 0.0            # 3rd ...
x0      = 0.0            # x(0) initial condition
y0      = 1.0            # y(0) ...
z0      = 0.0            # z(0) ...
dim     = 0              # Dimension to use as signal. 0 is x, 1 is y, 2 is z
t_max   = 50             # Maximum time value 
in_len  = 401            # Number of input samples

# Modeling parameters:
delays  = See table      # Delay times (in samples)
layers  = See table      # Numbers of neurons in the layers
act_fun = See table      # Activation function (identity, tanh, logistic, relu)
seed    = See table      # Seed for PRNG
fit_tol = 1.e-16         # Tolerance for fitting
max_its = 10000          # Maximum number of training iterations (epochs?)

# Resynthesis parameters:
syn_len = 400            # Length of resynthesized signal
syn_beg = 150            # Beginning of resynthesis        

The MaxErr values are rounded down (because the values were copy-and-pasted). 
The subdivisions of the table occur whenever the model type is changed, i.e. a 
different set of delays, hidden neurons or activation function is used

===============================================================
|    Delays    | Layers | ActFunc  | Seed | MaxErr |  Look    |
|=============================================================|
| 1,2          | 3      | tanh     |  0   | 1.7101 | Trash    |
| 1,2          | 3      | tanh     |  1   | 1.1268 | Wobbly   |
|-------------------------------------------------------------|
| 1,2,3        | 3      | tanh     |  0   | 0.7939 | Fast     |
| 1,2,3        | 3      | tanh     |  1   | 1.8415 | Trash    |
| 1,2,3        | 3      | tanh     |  2   | 1.9640 | Trash    |
| 1,2,3        | 3      | tanh     |  3   | 1.6238 | Trash    |
| 1,2,3        | 3      | tanh     |  4   | 2.0073 | Trash    |
| 1,2,3        | 3      | tanh     |  5   | 2.1479 | Unstable |
| 1,2,3        | 3      | tanh     |  6   | 0.9008 | Fast     |
| 1,2,3        | 3      | tanh     |  7   | 1.8362 | Trash    |
| 1,2,3        | 3      | tanh     |  8   | 0.6420 | Slow     |
| 1,2,3        | 3      | tanh     |  9   | 0.3766 | Good     |
|-------------------------------------------------------------|
| 1,2,3,4      | 10     | tanh     |  0   | 0.0572 | Perfect  |  *
| 1,2,3,4      | 10     | tanh     | 1..3 |        | Fast     |
| 1,2,3,4      | 10     | tanh     |  4   |        | Slow     |
| 1,2,3,4      | 10     | tanh     |  5   | 0.0732 | Perfect  |
|-------------------------------------------------------------|
| 1,2,3,4      | 20     | tanh     |  2   | 0.0535 | Perfect  |  *
| 1,2,3,4      | 20     | tanh     |  6   | 0.0213 | Perfect  |  *
|-------------------------------------------------------------|
| 1,2,3,4      | 32     | tanh     |0..20 |        | All Good | 
|-------------------------------------------------------------|
| 1,2,3,4      | 6,3    | tanh     |  0   | 2.7651 | Unstable |
|-------------------------------------------------------------|
| 1,2,3,4      | 7,3    | tanh     |  0   | 0.8076 | Fast     |
|-------------------------------------------------------------|
| 1,2,3,4      | 6,4    | tanh     |  0   | 1.0264 | Fast     |
|-------------------------------------------------------------|
| 1,2,3,4      | 5,5    | tanh     |  0   | 0.4652 | Good     |
|-------------------------------------------------------------|
| 1,2,3,4      | 8,4,2  | tanh     |  0   | 0.2554 | Wobbly   |
| 1,2,3,4      | 8,4,2  | tanh     |  1   | 1.8763 | Shifted  |
| 1,2,3,4      | 8,4,2  | tanh     |  2   | 1.9080 | Trash    |
| 1,2,3,4      | 8,4,2  | tanh     |  3   | 1.7335 | Fast     |
| 1,2,3,4      | 8,4,2  | tanh     |  4   | 1.8063 | F,S      |
| 1,2,3,4      | 8,4,2  | tanh     |  5   | 1.1275 | Fast     |
| 1,2,3,4      | 8,4,2  | tanh     |  6   | 1.7338 | Fast     |
| 1,2,3,4      | 8,4,2  | tanh     |  7   | 1.6416 | Bad      |
| 1,2,3,4      | 8,4,2  | tanh     |  8   | 1.8659 | Const    |
| 1,2,3,4      | 8,4,2  | tanh     |  9   | 1.6370 | Wobbly   |
|-------------------------------------------------------------|
| 1,2,3,4      | 4,3,2  | tanh     |  5   | 2.4396 | Unstable |
|-------------------------------------------------------------|
| 1,2,3        | 3      | logistic | 0..5 |        | Trash    |
| 1,2,3        | 3      | logistic |  6   | 0.3398 | Good     |
|-------------------------------------------------------------|
| 1,2,3,4      | 10     | relu     |  0   | 0.1103 | Perfect  |
===============================================================  *

The MaxErr values might be wrong because they were obtained with the buggy 
code:
    
  max_err = max(error) / max(s_chunk)

which has now been fixed to:
    
  max_err = max(abs(error)) / max(abs(s_chunk))
  
So - yeah - this data needs to be re-collected. This requires some manual grunt
work. ...someday...maybe...




The table contains some visual assesment of the outputs produced by the model. 
The words have the following meaning:

Perfect:   A very good fit - almost perfect.
Good:      Good fit
Bad:       Bad fit
Fast:      Frequency too high
Slow:      Frequency too low
Wobbly:    Frequency wobbles (sometimes fast, sometimes slow)
Shifted:   Shape looks okay, but signal is shifted in time
Trash:     Total garbage that has nothing to do with original
Unstable:  Runaway oscillations or explosion

Sometimes, the signal has more than one of the features (like shifted and 
fast). In such cases, we use only the first letters like F,S. ToDo: Some of the
settings classified as "Trash" could be further specified as "Const" because
they converge to a constant. Apparently, the quality of the attractor has 
changed from quasi-periodic to a fixed point.

- If enough hidden neurons are used, the results seem to become less dependent 
  on the seed. With 32 neurons, I got good results with every seed I tried 
  (which was 0..20).

- I tried to sample the input signal with higher and lower sample rates by
  increasing in_len (and also syn_len accordingly). The result was that the 
  models tend to give better fits when in_len is smaller. A lower sampling rate 
  seems to make the modeling task easier. Thinking about it, this seems to be 
  plausible because our unit sample delays cover a greater time window. We see
  "more" of the past signal in terms of the absolute time window. Maybe with
  higher sampling rates, it could be beneficial to apply a Haar transform on
  the delayed samples first. Don't use the delayed samples directly but instead
  use their Haar trafo (or maybe Walsh-Hadamard trafo or other kind of Wavelet
  trafo - maybe Daubechies is worth to try as well). See:
  https://en.wikipedia.org/wiki/Haar_wavelet
  https://en.wikipedia.org/wiki/Hadamard_transform
  https://en.wikipedia.org/wiki/Daubechies_wavelet
  But this kind of pre-processing of the vector of delayed values makes only 
  sense if the delays form a continuous sequences like 1,2,3,4 - but not 
  something like 1,2,5,8. It would be linear pre-processing layer.
  
- Another idea for the pre-processing transformation: For the purpose of an 
  example, let's assume d = [1,2,3,4]. Maybe we can fit a cubic polynomial to 
  these points s[n-1]...s[n-4] and evaluate its 0th, 1st, 2nd and 3rd 
  derivative at n = 0. This vector of 4 derivative values is then our 
  transformed vector on which we perform the usual MLP training. This idea can 
  be generalized to arbitrary degrees. Such a linear pre-processing stage can 
  generally expressed by a matrix. Maybe for research purposes, we should have 
  an API the let's us specify such a pre-processing matrix. In production, 
  we'll probably want to use special matrices for which the matrix-vector 
  product can be computed efficiently (like in O(N) or O(N*log(N)))

- OLD - but maybe some settings should be included in the table above:
  Let K be the number of neurons in the (single) hidden layer and let's pick 
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
  to have gone completely. ...okay - I have now set fit_tol to 1.e-16. It 
  doesn't seem to unduly lengthen the training process and seems to solve the
  problem. ToDo: figure out and document, what the "tol" parameter in 
  MLPRegressor exactly does. Is it a tolerance for the norm of the gradient or
  what?
    
Conclusions:

- The result depends crucially on the random seed. Maybe to find a global 
  optimum we should train a lot of networks of each architechture and pick
  the best among them.
  
- Maybe we could take several of the best ones and try to create even better 
  ones by means of evolutionary algorithms
  
- With tanh, we seem to get good results only with 1 or 2 hidden layers. More 
  hidden layers tend to yield unstable systems. ..or well..maybe not generally.
  (5,4,3,2) seems to be stable
  
-------------------------------------------------------------------------------
Modeling the Lorenz system:

- I did not yet find a good set of parameters to model it well. My current best
  attempt was:

      
# Signal parameters:
ode     = 'lorenz'       # Select the ODE system
p1      = 10.0           # 1st parameter
p2      = 28.0           # 2nd ...
p3      = 8.0/3          # 3rd ...
x0      = -1.81          # x(0) initial condition
y0      = -0.89          # y(0) ...
z0      = 21.38          # z(0) ...
dim     = 1              # Dimension to use as signal. 0 is x, 1 is y, 2 is z
t_max   = 200             # Maximum time value 
in_len  = 10001           # Number of input samples

# Modeling parameters:
delays  = [1,2,3,5,6,7]  # Delay times (in samples)
layers  = (50)           # Numbers of neurons in the layers
act_fun = "tanh"         # Activation function (identity, tanh, logistic, relu)
seed    = 3              # Seed for PRNG
fit_tol = 1.e-16         # Tolerance for fitting
max_its = 10000          # Maximum number of training iterations (epochs?)

# Resynthesis parameters:
syn_len =  10000          # Length of resynthesized signal
syn_beg =   100          # Beginning of resynthesis

Variations of the above settings that also gave reasonable looking results:
delays = [1,2,3,5,6,7], layers = (30), seed = 4
delays = [1,2,3,5,6,7], layers = (20), seed = 1

- It appears that we never seem to get enough downward teeth. The models seem 
  to want to do only two oscillations in the bottom section.

- Using  delays = [1,2,3,5,6,7], layers = (60), seed = 1 
  gives a quasi-periodic signal alternating between 3 up and 3 down teeth

- Using  delays = 2*[1,2,3,5,6,7], layers = (50), seed = 5
  also kinda worked, so we may also use delays that are spaced further apart.



An older (worse, low order) attempt was:

    
# Signal parameters:
ode     = 'lorenz'       # Select the ODE system
p1      = 10.0           # 1st parameter
p2      = 28.0           # 2nd ...
p3      = 8.0/3          # 3rd ...
x0      = -1.81          # x(0) initial condition
y0      = -0.89          # y(0) ...
z0      = 21.38          # z(0) ...
dim     = 1              # Dimension to use as signal. 0 is x, 1 is y, 2 is z
t_max   = 20             # Maximum time value 
in_len  = 1001           # Number of input samples

# Modeling parameters:
delays  = [1,2,3,4]      # Delay times (in samples)
layers  = (3)            # Numbers of neurons in the layers
act_fun = "tanh"         # Activation function (identity, tanh, logistic, relu)
seed    = 2              # Seed for PRNG
fit_tol = 1.e-16         # Tolerance for fitting
max_its = 10000          # Maximum number of training iterations (epochs?)

# Resynthesis parameters:
syn_len =  1000          # Length of resynthesized signal
syn_beg =   100          # Beginning of resynthesis







- The initial conditions were chosen to be a point on or near the attractor to
  get rid of the transient in the time series because such a transient is not
  representative of the system dynamics and therefore we don't want to see it
  in our training data.


-------------------------------------------------------------------------------  
  
  
  
ToDo:
    
- Do a more quantitative evaluations of the different trained networks. 
  Currently, I just say good or garbage based on visual inspection. Maybe 
  compute a prediction error compare the values of the different networks.
  
- Add the delay vector to the table of results, Maybe also the value of mu and
  which signal coordinate we model (x or y)
  
- Maybe for the training, we should remove the transient part of the signal to
  avoid fitting thoses parts of the signal that are not representative of the
  dynamics - or are they? ...not sure

- Try a different loss function. Check, what is available. I guess, using 
  different loss functions alters the loss function landscape ins ways that may
  make it more or less easy to find a (global) minimum. Maybe it could even be 
  beneficial to use different loss functions during different phases of the 
  training?

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
  In the local Ananconda installation in Windows, the file is here:
      
  C:/Users/rob/anaconda3/Lib/site-packages/sklearn/neural_network
  ....this path with backslashes trips up the Python interpreter!!!
  
  But maybe use TensorFlow or PyTorch instead. Or better: use Keras which is
  a high-level interface that can use both of them as backend. To define custom
  activation functions in keras, see:
  https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
  
- I think, it would also be very nice, if we could specify the activation 
  function per layer rather than for the whole network at once. Ideally, I may
  at some stage be able to specify it per neuron. Maybe for certain tasks, it
  may make sense to have some tanh neurons, some linear neurons, some swish
  neurons etc. per layer.
  
- The sklearn MLP seems to be indeed rather limited, see:
  https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron  
  So, we should really swicth to a more powerful library. I guess, I should 
  look into Keras next.
  
- Eventually, the goal is to apply it to musical instrument samples as 
  explained in the paper "Neural Network Modeling of Speech and Music Signals" 
  by Axel Roebel, see: https://hal.science/hal-02911718  
  
- Try to model both time series x[n], y[n] simultaneously using an MLP with 2
  output neurons. I'm not sure, if that's possible with sklearn. If not, look 
  into keras for that
  
- Create a notebook from this - this can be used for a portfolio in a job 
  application
  
- Try to explore the space of all possible models more systematically to find a 
  globally optimal one. Maybe start with a bunch of smaller models, select the 
  best and apply some growing strategy. Like: (1) Copy the model including the
  weights, (2) Add a neuron, (3) Re-train. (4) Accept bigger model only if it's
  really better than it's smaller predecessor. Maybe write a class 
  ModelExplorer. Maybe we need a better way to measure the error. Maybe a 
  (stretched?) cross-correlation could be appropriate. By "stretched", I mean 
  to first apply time stretching to the synthesized signal to compesate for 
  models that are too fast or slow. We still may want to rate them goof, if the 
  shape matches well and only the frequency is off. A wrong frequency can be 
  dealt with by interpolation.
  
- For network pruning, we could compute the correlations between the input
  weight vectors and if two have a high correlation, we could perhaps collapse
  the neurons into a single one. The input weights of the new neuron that 
  replaces the two correlated ones may be given by the average of the two and 
  the output weights by their sum. Rationale: The new neuron's activation 
  should respond to inputs similarly to the two original ones - that's why we 
  average the input weights. The neurons in the next layer should be affected
  by the new neuron's activation similarly to the two original ones - that's
  why we take the sum.
  
- Figure out if the final performance of the model correlates with the early
  performance. The goal is to find out, if in our exploration of the model 
  space, it may be possible to identify promising models early, i.e. after 
  partial training. If a model that eventually performs well already shows this
  in early stages of the training, we do not need to train all explored models
  fully which saves a lot of training time. ..Hmm...doesn't seem to be the case
  
"""