"""
We do the same thing as in AR_MLP_sklearn_vanderpol but this time with keras 
instead of sklearn.
"""

#==============================================================================
# Imports and Config

import os
import sys
import numpy             as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint         # Numerical ODE solver

# Configuration:
os.environ["KERAS_BACKEND"] = "torch"      # Keras shall use PyTorch as backend
plt.style.use('dark_background')           # Plot in dark mode
sys.path.append("../../Libraries")         # Make available for import

# Imports from Keras:
import keras    
import keras.optimizers as optis
from keras.models     import Sequential
from keras.layers     import Input
from keras.layers     import Dense
from keras.callbacks  import EarlyStopping 

# Imports from my own libraries:
from rs.dynsys     import van_der_pol            # To generate the input data
from rs.datatools  import signal_ar_to_nn        # For data reformatting
from rs.learntools import synthesize_keras_mlp   # Resynthesize via MLP model

#tf.config.experimental.enable_op_determinism()
# If using TensorFlow, this will make GPU ops as deterministic as possible
# but it will affect performance. See:
# https://keras.io/examples/keras_recipes/reproducibility_recipes/

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
delays  = [1,2,3,4]      
layers  = [10]           # Numbers of neurons in the hidden layers
act_fun = "relu"         # Activation function
seed    = 0              # Seed for PRNG
epochs  = 200
verbose = 1

loss    = 'mse'

#opt     = optis.RMSprop()
#opt     = optis.SGD()
opt     = optis.Adam()      # Yep
#opt     = optis.Adadelta()  # Nope!
#opt = optis.Adagrad()        # Nope!
#opt = optis.AdamW()         # Yep
#opt = optis.Adamax()         # Mixed

# d = [1,2,3,4], l = [3], tanh, seed = 6, epochs = 200, opti = Adam
# -> high freq oscillations / unstable. 1,2,4,5,9 lead to constant 
# output, 7 seems to produce the right shape after some settling time. 0 or 3 
# produces a sine-like shape.

# Good results were achieved with:
# d = [1,2,3,4], l = [10], tanh, seed = 6,7, epochs = 200, opti = AdamW,Adam 

# With l = [8,4,2] and tanh, I get mostly bad results, relu with seed 1 or 4 
# seems to work well (using Adam)

# Resynthesis parameters:
syn_len = 400            # Length of resynthesized signal
syn_beg = 150            # Beginning of resynthesis

#==============================================================================
# Processing

# Create signal:
t  = np.linspace(0.0, t_max, in_len)        # Time axis    
vt = odeint(van_der_pol,                    # Solution to the ODE
            [x0, y0], t, args=(mu,))
s = vt[:, dim]                              # Select one dimension

# Set seed for reproducible results:
keras.utils.set_random_seed(seed)
#
# - Calling keras.utils.set_random_seed will set the seed in Python, numpy and 
#   Keras' backend. See:
#   https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed
#
# - We seem to have to do this before building the model. Doing it right before
#   the call to model.fit() seems to be too late. When doing that, the first 
#   run of the script after a kernel reset and variable clearing will produce a 
#   different result than subsequent runs. Moreover, the result of that first 
#   run is different every time. ToDo: Figure out and document why!

# Build the model:
model = Sequential()
model.add(Input(shape=(len(delays),)))      # Input layer
for i in range(0, len(layers)):
    L = Dense(layers[i],                    # Hidden layer i
              activation = act_fun)  
    model.add(L)
model.add(Dense(1, activation = 'linear'))  # Output layer
model.compile(
   loss = loss, 
   optimizer = opt, 
   metrics = ['mean_absolute_error']
)
# Notes:
#
# - For the output layer, 'linear' is apparently the default activation 
#   function anyway, but for documentation's sake, it's nice to state it 
#   explicitly.
#
# - ToDo: explain some of the other settings like the loss-function, optimizer,
#   metrics, etc. Maybe pass some more parameters for further customization.

# Train the model:
X, y = signal_ar_to_nn(s, delays)           # Extract/convert data for modeling
history = model.fit(
   X, y,    
   #batch_size=128, 
   epochs  = epochs, 
   verbose = verbose,
   #validation_split = 0.2, 
   callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20)]
)
# ToDo:
#
# - Explain some of the other parameters licke epochs, etc. This seems to
#   be different from sklearn's MLPRegressor which has a tolerance parameter. 
#   Figure this out! And what is the purpose of this this "callbacks" thing?

# Evaluate the model:
score = model.evaluate(X, y, verbose = 1) 
print('Test loss:',     score[0]) 
print('Test accuracy:', score[1])

# Use the model for synthesis:
D  = max(delays)
qs = s[(syn_beg-D):syn_beg]            # Initial section to be used
q  = synthesize_keras_mlp(model, delays, qs, syn_len)    

# Compute synthesis error signal for the region where input and synthesized 
# signals overlap:
s_chunk = s[syn_beg-D:in_len-D]
q_chunk = q[0:len(s_chunk)]
error   = s_chunk - q_chunk
max_err = max(abs(error)) / max(abs(s_chunk))    #  Maximum relative error

#==============================================================================
# Visualization

# Create shifted time axis for resynthesized signal:
tr = np.linspace(syn_beg-D, syn_beg-D+syn_len, syn_len)
tr = tr * (t_max / (in_len-1))

# Plot reference and predicted signal:
#plt.figure()    
#plt.plot(t, s)                         # Input signal
#plt.plot(tr, q)                        # Synthesized signal

# Plot original, synthesized and error signal for the region where they 
# overlap:
plt.figure()
plt.plot(s_chunk)
plt.plot(q_chunk)
plt.plot(error)
print("Max error: ", max_err)



'''
Observations:
    
- When observing the training progress (by passing "verbose = 1" to model.fit),
  we notice that neither the loss nor mean_absolute_error decreases 
  monotonically. At least not when using "optimizer = RMSprop()". Maybe that's
  a feature of this particular optimizer? It seems weird to me. Try other 
  optimizers! Aha! SGD seems to give much better results than RMSOpt

- With th SGD optimizer, we seem to need something like 200 epochs. With less,
  we seem to get models that produce a constant output


ToDo:
    
- Try using TensorFlow as backend just to show that it works, too. Maybe we 
  will get different results because the random weight initialization works 
  differently in TensorFlow? Try it!

Activation functions:
https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
https://keras.io/api/layers/activations/
https://datascience.stackexchange.com/questions/58884/how-to-create-custom-activation-functions-in-keras-tensorflow

Apparently, we do not need to specify the derivative. This:
https://stackoverflow.com/questions/51754639/how-to-define-the-derivative-of-a-custom-activation-function-in-keras
says that the derivative will be computed by automatic differentiation. But 
maybe only when using TensorFlow as backend?

https://sefiks.com/2018/12/01/using-custom-activation-functions-in-keras/

https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/


How about asinh? or x^(3/5), x^(5/7) ...I think, numerator and denominator 
should be odd (for odd symmetry) and the fraction should be greater than 1/2 
for finite gradient at 0

'''












