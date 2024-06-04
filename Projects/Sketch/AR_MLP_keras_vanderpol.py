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
from keras.models     import Sequential
from keras.layers     import Input
from keras.layers     import Dense
from keras.optimizers import RMSprop
from keras.callbacks  import EarlyStopping 

# Imports from my own libraries:
from rs.dynsys     import van_der_pol            # To generate the input data
from rs.datatools  import signal_ar_to_nn        # For data reformatting
#from rs.learntools import synthesize_skl_mlp     # Resynthesize via MLP model

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance:
#tf.config.experimental.enable_op_determinism()
# See: https://keras.io/examples/keras_recipes/reproducibility_recipes/

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
layers  = [3]            # Numbers of neurons in the layers
act_fun = "tanh"         # Activation function
seed    = 0              # Seed for PRNG

#==============================================================================
# Processing

# Create signal:
t  = np.linspace(0.0, t_max, in_len)        # Time axis    
vt = odeint(van_der_pol,                    # Solution to the ODE
            [x0, y0], t, args=(mu,))
s = vt[:, dim]                              # Select one dimension

# Build the model:
model = Sequential()
model.add(Input(shape=(len(delays),)))      # Input layer
for i in range(0, len(layers)):
    L = Dense(layers[i],                    # Hidden layer i
              activation = act_fun)
    model.add(L) 
model.add(Dense(1))                         # Output layer
model.compile(
   loss = 'mse', 
   optimizer = RMSprop(), 
   metrics = ['mean_absolute_error']
)
# ToDo: Check, which activation function is used by the output layer. Maybe
# it's the default ReLU? I guess linear would be better for the output layer.
# Check also activation function for the input layer. But I guess, the input 
# layer just sends the value as-is to the next layer

# Train the model:
       
# Calling keras.utils.set_random_seed will set the seed in Python, numpy and 
# Keras' backend:
keras.utils.set_random_seed(seed)
# See: https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed

    
X, y  = signal_ar_to_nn(s, delays)          # Extract data for modeling
history = model.fit(
   X, y,    
   #batch_size=128, 
   epochs  = 100, 
   verbose =   1,
   #validation_split = 0.2, 
   callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20)]
)
# 






'''

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












