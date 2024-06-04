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
from keras.models     import Sequential
from keras.layers     import Input
from keras.layers     import Dense
from keras.optimizers import RMSprop
from keras.callbacks  import EarlyStopping 

# Imports from my own libraries:
from rs.dynsys     import van_der_pol            # To generate the input data
from rs.datatools  import signal_ar_to_nn        # For data reformatting
#from rs.learntools import synthesize_skl_mlp     # Resynthesize via MLP model

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
act_fun = "tanh"         # Activation function (identity, tanh, logistic, relu)

#==============================================================================
# Processing

# Create signal:
t  = np.linspace(0.0, t_max, in_len)     # Time axis    
vt = odeint(van_der_pol,                 # Solution to the ODE
            [x0, y0], t, args=(mu,))
s = vt[:, dim]                           # Select one dimension for time series

# Create the model:
X, y  = signal_ar_to_nn(s, delays)          # Extract data for modeling
model = Sequential()
model.add(Input(shape=(len(delays),)))      # Input layer
#model.add(Dense(3, activation = 'tanh')) 
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

history = model.fit(
   X, y,    
   #batch_size=128, 
   epochs  = 80, 
   verbose =  1,
   #validation_split = 0.2, 
   callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20)]
)
# 















