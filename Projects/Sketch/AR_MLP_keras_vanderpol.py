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
#from keras.optimizers import RMSprop

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

#==============================================================================
# Processing

# Create signal:
t  = np.linspace(0.0, t_max, in_len)   # Time axis    
vt = odeint(van_der_pol,               # Solution to the ODE
            [x0, y0], t, args=(mu,))
s = vt[:, dim]                         # Select one dimension for time series

# Create the model
X, y = signal_ar_to_nn(s, delays)      # Extract data for modeling

model = Sequential()
model.add(Input(shape=(4,)))     # use len(delays)
#model.add(Dense(64, kernel_initializer = 'normal', activation = 'relu',
#                input_shape = (13,))) 
model.add(Dense(10, activation = 'tanh')) 
model.add(Dense(1))
# This produces a warning!

















