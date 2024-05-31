"""
My very first attempt at creating a neutal network model using keras. I try to
model a sine wave using as target output the scalar x[n] and as input the 2D 
vector (x[n-1], x[n-2]). It should actually be possible to predict this signal 
perfectly using linear units. Let's see, if this does indeed work...
"""

# Import and configure libraries:
import numpy as np                             # For array data types
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt                # For plotting the results    
plt.style.use('dark_background')   

# Trying to run it produces:
#
#   ModuleNotFoundError: No module named 'keras'
#
# Apparently, keras is not part of Anaconda and has to be installed seperately.
# For the time being, I'll pass on this and try to do it with SciKitLearn 
# instead

"""
References:
    
  (1) https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
  (2) https://www.activestate.com/resources/quick-reads/how-to-create-a-neural-network-in-python-with-and-without-keras/
"""