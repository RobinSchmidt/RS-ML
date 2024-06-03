"""
My first attempt with keras ...TBC...
"""

# Tell the keras library that we want to use PyTorch as backend:
import os
os.environ["KERAS_BACKEND"] = "torch"
# Without this configuration, the "import keras" statement below will throw the 
# following error message:
#
#   "ModuleNotFoundError: No module named 'tensorflow'"
#
# unless TensorFlow is installed. This is because keras uses TensorFlow as 
# backend by default.

import numpy as np
import keras 
#from keras.models     import Sequential 
#from keras.layers     import Dense 
#from keras.optimizers import RMSprop 
#from keras.callbacks import EarlyStopping 



'''

Hmmm...Keras throws an error - it doesn't find the tensorflow mdoule

See:
    
Seems to need only keras:
https://www.tutorialspoint.com/keras/keras_regression_prediction_using_mpl.htm    
    
TensorFlow based:    
https://pyimagesearch.com/2019/01/21/regression-with-keras/
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

PyTorch based:
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
https://machinelearningmastery.com/building-a-regression-model-in-pytorch/
https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md

    

'''