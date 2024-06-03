"""
ToDo: rename thsi file and move into the "Learn" folder

My first attempt with keras. I follow through this tutorial:

https://www.tutorialspoint.com/keras/keras_regression_prediction_using_mpl.htm

I also reproduce the explanatory text from the website in the docstring-type
comments. The #-type comments are my own.
"""

'''
===============================================================================
In this chapter, let us write a simple MPL based ANN to do regression 
prediction. Till now, we have only done the classification based prediction. 
Now, we will try to predict the next possible value by analyzing the previous 
(continuous) values and its influencing factors. The core features of the model 
are as follows:

- Input layer consists of (13,) values.

- First layer, Dense consists of 64 units and ‘relu’ activation function with 
  ‘normal’ kernel initializer.
  
- Second layer, Dense consists of 64 units and ‘relu’ activation function.

- Output layer, Dense consists of 1 unit.

- Use mse as loss function.

- Use RMSprop as Optimizer.

- Use accuracy as metrics.

- Use 128 as batch size.

- Use 500 as epochs.
'''

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

'''
===============================================================================
Step 1 − Import the modules

Let us import the necessary modules.
'''

import keras 
from keras.datasets import boston_housing 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import RMSprop 
from keras.callbacks import EarlyStopping 
from sklearn import preprocessing 
from sklearn.preprocessing import scale

'''
===============================================================================
Step 2 − Load data

Let us import the Boston housing dataset.
'''

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

'''
Here, boston_housing is a dataset provided by Keras. It represents a collection 
of housing information in Boston area, each having 13 features.
'''










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