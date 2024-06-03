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
===============================================================================
Step 3 − Process the data

Let us change the dataset according to our model, so that, we can feed into our 
model. The data can be changed using below code.
'''

x_train_scaled = preprocessing.scale(x_train) 
scaler = preprocessing.StandardScaler().fit(x_train) 
x_test_scaled = scaler.transform(x_test)

'''
Here, we have normalized the training data using sklearn.preprocessing.scale 
function. preprocessing.StandardScaler().fit function returns a scalar with the 
normalized mean and standard deviation of the training data, which we can apply
to the test data using scalar.transform function. This will normalize the test
data as well with the same setting as that of training data.
'''

'''
===============================================================================
Step 4 − Create the model

Let us create the actual model.
'''

model = Sequential() 
model.add(Dense(64, kernel_initializer = 'normal', activation = 'relu',
                input_shape = (13,))) 
model.add(Dense(64, activation = 'relu')) 
model.add(Dense(1))
# This produces a warning!

'''
===============================================================================
Step 5 − Compile the model

Let us compile the model using selected loss function, optimizer and metrics.
'''

model.compile(
   loss = 'mse', 
   optimizer = RMSprop(), 
   metrics = ['mean_absolute_error']
)

'''
===============================================================================
Step 6 − Train the model

Let us train the model using fit() method.
'''

history = model.fit(
   x_train_scaled, y_train,    
   batch_size=128, 
   epochs = 500, 
   verbose = 1, 
   validation_split = 0.2, 
   callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20)]
)

'''
Here, we have used callback function, EarlyStopping. The purpose of this 
callback is to monitor the loss value during each epoch and compare it with 
previous epoch loss value to find the improvement in the training. If there is
no improvement for the patience times, then the whole process will be stopped.

Executing the application will give the below information as output:
    
Train on 323 samples, validate on 81 samples Epoch 1/500
...
Epoch 271/500
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 6.4838 - mean_absolute_error: 
1.8082 - val_loss: 11.5821 - val_mean_absolute_error: 2.3138
...
'''








#==============================================================================
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