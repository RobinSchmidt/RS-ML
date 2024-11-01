'''
Convenience functions to deal with Python's machine learning libraries sich as
sklearn, etc ...TBC...
'''

import numpy as np 
from .datatools import delayed_values


def synthesize_skl_mlp(mlp, d, init_sect, length):
    
    # Initialization:
    s  = np.zeros(length)                  # Signal that we want to generate
    Li = len(init_sect)                    # Length of given initial section
    s[0:Li] = init_sect                    # Copy initial section into result
    
    # Recursive prediction of the values q[n] for n >= Li:
    for n in range(Li, length):
        X = delayed_values(s, n, d)
        y = mlp.predict(X.reshape(1, -1))  # reshape: 1D -> 2D 
        s[n] = y[0]                        # [0]:     1D -> 0D (scalar)
    return s

# ToDo:
#
# - We should somehow communicate that the mlp input parameter is supposed to 
#   be of type MLPRegressor from sklearn.neural_network - at least in the
#   documentation but preferably also in the code. Maybe use type hints. See:
#   https://docs.python.org/3/library/typing.html    
#   https://stackoverflow.com/questions/2489669/how-do-python-functions-handle-the-types-of-parameters-that-you-pass-in
#   ...but: maybe we could also use other types of models with the same API. 
#   Maybe we should rename the "mlp" parameter to "model". See also:
#   https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
#   ...ok - we now have a dispatch between two different model types
#  
# Notes:
#    
# - The _skl_ in the function name is supposed to indicate that the function is
#   using scikit-learn. We may later have similar functions for models created
#   with other ML frameworks (such as TensorFlow, PyTorch or Keras). For these,
#   maybe use _tf_, _pt_, _krs_ instead. I think, we do not need to distinguish
#   between MLPRegressor and MLPClassifier - the "synthesize" already implies
#   that we are dealing with regression. Maybe for classification, use classify
#   instead
#
# - Hmm - the function also seems to work with a keras model. Apparently, the
#   predict method of keras and sklearn have the same API? Figure out!



def synthesize_keras_mlp(mlp, d, init_sect, length):
    
    # Initialization:
    s  = np.zeros(length)                  # Signal that we want to generate
    Li = len(init_sect)                    # Length of given initial section
    s[0:Li] = init_sect                    # Copy initial section into result
    
    # Recursive prediction of the values q[n] for n >= Li:
    for n in range(Li, length):
        X = delayed_values(s, n, d)
        y = mlp(X.reshape(1, -1))          # reshape: 1D -> 2D 
        s[n] = y                        
    return s

# Notes
#
# - This function has a lot of code duplication from synthesize_skl_mlp. I 
#   tried to refactor it to factor out a "predict" function as free function
#   that can take both types of models and dispatches based on isinstance(). 
#   But that caused more problems than it solves, so for the time being, I 
#   accept that duplication. The function is currently in the file Attic.txt 
#   just in case...but I need to get more experienced with Python before I can 
#   really make an informed decision what the best way to deal with this is.





