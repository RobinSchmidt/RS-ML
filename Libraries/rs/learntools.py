'''
Convenience functions to deal with Python's machine learning libraries sich as
sklearn, etc ...TBC...
'''

import numpy as np 
#from sklearn.neural_network import MLPRegressor
from .datatools import delayed_values


def synthesize_skl_mlp(mlp, d, init_sect, length):
    
    # Initialization:
    q  = np.zeros(length)     # Signal that we want to generate
    Li = len(init_sect)       # Length of given initial section
    q[0:Li] = init_sect       # Copy initial section into our result
    
    # Recursive prediction of the values q[n] for n >= Li:
    for n in range(Li, length):
        xn = delayed_values(q, n, d)
        yn = mlp.predict(xn.reshape(1, -1))
        q[n] = yn[0]
    return q

# ToDo:
#
# - We should somehow communicate that the mlp input parameter is supposed to 
#   be of type MLPRegressor from sklearn.neural_network - at least in the
#   documentation but preferably also in the code. Maybe use type hints. See:
#   https://docs.python.org/3/library/typing.html    
#   https://stackoverflow.com/questions/2489669/how-do-python-functions-handle-the-types-of-parameters-that-you-pass-in