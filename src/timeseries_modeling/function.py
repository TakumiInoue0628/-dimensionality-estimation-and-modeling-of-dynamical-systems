import numpy as np

### Activate function
def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def linear(x):
    return x