from os.path import dirname, abspath
import sys
import numpy as np
from tqdm import tqdm
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from util import set_seed
from timeseries_modeling.function import *

class Three_Layer_Feedforward_NN_using_ELM():

    def __init__(self, input_size, hidden_size, output_size,
                 input_weight_scale=0.1, activation_function="sigmoid", seed=0):
        set_seed(seed)
        self.N_x = input_size
        self.N_z = hidden_size
        self.N_y = output_size
        self.sigma = input_weight_scale
        if activation_function=="sigmoid": self.g = sigmoid
        elif activation_function=="tanh": self.g = tanh
        else: self.g = linear
        ### Create W_in
        self.W_in = np.random.uniform(-self.sigma, self.sigma, (self.N_z, self.N_x)).astype(np.float64)
        ### Create I (identity array)
        self. I = np.identity(self.N_z)

    def fit(self, X, Y, ridge_parameter=1e-3):
        X = X.T
        Y = Y.T
        self.alpha = ridge_parameter
        ### Mapping from X (input vector) to Z (hidden vector)
        Z = self.g(self.W_in @ X)
        ### Ridge regression
        self.W_out = Y @ Z.T @ np.linalg.pinv(Z @ Z.T + self.alpha * self.I)

    def predict(self, X):
        X = X.T
        ### Mapping from X (input vector) to Z (hidden vector)
        Z = self.g(self.W_in @ X)
        ### Mapping from Z (hidden vector) to Y (output vector)
        Y = self.W_out @ Z
        return Y.T
    
    def freerun(self, X_init, run_size=100):
        X = np.zeros((self.N_x, run_size+1))
        Z = np.zeros((self.N_z, run_size+1))
        Y = np.zeros((self.N_y, run_size))
        X[:, 0] = X_init.T
        for i in tqdm(range(run_size), desc='Freerun Prediction', leave=False):
            ### Mapping from X (input vector) to Z (hidden vector)
            Z[:, i] = self.g(self.W_in @ X[:, i])
            ### Mapping from Z (hidden vector) to Y (output vector)
            Y[:, i] = self.W_out @ Z[:, i]
            ### x(i+1) <--- y(i)
            X[:, i+1] = Y[:, i]
        return Y.T


