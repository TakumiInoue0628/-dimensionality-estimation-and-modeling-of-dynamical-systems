from os.path import dirname, abspath
import sys
import numpy as np
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from intrinsic_dimension_estimation.function import maximum_likelihood_id_estimator

class Maximum_Likelihood_Estimation_of_ID:

    def __init__(self):
        pass

    def fit(self, X, k_list, standardization=True, unbiased=True):
        if standardization:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            X = (X - self.X_mean) / self.X_std
        return maximum_likelihood_id_estimator(X, k_list, unbiased)


        