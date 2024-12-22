from os.path import dirname, abspath
import sys
import numpy as np
from tqdm import tqdm
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
### Import module
from util import set_seed
from dynamical_system.function import runge_kutta_4

def simulate_rk4(system_model, 
                 X_init,
                 model_params, 
                 params={'start time':0.,
                         'ending time':50.,
                         'time step size':1e-2,},
                 seed=0,):
    set_seed(seed)
    dt = params['time step size']
    T = np.arange(params['start time'], params['ending time']+dt, dt)
    X = np.zeros((T.shape[0], len(X_init)))
    X[0] = X_init
    ### Simulation
    for k in tqdm(range(T.shape[0]-1), desc="Simulation", leave=False):
        X[k+1] = runge_kutta_4(T[k], X[k], system_model, model_params, dt)
    return X, T[:X.shape[0]]