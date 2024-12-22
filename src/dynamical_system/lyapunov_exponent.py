from os.path import dirname, abspath
import sys
import numpy as np
from tqdm import tqdm
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)

def shimada_nagashima_algorithm(Jacobian, X, dt, parameter_list=[], qr_step=1):
    Q = np.eye(len(X[0]))
    r = 0   
    for i, x in tqdm(enumerate(X), total=len(X), desc="Estimation", leave=False):
        J = Jacobian(x, parameter_list)
        k1 = np.dot(J, Q)
        k2 = np.dot(J, Q + 0.5 * dt * k1)
        k3 = np.dot(J, Q + 0.5 * dt * k2)
        k4 = np.dot(J, Q + dt * k3)
        Q = Q + dt/6*(k1 + 2 * k2 + 2 * k3 + k4)
        if i % qr_step == 0:
            Q, R = np.linalg.qr(Q)        
            r += np.log(np.abs(np.diag(R)))   
    return r / ((i+1) * dt)

def fractal_dim_from_lyapunov(lyapunov_exponent):
    return np.argmin(np.where(np.cumsum(lyapunov_exponent)>0, np.cumsum(lyapunov_exponent), 0))