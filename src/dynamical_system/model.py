import numpy as np

### Lorenz
def lorenz(t, X, parameter_list):
    dX = np.zeros(X.shape)
    [sigma, rho, beta] = parameter_list
    dX[0] = sigma * (X[1] - X[0])
    dX[1] = X[0] * (rho - X[2]) - X[1]
    dX[2] = X[0] * X[1] - beta * X[2]
    return dX

### Lorenz-96
def lorenz96(t, X, F):
    X_p1 = np.roll(X, -1)
    X_m1 = np.roll(X, 1)
    X_m2 = np.roll(X, 2)
    dX = (X_p1 - X_m2) * X_m1 - X + F
    return dX

def lorenz96_Jacobian(X, parameter_list):
    N = len(X)
    X_p1 = np.roll(X, -1)
    X_m1 = np.roll(X, 1)
    X_m2 = np.roll(X, 2)
    dx1 = np.eye(N)
    dx2 = np.roll(np.diag(X_m1), 1, axis=1)
    dx3 = np.roll(np.diag(X_m1), -2, axis=1)
    dx4 = np.roll(np.diag(X_p1 - X_m2), -1, axis=1)
    J = - dx1 + dx2 - dx3 + dx4
    return J