import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

### Distances to the neighbors of each point
def knn(X, n_neighbors, n_jobs=None):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(X)
    dists, _ = neigh.kneighbors(X)
    return dists

### Maximum Likelihood Intrinsic-Dimension Estimator
def maximum_likelihood_id_estimator(X, k_list=[], unbiased=True):
    m_k_list = []
    for k in tqdm(k_list, desc="Maximum Likelihood ID Estimation", leave=False):
        ### Compute NN(nearest neighbors) distances
        distances = knn(X, k, n_jobs=None)
        ### Euclidean-distances from x to k-th NN
        T_k = distances[:, -1:]
        ### Euclidean-distances from x to 1~(k-1)-th NN
        T_j = distances[:, 1:-1]
        ### Estimated dimension
        if unbiased: m_k = 1 / (np.sum(np.log(T_k / T_j), axis=1) / (k - 2))
        else: m_k = 1 / (np.sum(np.log(T_k / T_j), axis=1) / (k - 1))
        ### Averaging for all samples
        m_k = np.mean(m_k, axis=0)
        m_k_list.append(m_k)
        del distances, T_k, T_j
    return m_k_list