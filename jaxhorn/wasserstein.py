import numpy as np

from sinkhorn import sinkhorn, sinkhorn_log


def w2_euclidean(a, b, a_weights, b_weights, reg, tol=1e-6, grad_iters=10):
    dists = -2 * a @ b.T
    dists = dists + np.einsum('ij,ij->i', a, a)[...,None]
    dists = dists + np.einsum('ij,ij->i', b, b)[...,None,:]
    K, u, v = sinkhorn(
        -dists, a_weights, b_weights, reg,
        tol=tol, grad_iters=grad_iters)
    return np.sqrt(np.einsum(
        'ij,ij,i,j->', K, dists, u, v))

def w2_euclidean_log(a, b, a_log_weights, b_log_weights, reg, tol=1e-6, grad_iters=10):
    dists = -2 * a @ b.T
    dists = dists + np.einsum('ij,ij->i', a, a)[...,None]
    dists = dists + np.einsum('ij,ij->i', b, b)[...,None,:]
    log_K, log_u, log_v = sinkhorn_log(
        -dists, a_log_weights, b_log_weights, reg,
        tol=tol, grad_iters=grad_iters)
    P = np.exp(log_K + log_u[...,None] + log_v[...,None,:])
    return np.sqrt(np.einsum('ij,ij', P, dists))


def main():
    import numpy as onp
    import scipy.stats
    means = 3., 7.
    vars_ = 7., 3.
    a = onp.random.uniform(-10, 20, size=99)
    b = onp.random.uniform(-10, 20, size=101)
    a_weights = scipy.stats.norm.pdf(a, means[0], onp.sqrt(vars_[0]))
    b_weights = scipy.stats.norm.pdf(b, means[1], onp.sqrt(vars_[1]))
    a = a.reshape((99,1))
    b = b.reshape((101,1))
    a_weights /= a_weights.sum()
    b_weights /= b_weights.sum()
    reg = 10.
    print(w2_euclidean(a, b, a_weights, b_weights, reg))


def main_log():
    from jax.scipy.special import logsumexp
    import numpy as onp
    import scipy.stats
    means = 3., 7.
    vars_ = 7., 3.
    a = scipy.stats.norm(means[0], onp.sqrt(vars_[0])).rvs(size=99)
    b = scipy.stats.norm(means[1], onp.sqrt(vars_[1])).rvs(size=101)
    a_weights = np.zeros_like(a)
    b_weights = np.zeros_like(b)
    a = a.reshape((99,1))
    b = b.reshape((101,1))
    a_weights -= logsumexp(a_weights)
    b_weights -= logsumexp(b_weights)
    reg = 10.
    print(w2_euclidean_log(a, b, a_weights, b_weights, reg))

if __name__ == '__main__':
    main()
    main_log()
