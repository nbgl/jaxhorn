import jax
import jax.numpy as np
from jax.scipy.special import logsumexp


def _sinkhorn_iteration(K, a, b, u, v):
    v = b / (K.T @ u)
    u = a / (K @ v)
    return u, v


def _sinkhorn_until_convergence(K, a, b, tol):
    init_u = np.full_like(a, 1. / a.shape[-1])
    init_v = np.full_like(b, 1. / b.shape[-1])

    def cond(arg):
        u, v = arg
        right_marginal = np.einsum('i,ij,j->j', u, K, v)
        distsq = (right_marginal @ right_marginal
                  + b @ b
                  - 2 * right_marginal @ b)
        return distsq > tol * tol * b.shape[-1]

    def body(arg):
        u, v = arg
        new_u, new_v = _sinkhorn_iteration(K, a, b, u, v)
        return new_u, new_v

    found_u, found_v = jax.lax.while_loop(cond, body, (init_u, init_v))
    return found_u, found_v


def sinkhorn(M, a, b, reg, tol=1e-6, grad_iters=10):
    K = np.exp(M / reg)
    u, v = jax.lax.stop_gradient(_sinkhorn_until_convergence(K, a, b, tol))
    for _ in range(grad_iters):
        u, v = _sinkhorn_iteration(K, a, b, u, v)
    return K, u, v


def _logspace_vvp(v1, v2):
    assert len(v1.shape) == 1
    assert v1.shape == v2.shape
    return logsumexp(v1 + v2)


def _logspace_mvp(m, v):
    assert len(m.shape) == 2
    assert len(v.shape) == 1
    assert m.shape[1] == v.shape[0]
    return jax.vmap(_logspace_vvp, in_axes=(0, None), out_axes=0)(m, v)


def _sinkhorn_iteration_log(log_K, log_a, log_b, log_u, log_v):
    log_v = log_b - _logspace_mvp(log_K.T, log_u)
    log_u = log_a - _logspace_mvp(log_K, log_v)
    return log_u, log_v


def _sinkhorn_until_convergence_log(log_K, log_a, log_b, tol):
    init_log_u = np.full_like(log_a, -np.log(log_a.shape[-1]))
    init_log_v = np.full_like(log_b, -np.log(log_b.shape[-1]))
    b = np.exp(log_b)

    def cond(arg):
        log_u, log_v = arg
        right_marginal = np.exp(_logspace_mvp(log_K.T, log_u) + log_v)
        distsq = (right_marginal @ right_marginal
                  + b @ b
                  - 2 * right_marginal @ b)
        return distsq > tol * tol * b.shape[-1]

    def body(arg):
        log_u, log_v = arg
        new_log_u, new_log_v = _sinkhorn_iteration_log(
            log_K, log_a, log_b, log_u, log_v)
        return new_log_u, new_log_v

    found_log_u, found_log_v = jax.lax.while_loop(
        cond, body, (init_log_u, init_log_v))
    return found_log_u, found_log_v


def sinkhorn_log(M, log_a, log_b, reg, tol=1e-6, grad_iters=1000):
    log_K = M / reg
    log_u, log_v = jax.lax.stop_gradient(_sinkhorn_until_convergence_log(
        log_K, log_a, log_b, tol))
    for _ in range(grad_iters):
        log_u, log_v = _sinkhorn_iteration_log(log_K, log_a, log_b, log_u, log_v)
    return log_K, log_u, log_v
