from utils.kernels import *
import jax
import jax.numpy as jnp


def KQ_RBF_Gaussian(rng_key, X, f_X, mu_X_theta, var_X_theta):
    """
    KQ, not Vectorized.

    \int f(x, theta) N(x|mu_X_theta, var_X_theta) dx

    Args:
        rng_key: random number generator
        X: shape (N, D)
        f_X: shape (N, )
        var_X_theta: (D, D)
        mu_X_theta: (D, )
    Returns:
        I_NKQ: float
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    # l = 1.0
    l = jnp.median(jnp.abs(X - X.mean(0)))  # Median heuristic
    A = 1.0

    K = A * my_RBF(X, X, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_RBF_Gaussian(mu_X_theta, var_X_theta, l, X)
    # varphi = A * kme_double_RBF_Gaussian(mu_X_theta, var_X_theta, l)

    I_NKQ = phi.T @ K_inv @ f_X
    # I_NKQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))
    pause = True
    return I_NKQ


def KQ_RBF_Gaussian_Vectorized(rng_key, X, f_X, mu_X_theta, var_X_theta):
    """
    KQ, Vectorized over the first indice of X, f_X, mu_X_theta and var_X_theta.

    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        f_X: shape (T, N)
        mu_X_theta: (T, D)
        var_X_theta: (T, D, D)
    Returns:
        I_NKQ: (T, )
        I_NKQ_std: (T, )
    """
    vmap_func = jax.vmap(KQ_RBF_Gaussian, in_axes=(None, 0, 0, 0, 0))
    return vmap_func(rng_key, X, f_X, mu_X_theta, var_X_theta)


def KQ_RBF_Uniform(rng_key, X, f_X, a, b):
    """
    KQ, not Vectorized.

    \int f(x, theta) U(x|a,b) dx

    Args:
        rng_key: random number generator
        X: shape (N, D)
        f_X: shape (N, )
        a: float
        b: float
    Returns:
        I_NKQ: float
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    A = 1.
    l = jnp.median(jnp.abs(X - X.mean(0)))  # Median heuristic
    # l = 1.

    K = A * my_RBF(X, X, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_RBF_uniform(a, b, l, X)
    # varphi = A

    I_NKQ = phi.T @ K_inv @ f_X
    # I_NKQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))
    pause = True
    return I_NKQ.squeeze()


def KQ_RBF_Uniform_Vectorized(rng_key, X, f_X, a, b):
    """
    KQ, Vectorized over the first indice of X and f_X.
    a and b are scalars

    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        f_X: shape (T, N)
        a: float
        b: float
    Returns:
        I_NKQ: (T, )
        I_NKQ_std: (T, )
    """
    vmap_func = jax.vmap(KQ_RBF_Uniform, in_axes=(None, 0, 0, None, None))
    return vmap_func(rng_key, X, f_X, a, b)


def KQ_Matern_Gaussian(rng_key, X, f_X):
    """
    KQ, not Vectorized over theta.
    Only works for one-d and for standard normal distribution

    \int f(x) N(x|0,1) dx

    Args:
        rng_key: random number generator
        X: shape (N, D)
        f_X: shape (N, )
        a: float
        b: float
    Returns:
        I_NKQ: float
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    A = 1.
    l = jnp.median(jnp.abs(X - X.mean(0)))  # Median heuristic
    # l = 1.

    K = A * my_Matern(X, X, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_Matern_Gaussian(l, X)
    # varphi = A

    I_NKQ = phi.T @ K_inv @ f_X
    pause = True
    return I_NKQ.squeeze()


def KQ_Matern_Gaussian_Vectorized(rng_key, X, f_X):
    """
    KQ, Vectorized over the first indice of X and f_X.
    Only works for one-d and for standard normal distribution
    Actually, every integral can be written as integration wrt to a standard normal distribution
    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        f_X: shape (T, N)
    Returns:
        I_NKQ: (T, )
        I_NKQ_std: (T, )
    """
    vmap_func = jax.vmap(KQ_Matern_Gaussian, in_axes=(None, 0, 0))
    return vmap_func(rng_key, X, f_X)


def KQ_Matern_Uniform(rng_key, X, f_X, a, b):
    """
    KQ, not Vectorized over theta.
    Only works for one-d, D = 1

    \int f(x) U(x| a , b) dx

    Args:
        rng_key: random number generator
        X: shape (N, D)
        f_X: shape (N, )
        a: float
        b: float
    Returns:
        I_NKQ: float
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    A = 1.
    l = jnp.median(jnp.abs(X - X.mean(0)))  # Median heuristic
    # l = 1.

    K = A * my_Matern(X, X, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_Matern_Uniform(a, b, l, X)
    # varphi = A

    I_NKQ = phi.T @ K_inv @ f_X
    pause = True
    return I_NKQ.squeeze()



def KQ_Matern_Uniform_Vectorized(rng_key, X, f_X, a, b):
    """
    KQ, Vectorized over the first indice of X and f_X.
    Only works for one-d D = 1 and for standard normal distribution

    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        f_X: shape (T, N)
        a: float
        b: float
    Returns:
        I_NKQ: (T, )
        I_NKQ_std: (T, )
    """
    vmap_func = jax.vmap(KQ_Matern_Uniform, in_axes=(None, 0, 0, None, None))
    return vmap_func(rng_key, X, f_X, a, b)
