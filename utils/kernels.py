import jax.numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp
from functools import partial
from jax.scipy.stats import norm
from jax.scipy.special import erf, erfc
import time


@jax.jit
def my_Matern_32(x, y, l):
    """
    Matern three halves kernel.

    Args:
        x: (N, )
        y: (M, )
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    K = kernel.matrix(x[:, None], y[:, None])
    return K


@jax.jit
def my_Matern_12(x, y, l):
    """
    Matern three halves kernel.

    Args:
        x: (N, )
        y: (M, )
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    kernel = tfp.math.psd_kernels.MaternOneHalf(amplitude=1., length_scale=l)
    K = kernel.matrix(x[:, None], y[:, None])
    return K


@jax.jit
def my_Matern_12_product(x, y, l):
    """
    Product Matern three halves kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: (D, )

    Returns:
        kernel matrix: (N, M)
    """
    high_d_map = jax.vmap(my_Matern_12, in_axes=(0, 0, 0))
    K_all_d = high_d_map(x.T, y.T, l)
    return jnp.mean(K_all_d, axis=0)


@jax.jit
def my_Matern_32_product(x, y, l):
    """
    Product Matern three halves kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: (D, )

    Returns:
        kernel matrix: (N, M)
    """
    high_d_map = jax.vmap(my_Matern_32, in_axes=(0, 0, 0))
    K_all_d = high_d_map(x.T, y.T, l)
    return jnp.prod(K_all_d, axis=0)

@jax.jit
def my_RBF(x, y, l):
    """
    RBF kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=l)
    K = kernel.matrix(x, y)
    return K


def jax_dist(x, y):
    return jnp.sqrt(((x - y) ** 2).sum(-1)).squeeze()


distance = jax.vmap(jax_dist, in_axes=(None, 0), out_axes=1)
sign_func = jax.vmap(jnp.greater, in_axes=(None, 0), out_axes=1)


def my_Laplace(x, y, l):
    """
    Laplace kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    r = distance(x, y).squeeze()
    return jnp.exp(- r / l)


def dx_Laplace(x, y, l):
    sign = sign_func(x, y).squeeze().astype(float) * 2 - 1
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * (-sign)
    return part1


def dy_Laplace(x, y, l):
    sign = sign_func(x, y).squeeze().astype(float) * 2 - 1
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * sign
    return part1


def dxdy_Laplace(x, y, l):
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * (-1)
    return part1


@jax.jit
def kme_Matern_32_Gaussian(l, y):
    """
    The implementation of the kernel mean embedding of the Matern three halves kernel with Gaussian distribution
    Only in one dimension, and the Gaussian distribution is N(0, 1)
    
    Args:
        y: (M, D)
        l: scalar

    Returns:
        kernel mean embedding: (M, )
    """
    E10 = 1 - jnp.sqrt(3) * y / l
    E11 = jnp.sqrt(3) / l
    muA = -jnp.sqrt(3) / l

    part11 = jnp.exp((3 + 2 * jnp.sqrt(3) * y * l) / (2 * l ** 2))
    part12 = (E10 + E11 * muA) * norm.cdf(muA - y)
    part13 = E11 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(y - muA) ** 2 / 2)
    part1 = part11 * (part12 + part13)

    E20 = 1 + jnp.sqrt(3) * y / l
    E21 = jnp.sqrt(3) / l
    muB = jnp.sqrt(3) / l

    part21 = jnp.exp((3 - 2 * jnp.sqrt(3) * y * l) / (2 * l ** 2))
    part22 = (E20 - E21 * muB) * norm.cdf(y - muB)
    part23 = E21 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(y - muB) ** 2 / 2)
    part2 = part21 * (part22 + part23)

    final = part1 + part2
    pause = True
    return final

@jax.jit
def kme_Matern_32_Uniform_1d(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the Matern three halves kernel with Uniform distribution U[a,b]
    Only in one dimension, D = 1
    
    Args:
        y: (M, )
        l: scalar

    Returns:
        kernel mean embedding: (M, )
    """
    r = b - a
    sqrt_3 = jnp.sqrt(3)
    term1_1 = (2 * l) / sqrt_3
    term1_2 = (3 * b + 2 * sqrt_3 * l - 3 * y) * jnp.exp(sqrt_3 * (y - b) / l) / 3
    term1 = (term1_1 - term1_2) / r

    term2_1 = (2 * l) / sqrt_3
    term2_2 = (-3 * a + 2 * sqrt_3 * l + 3 * y) * jnp.exp(sqrt_3 * (a - y) / l) / 3
    term2 = (term2_1 - term2_2) / r

    kme = term1 + term2
    return kme


@jax.jit
def kme_Matern_32_Uniform(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the Matern three halves kernel with Uniform distribution U[a,b]
    Only works for product Matern kernel
    
    Args:
        a: (D, )
        b: (D, )
        l: (D, )
        y: (M, D)

    Returns:
        kernel mean embedding: (M, )
    """
    high_d_map = jax.vmap(kme_Matern_32_Uniform_1d, in_axes=(0, 0, 0, 0))
    kme_all_d = high_d_map(a, b, l, y.T)
    return jnp.prod(kme_all_d, axis=0)

@jax.jit
def kme_Matern_12_Gaussian_1d(l, y):
    """
    The implementation of the kernel mean embedding of the Matern one half kernel with Gaussian distribution
    Only in one dimension, and the Gaussian distribution is N(0, 1)
    
    Args:
        y: (M, )
        l: scalar

    Returns:
        kernel mean embedding: (M, )
    """
    part1 = jnp.exp((1 - 2 * l * y) / (2 * l ** 2)) * (1 + erf((-1 + l * y) / (jnp.sqrt(2) * l)))
    part2 = jnp.exp((1 + 2 * l * y) / (2 * l ** 2)) * erfc((1 / l + y) / jnp.sqrt(2))

    return (part1 + part2) / 2

@jax.jit
def kme_Matern_12_Gaussian(l, y):
    """
    The implementation of the kernel mean embedding of the Matern one half kernel with Gaussian distribution
    Only in one dimension, and the Gaussian distribution is N(0, 1)
    
    Args:
        y: (M, D)
        l: (D, )

    Returns:
        kernel mean embedding: (M, )
    """
    high_d_map = jax.vmap(kme_Matern_12_Gaussian_1d, in_axes=(0, 0))
    kme_all_d = high_d_map(l, y.T)
    return jnp.mean(kme_all_d, axis=0)

@jax.jit
def kme_Matern_12_Uniform_1d(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the Matern one half kernel with Uniform distribution U[a,b]
    Only in one dimension, D = 1
    
    Args:
        y: (M, )
        l: scalar

    Returns:
        kernel mean embedding: (M, )
    """
    r = b - a
    term1 = (l - jnp.exp((a - y) / l) * l) / r
    term2 = (l - jnp.exp((y - b) / l) * l) / r
    kme = term1 + term2
    return kme


@jax.jit
def kme_Matern_12_Uniform(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the Matern one half kernel with Uniform distribution U[a,b]
    Only works for product Matern kernel
    
    Args:
        a: (D, )
        b: (D, )
        l: (D, )
        y: (M, D)

    Returns:
        kernel mean embedding: (M, )
    """
    high_d_map = jax.vmap(kme_Matern_12_Uniform_1d, in_axes=(0, 0, 0, 0))
    kme_all_d = high_d_map(a, b, l, y.T)
    return jnp.prod(kme_all_d, axis=0)

@jax.jit
def kme_RBF_Gaussian(mu, Sigma, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution
    A fully vectorized implementation.

    Args:
        mu: Gaussian mean, (D, )
        Sigma: Gaussian covariance, (D, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel mean embedding: (M, )
    """
    kme_RBF_Gaussian_func_ = partial(kme_RBF_Gaussian_func, mu, Sigma, l)
    kme_RBF_Gaussian_vmap_func = jax.vmap(kme_RBF_Gaussian_func_)
    return kme_RBF_Gaussian_vmap_func(y)


@jax.jit
def kme_RBF_Gaussian_func(mu, Sigma, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution.
    Not vectorized.

    Args:
        mu: Gaussian mean, (D, )
        Sigma: Gaussian covariance, (D, D)
        y: (D, )
        l: float

    Returns:
        kernel mean embedding: scalar
    """
    D = mu.shape[0]
    l_ = l ** 2
    Lambda = jnp.eye(D) * l_
    Lambda_inv = jnp.eye(D) / l_
    part1 = jnp.linalg.det(jnp.eye(D) + Sigma @ Lambda_inv)
    part2 = jnp.exp(-0.5 * (mu - y).T @ jnp.linalg.inv(Lambda + Sigma) @ (mu - y))
    return part1 ** (-0.5) * part2


@jax.jit
def kme_RBF_uniform_func(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Uniform distribution.
    Not vectorized.

    Args:
        a: float (lower bound)
        b: float (upper bound)
        l: float
        y: float

    Returns:
        kernel mean embedding: scalar
    """
    part1 = jnp.sqrt(jnp.pi / 2) * l / (b - a)
    part2 = jax.scipy.special.erf((b - y) / (l * jnp.sqrt(2))) - jax.scipy.special.erf((a - y) / (l * jnp.sqrt(2)))
    return part1 * part2

@jax.jit
def kme_RBF_uniform(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution
    A fully vectorized implementation.

    Args:
        a: float (lower bound)
        b: float (upper bound)
        l: float
        y: (M, D)

    Returns:
        kernel mean embedding: (M, )
    """
    kme_RBF_uniform_func_ = partial(kme_RBF_uniform_func, a, b, l)
    kme_RBF_uniform_vmap_func = jax.vmap(kme_RBF_uniform_func_)
    kme_all_d = kme_RBF_uniform_vmap_func(y)
    return jnp.prod(kme_all_d, axis=1)


@jax.jit
def kme_double_RBF_Gaussian(mu, Sigma, l):
    """
    The implementation of the initial of the RBF kernel with Gaussian distribution.

    Args:
        mu: Gaussian mean, (D, )
        Sigma: Gaussian covariance, (D, D)
        l: scalar

    Returns:
        initial error: scalar
    """
    l_ = l ** 2
    D = mu.shape[0]
    Lambda = jnp.eye(D) * l_
    Lambda_inv = jnp.eye(D) / l_
    part1 = jnp.linalg.det(jnp.eye(D) + Sigma @ Lambda_inv)
    part2 = jnp.linalg.det(jnp.eye(D) + Sigma @ jnp.linalg.inv(Lambda + Sigma))
    return part1 ** (-0.5) * part2 ** (-0.5)


@jax.jit
def my_log_RBF(x, y, l):
    """
    Log normal RBF kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    return my_RBF(jnp.log(x), jnp.log(y), l)


@jax.jit
def kme_log_normal_log_RBF(mu, std, y, l):
    """
    The implementation of the kernel mean embedding of the log RBF kernel with log normal distribution.

    Args:
        y: (M, D)
        l: kernel bandwidth scalar
        mu: mean for the log normal distribution, scalar
        std: std for the log normal distribution, scalar
        

    Returns:
        kernel mean embedding: (M, )
    """
    part1 = jnp.exp(-(mu ** 2 + jnp.log(y) ** 2) / (2 * (std ** 2 + l ** 2)))
    part2 = jnp.power(y, mu / (std ** 2 + l ** 2))
    part3 = std * jnp.sqrt(std ** (-2) + l ** (-2))
    return part1 * part2 / part3


def main():
    seed = 0
    rng_key = jax.random.PRNGKey(seed)
    # Test RBF uniform kernel
    N = 1000
    x = jax.random.uniform(rng_key, shape=(N, 1))
    l = 0.5
    K = my_RBF(x, x, 0.5)
    print(f"Empirical kernel mean: {K.mean(0)[:10]}")
    kme = kme_RBF_uniform(0., 1., l, x)
    print(f"Analytic kernel mean: {kme.flatten()[:10]}")

    # x = jax.random.uniform(rng_key, shape=(3, 2))
    # rng_key, _ = jax.random.split(rng_key)
    # y = jax.random.uniform(rng_key, shape=(3, 2))
    # l = 0.5
    # batch_kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=.5)
    # K1 = batch_kernel.matrix(x, y)
    # K2 = my_Matern(x, y, 0.5)
    # print(K1)
    # print(K2)

    # print("============")
    # batch_kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=.5)
    # grad_x_K_fn = jax.grad(batch_kernel.apply, argnums=(0,))
    # vec_grad_x_K_fn = jax.vmap(grad_x_K_fn, in_axes=(0, 0), out_axes=1)
    # grad_y_K_fn = jax.grad(batch_kernel.apply, argnums=(1,))
    # vec_grad_y_K_fn = jax.vmap(grad_y_K_fn, in_axes=(0, 0), out_axes=1)

    # seed = 0
    # rng_key = jax.random.PRNGKey(seed)
    # N = 2
    # D = 3
    # l = 0.5

    # rng_key = jax.random.PRNGKey(seed)
    # x = jax.random.uniform(rng_key, shape=(N, D))
    # rng_key, _ = jax.random.split(rng_key)
    # y = jax.random.uniform(rng_key, shape=(N, D))

    # x_dummy = jnp.stack((x, x), axis=0).reshape(N * N, D)
    # y_dummy = jnp.stack((y, y), axis=1).reshape(N * N, D)

    # dx_K = vec_grad_x_K_fn(x_dummy, y_dummy)[0].reshape(N, N, D)
    # dy_K = vec_grad_y_K_fn(x_dummy, y_dummy)[0].reshape(N, N, D)

    # print(dx_K)
    # print(dy_K)

    # print(grad_x_K_fn(x[0, :], y[0, :]))
    # print(grad_x_K_fn(x[0, :], y[1, :]))
    # print(grad_x_K_fn(x[1, :], y[0, :]))
    # print(grad_x_K_fn(x[1, :], y[1, :]))

    # print(grad_y_K_fn(x[0, :], y[0, :]))
    # print(grad_y_K_fn(x[0, :], y[1, :]))
    # print(grad_y_K_fn(x[1, :], y[0, :]))
    # print(grad_y_K_fn(x[1, :], y[1, :]))


if __name__ == '__main__':
    main()