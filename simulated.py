import jax.numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp
from kernels import *
from jax.config import config
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)


def f(x, theta):
    return jnp.sqrt(2 / jnp.pi) * jnp.exp(-2 * (x - theta) ** 2)

def g(x):
    return jnp.log(x)

def simulate_theta(T, rng_key):
    Theta = jax.random.uniform(rng_key, shape=(T, 1), minval=-1., maxval=1.)
    return Theta


def simulate_x_theta(N, Theta, rng_key):
    def simulate_x_per_theta(N, theta, rng_key):
        rng_key, _ = jax.random.split(rng_key)
        x = jax.random.normal(rng_key, shape=(N,))
        return x
    vmap_func = jax.vmap(simulate_x_per_theta, in_axes=(None, 0, None))
    X = vmap_func(N, Theta, rng_key)
    return X


def BQ_RBF_Gaussian(rng_key, X, f_X, mu_X_theta, var_X_theta):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    The kernel_x is RBF, and the hyperparameters are selected by minimizing the negative log-likelihood (NLL).
    Not vectorized over theta.

    Args:
        rng_key: random number generator
        X: shape (N, D)
        f_X: shape (N, )
        var_X_theta: (D, D)
        mu_X_theta: (D, )
    Returns:
        I_BQ_mean: float
        I_BQ_std: float
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    l = 1.0
    A = 1.0

    K = A * my_RBF(X, X, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_RBF_Gaussian(mu_X_theta, var_X_theta, l, X)
    varphi = A * kme_double_RBF_Gaussian(mu_X_theta, var_X_theta, l)

    I_BQ_mean = phi.T @ K_inv @ f_X
    I_BQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))
    pause = True
    return I_BQ_mean, I_BQ_std


def BQ_RBF_Gaussian_vectorized_on_T(rng_key, X, f_X, mu_X_theta, var_X_theta):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Vectorized over Theta.

    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        f_X: shape (T, N)
        mu_X_theta: (T, D)
        var_X_theta: (T, D, D)
    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """
    vmap_func = jax.vmap(BQ_RBF_Gaussian, in_axes=(None, 0, 0, 0, 0))
    return vmap_func(rng_key, X, f_X, mu_X_theta, var_X_theta)


def BQ_RBF_uniform(rng_key, X, f_X, a, b):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    The kernel_x is RBF, and the hyperparameters are selected by minimizing the negative log-likelihood (NLL).
    Not vectorized over theta.

    Args:
        rng_key: random number generator
        X: shape (N, D)
        f_X: shape (N, )
        a: float
        b: float
    Returns:
        I_BQ_mean: float
        I_BQ_std: float
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    A = 1.
    l = 1.

    K = A * my_RBF(X, X, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_RBF_uniform(a, b, l, X)
    varphi = A

    I_BQ_mean = phi.T @ K_inv @ f_X
    I_BQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))
    pause = True
    return I_BQ_mean.squeeze(), I_BQ_std


def BQ_RBF_uniform_vectorized_on_T(rng_key, X, f_X, a, b):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Vectorized over Theta.

    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        f_X: shape (T, N)
        a: float
        b: float
    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """
    vmap_func = jax.vmap(BQ_RBF_uniform, in_axes=(None, 0, 0, None, None))
    return vmap_func(rng_key, X, f_X, a, b)


def run(N, T, rng_key):
    # This is a simulation study from Tom rainforth's paper
    # theta ~ U(-1, 1)
    # x ~ N(0, 1)
    # f(x, theta) = jnp.sqrt(2/pi) exp(-2 (x - theta)^2)
    # g(x) = log(x)
    # I = \E_{theta} g ( \E_{x} [f(x, theta)] )

    rng_key, _ = jax.random.split(rng_key)
    Theta = simulate_theta(T, rng_key)
    X = simulate_x_theta(N, Theta, rng_key)
    f_X = f(X, Theta)

    # This is nested Monte Carlo
    I_theta_MC = f_X.mean(1)
    I_MC = g(I_theta_MC).mean(0)
    # print(f"Nested Monte Carlo: {I_MC}")

    # This is nest Bayesian quadrature
    mu = jnp.zeros([T, 1])
    var = jnp.ones([T, 1, 1])
    I_theta_BQ, _ = BQ_RBF_Gaussian_vectorized_on_T(rng_key, X[:, :, None], f_X, mu, var)
    g_I_theta_BQ = g(I_theta_BQ)
    a, b = 0, 1
    I_BQ, _ = BQ_RBF_uniform(rng_key, I_theta_BQ[:, None], g_I_theta_BQ, a, b)
    # print(f"Nested Bayesian quadrature: {I_BQ}")
    pause = True
    return I_MC, I_BQ


def main():
    rng_key = jax.random.PRNGKey(int(time.time()))
    true_value = 0.5 * jnp.log(2 / 5 / jnp.pi) - 2 / 15
    print(f"True value: {true_value}")
    N_list = jnp.arange(10, 50, 5).tolist()
    T_list = jnp.arange(10, 50, 5).tolist()

    I_MC_err_dict = {}
    I_BQ_err_dict = {}
    num_seeds = 10

    rng_key = jax.random.PRNGKey(0)

    for N in N_list:
        for T in T_list:
            I_MC_errors = []
            I_BQ_errors = []
            for seed in tqdm(range(num_seeds)):
                rng_key, _ = jax.random.split(rng_key)
                I_MC, I_BQ = run(N, T, rng_key)
                I_MC_errors.append(jnp.abs(I_MC - true_value))
                I_BQ_errors.append(jnp.abs(I_BQ - true_value))
            I_MC_err = jnp.median(jnp.array(I_MC_errors))
            I_BQ_err = jnp.median(jnp.array(I_BQ_errors))
            I_MC_err_dict[(N, T)] = I_MC_err
            I_BQ_err_dict[(N, T)] = I_BQ_err

    # Plotting code
    fig1, axs1 = plt.subplots(len(N_list), 1, figsize=(6, 4*len(N_list)))
    for i, N in enumerate(N_list):
        T_values = []
        I_MC_err_values = []
        I_BQ_err_values = []
        for T in T_list:
            T_values.append(T)
            I_MC_err_values.append(I_MC_err_dict[(N, T)])
            I_BQ_err_values.append(I_BQ_err_dict[(N, T)])
        axs1[i].plot(T_values, I_MC_err_values, label=f'I_MC, N={N}')
        axs1[i].plot(T_values, I_BQ_err_values, label=f'I_BQ, N={N}')
        axs1[i].set_xlabel('T')
        axs1[i].set_ylabel('Absolute Error')
        axs1[i].legend()
        axs1[i].set_title(f'N={N}')
    fig1.savefig('fix_N_plot_T.png')

    fig2, axs2 = plt.subplots(len(T_list), 1, figsize=(6, 4*len(T_list)))
    for i, T in enumerate(T_list):
        N_values = []
        I_MC_err_values = []
        I_BQ_err_values = []
        for N in N_list:
            N_values.append(N)
            I_MC_err_values.append(I_MC_err_dict[(N, T)])
            I_BQ_err_values.append(I_BQ_err_dict[(N, T)])
        axs2[i].plot(N_values, I_MC_err_values, label=f'I_MC, T={T}')
        axs2[i].plot(N_values, I_BQ_err_values, label=f'I_BQ, T={T}')
        axs2[i].set_xlabel('N')
        axs2[i].set_ylabel('Absolute Error')
        axs2[i].legend()
        axs2[i].set_title(f'T={T}')
    fig2.savefig('fix_T_plot_N.png')


if __name__ == "__main__":
    main()