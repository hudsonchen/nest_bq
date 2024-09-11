import jax.numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import argparse
import os
import pwd
from utils.kernel_means import *

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

if pwd.getpwuid(os.getuid())[0] == 'hudsonchen':
    os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/nest_bq")
elif pwd.getpwuid(os.getuid())[0] == 'zongchen':
    # os.chdir("/home/zongchen/CBQ")
    os.chdir("/home/zongchen/nest_bq")
    # os.environ[
    #     "XLA_FLAGS"
    # ] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREAD"] = "1"
elif pwd.getpwuid(os.getuid())[0] == 'ucabzc9':
    os.chdir("/home/ucabzc9/Scratch/nest_bq")
else:
    pass

def get_config():
    parser = argparse.ArgumentParser(description='Toy example')
    # Args settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--kernel_x', type=str, default='rbf')
    parser.add_argument('--kernel_theta', type=str, default='rbf')
    parser.add_argument('--save_path', type=str, default='./')
    args = parser.parse_args()
    return args


# def g(x, theta):
#     return jnp.sqrt(2 / jnp.pi) * jnp.exp(-2 * (x - theta) ** 2)

# def f(x):
#     return jnp.log(x)

def f(x):
    return x ** 2

# def g(x, theta):
#     return (1 + jnp.sqrt(3) * jnp.abs(x - theta)) * jnp.exp(- jnp.sqrt(3) * jnp.abs(x - theta))

def g(x, theta):
    return jnp.abs(x - theta) ** 1.5

def simulate_theta(T, rng_key):
    rng_key, _ = jax.random.split(rng_key)
    Theta = jax.random.uniform(rng_key, shape=(T, 1), minval=0., maxval=1.)
    return Theta


def simulate_x_theta(N, Theta, rng_key):
    def simulate_x_per_theta(N, theta, rng_key):
        rng_key, _ = jax.random.split(rng_key)
        x = jax.random.uniform(rng_key, shape=(N, ), minval=0., maxval=1.)
        # x = jax.random.normal(rng_key, shape=(N, ))
        return x
    vmap_func = jax.vmap(simulate_x_per_theta, in_axes=(None, 0, None))
    X = vmap_func(N, Theta, rng_key)
    return X


def run(args, N, T, rng_key):
    # This is a simulation study from Tom rainforth's paper
    # theta ~ U(-1, 1)
    # x ~ U(-1, 1)
    # g(x, theta) = jnp.sqrt(2/pi) exp(-2 (x - theta)^2)
    # f(x) = log(x)
    # I = \E_{theta} f ( \E_{x | theta} [g(x, theta)] )

    rng_key, _ = jax.random.split(rng_key)
    Theta = simulate_theta(T, rng_key)
    X = simulate_x_theta(N, Theta, rng_key)
    g_X = g(X, Theta)

    # This is nested Monte Carlo
    I_theta_MC = g_X.mean(1)
    I_NMC = f(I_theta_MC).mean(0)
    # print(f"Nested Monte Carlo: {I_NMC}")

    # This is nest kernel quadrature
    a, b = 0, 1.
    mu, var = jnp.zeros([N, 1]), jnp.ones([N, 1, 1])
    if args.kernel_x == "rbf":
        I_theta_KQ = KQ_RBF_Uniform_Vectorized(X[:, :, None], g_X, a, b)
        # I_theta_KQ = KQ_RBF_Gaussian_Vectorized(X[:, :, None], g_X, mu, var)
    elif args.kernel_x == "matern":
        I_theta_KQ = KQ_Matern_Uniform_Vectorized(X[:, :, None], g_X, a, b)

    f_I_theta_KQ = f(I_theta_KQ)
    a, b = 0, 1
    if args.kernel_theta == "rbf":
        I_NKQ = KQ_RBF_Uniform(Theta, f_I_theta_KQ, a, b)
    elif args.kernel_theta == "matern":
        I_NKQ = KQ_Matern_Uniform(Theta, f_I_theta_KQ, a, b)
    
    # print(f"Nested kernel quadrature: {I_NKQ}")
    pause = True
    return I_NMC, I_NKQ


def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    # true_value = 0.5 * jnp.log(2 / 5 / jnp.pi) - 2 / 15

    # # Debug code: Use 10000 samples to estimate the true value
    # rng_key, _ = jax.random.split(rng_key)
    # Theta = simulate_theta(10000, rng_key)
    # X = simulate_x_theta(10000, Theta, rng_key)
    # g_X = g(X, Theta)
    # I_theta_MC = g_X.mean(1)
    # I_NMC = f(I_theta_MC).mean(0)
    #
    # true_value_1 = -60 - 37 * jnp.sqrt(3) + 48 * (7 + 4 * jnp.sqrt(3)) * jnp.exp(2 * jnp.sqrt(3))
    # true_value = (1/144) * (192 - 83 * jnp.sqrt(3) + jnp.exp(-4 * jnp.sqrt(3)) * true_value_1)
    true_value = 4/25 *(1/3 + 5 * jnp.pi / 512)

    print(f"True value: {true_value}")
    # N_list = jnp.arange(10, 50, 5).tolist()
    # T_list = jnp.arange(10, 50, 5).tolist()
    N_list = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # N_list = [1000]

    I_NMC_err_dict = {}
    I_NKQ_err_dict = {}

    num_seeds = 1

    for N in N_list:
        T = N
        I_NMC_errors = []
        I_NKQ_errors = []
        
        for s in range(num_seeds):
            rng_key, _ = jax.random.split(rng_key)
            I_NMC, I_NKQ = run(args, N, T, rng_key)
            I_NMC_errors.append(jnp.abs(I_NMC - true_value))
            I_NKQ_errors.append(jnp.abs(I_NKQ - true_value))
        
        I_NMC_err = jnp.median(jnp.array(I_NMC_errors))
        I_NKQ_err = jnp.median(jnp.array(I_NKQ_errors))
        I_NMC_err_dict[(N, T)] = I_NMC_err
        I_NKQ_err_dict[(N, T)] = I_NKQ_err

        methods = ["NKQ", "NMC"]
        errs = [I_NKQ_err, I_NMC_err]

        print(f"T = {T} and N = {N}")
        print("========================================")
        print("Methods:    " + " ".join([f"{method:<10}" for method in methods]))
        print("RMSE:       " + " ".join([f"{value:<10.6f}" for value in errs]))
        print("========================================\n\n")

    with open(f"{args.save_path}/seed_{args.seed}__NKQ", 'wb') as file:
        pickle.dump(I_NKQ_err_dict, file)
    with open(f"{args.save_path}/seed_{args.seed}__NMC", 'wb') as file:
        pickle.dump(I_NMC_err_dict, file)


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/toy/'
    args.save_path += f"kernel_x_{args.kernel_x}__kernel_theta_{args.kernel_theta}"
    os.makedirs(args.save_path, exist_ok=True)
    return args

if __name__ == "__main__":
    args = get_config()
    args = create_dir(args)
    main(args)
    print("========================================")
    print("Finished running")
    print(f"Results saved at {args.save_path}")