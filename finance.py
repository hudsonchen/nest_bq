import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import scipy
import jax.numpy as jnp
from tqdm import tqdm
import pickle
from utils.kernel_means import *
import os
import pwd
import shutil
import argparse
from jax import config
import warnings

warnings.filterwarnings("ignore", message="The balance properties of Sobol' points require n to be a power of 2.")
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

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

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def get_config():
    parser = argparse.ArgumentParser(description='Nested Kernel Quadrature for Financial Risk Management')

    # Data settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--multi_level', action='store_true', default=False)
    parser.add_argument('--qmc', action='store_true', default=False)
    parser.add_argument('--eps', type=float, default=0.01)
    args = parser.parse_args()
    return args

# Hyperparameters for the Black Scholes model
s = 0.2
sigma = 0.3
T_finance = 2
t_finance = 1
S0 = 100
K1 = 50
K2 = 150

def price(St, epsilon):
    """
    Computes the price ST at time T_finance.
    ST is sampled from the conditional distribution p(ST|St).
    Computes the loss \psi(ST) - \psi((1+s)ST) caused by the shock. 
    Their shape is T * N

    Args:
        St: (T, 1) the price at time t_finance        
    Returns:
        ST: (T, N, 1)
        f(ST): (T, N)
    """
    ST = St * jnp.exp(sigma * jnp.sqrt((T_finance - t_finance)) * epsilon - 0.5 * (sigma ** 2) * (T_finance - t_finance))
    psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
    psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * jnp.maximum(
        (1 + s) * ST - (K1 + K2) / 2, 0)
    return psi_ST_1 - psi_ST_2


def nested_monte_carlo(Theta, u_theta, x, u_x):
    f_val = price(Theta, x)
    I_MC = f_val.mean(1)
    I_NMC = jnp.maximum(I_MC, 0).mean(0)
    return I_NMC


def nested_kernel_quadrature(Theta, u_theta, x, u_x):
    T, N = x.shape[0], x.shape[1]
    f_val = price(Theta, x)
    lengthscale = 1.0    
    # This is nest kernel quadrature for the inner expectation
    scale, shift = f_val.std(1), f_val.mean(1)
    f_val_normalized = (f_val - shift[:, None]) / scale[:, None]
    f_val_normalized = jnp.nan_to_num(f_val_normalized, 0.)
    lmbda = 0.001 * N ** (-1)
    if T > 100:
        for t in range(T):
            I_KQ_ = KQ_RBF_Gaussian(x[t, :, None], f_val_normalized[t], jnp.zeros([1]), jnp.ones([1, 1]), lengthscale, lmbda)
            if t == 0:
                I_KQ = I_KQ_
            else:
                I_KQ = jnp.vstack([I_KQ, I_KQ_])
        I_KQ = I_KQ.squeeze()
    else:
        I_KQ = KQ_RBF_Gaussian_Vectorized(x[:, :, None], f_val_normalized, jnp.zeros([T, 1]), jnp.ones([T, 1]), lengthscale, lmbda)
    # I_KQ = KQ_Matern_12_Uniform_Vectorized(u_x[:, :, None], f_val_normalized, jnp.zeros([N, 1]), jnp.ones([N, 1]), scale)
    I_KQ = I_KQ * scale + shift
    f_I_KQ = jnp.maximum(I_KQ, 0)

    # This is nest kernel quadrature for the outer expectation
    scale, shift = f_I_KQ.std(), f_I_KQ.mean()
    f_I_KQ_normalized = (f_I_KQ - shift) / scale
    f_I_KQ_normalized = jnp.nan_to_num(f_I_KQ_normalized, 0.)
    # I_NKQ = KQ_RBF_Gaussian(epsilon_outer, f_I_KQ.squeeze(), jnp.zeros([1]), jnp.ones([1, 1]), scale)
    I_NKQ = KQ_Matern_12_Uniform(u_theta, f_I_KQ_normalized, jnp.zeros([1]), jnp.ones([1]), lengthscale, lmbda)
    I_NKQ = I_NKQ * scale + shift
    return I_NKQ


def nested_kernel_quadrature_multi_level(Theta, u_theta, x, u_x, x_prev, u_x_prev):
    T, N = x.shape[0], x.shape[1]
    lengthscale = 1.0
    # This is nest kernel quadrature for the inner expectation
    f_val = price(Theta, x)
    scale, shift = f_val.std(1), f_val.mean(1)
    f_val_normalized = (f_val - shift[:, None]) / scale[:, None]
    f_val_normalized = jnp.nan_to_num(f_val_normalized, 0.)
    lmbda = 1e-6
    # I_KQ = KQ_Matern_12_Uniform_Vectorized(u_x[:, :, None], f_val_normalized, jnp.zeros([T, 1]), jnp.ones([T, 1]), lengthscale, lmbda)
    I_KQ = KQ_RBF_Gaussian_Vectorized(x[:, :, None], f_val_normalized, jnp.zeros([T, 1]), jnp.ones([T, 1, 1]), lengthscale, lmbda)
    I_KQ = I_KQ * scale + shift
    f_I_KQ = jnp.maximum(I_KQ, 0)

    f_val_prev = price(Theta, x_prev)
    scale, shift = f_val_prev.std(1), f_val_prev.mean(1)
    f_val_prev_normalized = (f_val_prev - shift[:, None]) / scale[:, None]
    f_val_prev_normalized = jnp.nan_to_num(f_val_prev_normalized, 0.)
    # I_KQ_prev = KQ_Matern_12_Uniform_Vectorized(u_x_prev[:, :, None], f_val_prev_normalized, jnp.zeros([T, 1]), jnp.ones([T, 1]), lengthscale, lmbda)
    I_KQ_prev = KQ_RBF_Gaussian_Vectorized(x_prev[:, :, None], f_val_prev_normalized, jnp.zeros([T, 1]), jnp.ones([T, 1, 1]), lengthscale, lmbda)
    I_KQ_prev = I_KQ_prev * scale + shift
    f_I_KQ_prev = jnp.maximum(I_KQ_prev, 0)

    # This is nest kernel quadrature for the outer expectation
    lengthscale = 1.0
    f_difference = f_I_KQ - f_I_KQ_prev
    scale, shift = f_difference.std(), f_difference.mean()
    f_difference_normalized = (f_difference - shift) / scale
    f_difference_normalized = jnp.nan_to_num(f_difference_normalized, 0.)
    I_MLKQ = KQ_Matern_12_Uniform(u_theta, f_difference_normalized, jnp.zeros([1]), jnp.ones([1]), lengthscale, lmbda)
    I_MLKQ = I_MLKQ * scale + shift
    return I_MLKQ


def mlmc(eps, N0, L, use_kq, rng_key):
    rng_key, _ = jax.random.split(rng_key)
    # Check input parameters
    if L < 2:
        raise ValueError('error: needs L >= 2')

    # Initialisation
    if use_kq:
        Nl = 2 * (2 ** jnp.arange(L))
        Tl = N0 / eps * (2 ** (-1. * jnp.arange(L)))
    else:
        Nl = 2 * (2 ** jnp.arange(L))
        Tl = N0 / eps * (2 ** (-1. * jnp.arange(L)))     
          
    Yl = jnp.zeros(L)
    Cl = jnp.zeros(L)

    for l in range(L):
        if l == 0:
            N, T = int(Nl[l]), int(Tl[l]) + 1
            rng_key, _ = jax.random.split(rng_key)
            Theta, u_theta, x, u_x = sample(T, N, False, rng_key)

            if use_kq:
                Y = nested_kernel_quadrature(Theta, u_theta, x, u_x)
                Yl = Yl.at[l].set(Y)
            else:
                Y = nested_monte_carlo(Theta, u_theta, x, u_x)
                Yl = Yl.at[l].set(Y)
            Cl = Cl.at[l].set(N * T)
        else:
            N, N_prev, T = int(Nl[l]), int(Nl[l-1]), int(Tl[l]) + 1
            rng_key, _ = jax.random.split(rng_key)
            Theta, u_theta, x, u_x = sample(T, N, False, rng_key)
            x_prev, u_x_prev = x[:, :N_prev], u_x[:, :N_prev]

            if use_kq:
                Y_diff = nested_kernel_quadrature_multi_level(Theta, u_theta, x, u_x, x_prev, u_x_prev)
                Yl = Yl.at[l].set(Y_diff)
            else:
                Y_prev = nested_monte_carlo(Theta, u_theta, x_prev, u_x_prev)
                Y = nested_monte_carlo(Theta, u_theta, x, u_x)
                Yl = Yl.at[l].set(Y - Y_prev)
            Cl = Cl.at[l].set(N * T)

    # Final estimation
    P = Yl.sum()
    C = Cl.sum()
    return P, C


def sample(T, N, use_qmc, rng_key):
    if use_qmc:
        u_outer = scipy.stats.qmc.Sobol(1).random(T)
    else:
        rng_key, _ = jax.random.split(rng_key)
        u_outer = jax.random.uniform(rng_key, shape=(T, 1))
    epsilon_outer = jax.scipy.stats.norm.ppf(u_outer)
    St = S0 * jnp.exp(sigma * jnp.sqrt(t_finance) * epsilon_outer - 0.5 * (sigma ** 2) * t_finance)
    output_shape = (St.shape[0], N)

    if use_qmc:
        u_inner = [scipy.stats.qmc.Sobol(1).random(T) for _ in range(N)]
        u_inner = np.array(u_inner).squeeze().T
    else:
        rng_key, _ = jax.random.split(rng_key)
        u_inner = jax.random.uniform(rng_key, shape=output_shape)
    epsilon_inner = jax.scipy.stats.norm.ppf(u_inner)
    return St, u_outer, epsilon_inner, u_inner


def run(args):
    rng_key = jax.random.PRNGKey(args.seed)
    true_value = 3.077
    print(f"True value: {true_value}")

    L = 5
    I_mc_err_dict = {}
    I_kq_err_dict = {}

    N0 = int(2 ** L * args.eps) + 1
    
    if args.multi_level:
        use_kq = False
        I_MLMC_nmc, cost = mlmc(args.eps, N0, L, use_kq, rng_key)
        print(f"MLMC MC: {I_MLMC_nmc} with cost {cost}")
        I_mc_err_dict[f'cost_{cost}'] = jnp.abs(I_MLMC_nmc - true_value)

        if args.eps > 0.00003:
            use_kq = True
            I_MLMC_nkq, cost = mlmc(args.eps, N0, L, use_kq, rng_key)
        else:
            I_MLMC_nkq = jnp.nan
            cost = jnp.nan
        print(f"MLMC KQ: {I_MLMC_nkq} with cost {cost}")
        I_kq_err_dict[f'cost_{cost}'] = jnp.abs(I_MLMC_nkq - true_value)
    else:
        N0 = 1
        N_total = int((1. / args.eps) * N0)
        T_total = int((1. / args.eps) * N0)
        rng_key, _ = jax.random.split(rng_key)
        Theta, u_theta, x, u_x = sample(T_total, N_total, args.qmc, rng_key)
        cost = N_total * T_total

        I_nmc = nested_monte_carlo(Theta, u_theta, x, u_x)
        print(f"NMC (QMC {args.qmc}): {I_nmc} with cost {cost}")
        I_mc_err_dict[f'cost_{cost}'] = jnp.abs(I_nmc - true_value)

        I_nkq = nested_kernel_quadrature(Theta, u_theta, x, u_x)
        print(f"NKQ (QMC {args.qmc}): {I_nkq} with cost {cost}")
        I_kq_err_dict[f'cost_{cost}'] = jnp.abs(I_nkq - true_value)
    
    with open(f"{args.save_path}/seed_{args.seed}_MC", 'wb') as file:
        pickle.dump(I_mc_err_dict, file)
    with open(f"{args.save_path}/seed_{args.seed}_KQ", 'wb') as file:
        pickle.dump(I_kq_err_dict, file)
    return

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/finance/'
    args.save_path += f"multi_level_{args.multi_level}__qmc_{args.qmc}__eps_{args.eps}__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    run(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    import shutil
    if os.path.exists(f"{save_path}__complete"):
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")
    print("Finished running")
