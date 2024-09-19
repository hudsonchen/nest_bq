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
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for finance data')

    # Data settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    args = parser.parse_args()
    return args


def run(args, N, T, use_qmc, rng_key):
    # Hyperparameters for the Black Scholes model
    s = 0.2
    sigma = 0.3
    T_finance = 2
    t_finance = 1
    S0 = 100
    K1 = 50
    K2 = 150
    
    def price(St, N, epsilon):
        """
        Computes the price ST at time T_finance.
        ST is sampled from the conditional distribution p(ST|St).
        Computes the loss \psi(ST) - \psi((1+s)ST) caused by the shock. 
        Their shape is T * N
        
        Args:
            St: (T, 1) the price at time t_finance
            N: number of samples
            
        Returns:
            ST: (T, N, 1)
            f(ST): (T, N)
        """
        ST = St * jnp.exp(sigma * jnp.sqrt((T_finance - t_finance)) * epsilon - 0.5 * (sigma ** 2) * (T_finance - t_finance))
        psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
        psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * jnp.maximum(
            (1 + s) * ST - (K1 + K2) / 2, 0)
        return ST, psi_ST_1 - psi_ST_2

    if use_qmc:
        sequence = scipy.stats.qmc.Sobol(1).random(T)
        epsilon_outer = jax.scipy.stats.norm.ppf(sequence)
    else:
        rng_key, _ = jax.random.split(rng_key)
        epsilon_outer = jax.random.normal(rng_key, shape=(T, 1))
    St = S0 * jnp.exp(sigma * jnp.sqrt(t_finance) * epsilon_outer - 0.5 * (sigma ** 2) * t_finance)
    output_shape = (St.shape[0], N)

    if use_qmc:
        sequence = [scipy.stats.qmc.Sobol(1).random(T) for _ in range(N)]
        sequence = np.array(sequence).squeeze()
        epsilon_inner = jax.scipy.stats.norm.ppf(sequence).T
    else:
        rng_key, _ = jax.random.split(rng_key)
        epsilon_inner = jax.random.normal(rng_key, shape=output_shape)
    ST, loss = price(St, N, epsilon_inner)
    
    # Debug code
    # ST_large, loss_large = price(St, 2000, rng_key)
    # I = loss_large.mean(1)
    # Debug code

    # Nested Monte Carlo
    I_MC = loss.mean(1)
    I_NMC = jnp.maximum(I_MC, 0).mean(0)

    # This is nest kernel quadrature for the inner expectation
    mu_ST_St = -sigma ** 2 * (T_finance - t_finance) / 2 + jnp.log(St)
    std_ST_St = jnp.sqrt(sigma ** 2 * (T_finance - t_finance)) * jnp.ones_like(St)
    I_KQ = KQ_log_RBF_log_Gaussian_Vectorized(ST[:, :, None], loss, mu_ST_St, std_ST_St)
    f_I_KQ = jnp.maximum(I_KQ, 0)

    # This is nest kernel quadrature for the outer expectation
    mu_St = -sigma ** 2 * t_finance / 2 + jnp.log(S0)
    std_St = jnp.sqrt(sigma ** 2 * (t_finance))
    I_NKQ = KQ_log_RBF_log_Gaussian(St, f_I_KQ.squeeze(), mu_St, std_St)
    return I_NMC, I_NKQ


def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    true_value = 3.077

    print(f"True value: {true_value}")
    # N_list = jnp.arange(10, 50, 5).tolist()
    # T_list = jnp.arange(10, 50, 5).tolist()
    N_list = [50, 100, 200, 300, 400, 500]
    # N_list = [1000]

    I_NMC_err_dict, I_NKQ_err_dict = {}, {}
    I_NMC_err_qmc_dict, I_NKQ_err_qmc_dict = {}, {}

    num_seeds = 10

    for N in N_list:
        T = N
        I_NMC_errors, I_NKQ_errors = [], []
        I_NMC_qmc_errors, I_NKQ_qmc_errors = [], []

        for s in range(num_seeds):
            rng_key, _ = jax.random.split(rng_key)
            use_qmc = False
            I_NMC, I_NKQ = run(args, N, T, use_qmc, rng_key)

            use_qmc = True
            rng_key, _ = jax.random.split(rng_key)
            I_NMC_qmc, I_NKQ_qmc = run(args, N, T, use_qmc, rng_key)

            I_NMC_errors.append(jnp.abs(I_NMC - true_value))
            I_NKQ_errors.append(jnp.abs(I_NKQ - true_value))
            I_NMC_qmc_errors.append(jnp.abs(I_NMC_qmc - true_value))
            I_NKQ_qmc_errors.append(jnp.abs(I_NKQ_qmc - true_value))
        
        I_NMC_err = jnp.median(jnp.array(I_NMC_errors))
        I_NKQ_err = jnp.median(jnp.array(I_NKQ_errors))
        I_NKQ_err_qmc = jnp.median(jnp.array(I_NKQ_qmc_errors))
        I_NMC_err_qmc = jnp.median(jnp.array(I_NMC_qmc_errors))
        I_NMC_err_dict[(N, T)], I_NKQ_err_dict[(N, T)] = I_NMC_err, I_NKQ_err
        I_NMC_err_qmc_dict[(N, T)], I_NKQ_err_qmc_dict[(N, T)] = I_NMC_err_qmc, I_NKQ_err_qmc

        methods = ["NKQ", "NMC", "NKQ (QMC)", "NMC (QMC)"]
        errs = [I_NKQ_err, I_NMC_err, I_NKQ_err_qmc, I_NMC_err_qmc]

        print(f"T = {T} and N = {N}")
        print("========================================")
        print("Methods:    " + " ".join([f"{method:<10}" for method in methods]))
        print("ASE:       " + " ".join([f"{value:<10.6f}" for value in errs]))
        print("========================================\n\n")

    with open(f"{args.save_path}/seed_{args.seed}__NKQ", 'wb') as file:
        pickle.dump(I_NKQ_err_dict, file)
    with open(f"{args.save_path}/seed_{args.seed}__NMC", 'wb') as file:
        pickle.dump(I_NMC_err_dict, file)
    with open(f"{args.save_path}/seed_{args.seed}__NKQ_QMC", 'wb') as file:
        pickle.dump(I_NKQ_err_qmc_dict, file)
    with open(f"{args.save_path}/seed_{args.seed}__NMC_QMC", 'wb') as file:
        pickle.dump(I_NMC_err_qmc_dict, file)
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/finance/'
    args.save_path += f"seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(f"{args.save_path}/figures/", exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print(f'Device is {jax.devices()}')
    print(args.seed)
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")

