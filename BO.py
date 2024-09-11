from typing import List,Tuple 
import jax
from jax import config
import jax.numpy as np
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp
from functools import partial
import os
import time
import argparse
config.update("jax_enable_x64", True)
import gpjax as gpx
import warnings
from tqdm import tqdm
import shutil
import pwd

from utils.kernel_means import *
from datasets.bo_datasets import *
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")

cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


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
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--utility', type=str)
    parser.add_argument('--datasets', type=str)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--N', type=int, default=20)
    args = parser.parse_args()
    return args


class Kernel:
    def __init__(self, length_scale=1.0, variance=1.0, kernel_type='rbf'):
        self.length_scale = length_scale
        self.variance = variance
        self.kernel_type = kernel_type
    
    def compute(self, x1, x2):
        """
        Compute the kernel matrix between x1 and x2 based on the specified kernel type.
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        
        if self.kernel_type == 'rbf':
            return self._squared_exponential(x1, x2)
        elif self.kernel_type == 'matern_3_2':
            return self._matern_3_2(x1, x2)
        else:
            raise ValueError("Unsupported kernel type. Choose 'rbf' or 'matern_3_2'.")

    def _squared_exponential(self, x1, x2):
        """
        Compute the squared exponential (RBF) kernel between x1 and x2.
        """
        dists = np.sum(x1**2, 1)[:, np.newaxis] + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.variance * np.exp(-0.5 / self.length_scale**2 * dists)
    
    def _matern_3_2(self, x1, x2):
        """
        Compute the Matérn 3/2 kernel between x1 and x2.
        """
        x1 = np.asarray(x1) / self.length_scale
        x2 = np.asarray(x2) / self.length_scale
        dists = np.sqrt(np.sum((x1[:, np.newaxis] - x2)**2, axis=2))  # Euclidean distance
        term = (np.sqrt(3) * dists) 
        return self.variance * (1 + term) * np.exp(-term)
    
def gp_posterior(X_train, y_train, X_test, kernel):
    """
    Compute the GP posterior mean and variance given training and test data using the specified kernel.
    """
    # Compute the kernel matrices
    K = kernel.compute(X_train, X_train)  # Training kernel
    K_s = kernel.compute(X_train, X_test)  # Training vs. test kernel
    K_ss = kernel.compute(X_test, X_test)  # Test kernel
    
    # Add noise to the diagonal of K
    noise = 1e-6
    K += noise * np.eye(len(X_train))
    
    # Compute the inverse of K
    K_inv = np.linalg.inv(K)
    
    # Compute the posterior mean
    mu_s = K_s.T.dot(K_inv).dot(y_train)
    
    # Compute the posterior variance
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s  # Return mean and variance


def negative_ei_kq(args, x, posterior, X, y, y_best, num_samples, rng_key):
    sampler = partial(posterior, X_train=X, y_train=y)
    mu, var = sampler(X_test=x[None, :])
    samples = rng_key.normal(mu.squeeze(), np.sqrt(var.squeeze()), num_samples)
    increases = np.maximum(samples - y_best, 0)
    # ei = np.mean(increases)
    if args.kernel == 'rbf':
        ei = KQ_RBF_Gaussian(jnp.array(samples[:, None]), jnp.array(increases.squeeze()), jnp.array(mu[0]), jnp.array(var))
    elif args.kernel == 'matern':
        ei = KQ_Matern_Gaussian(jnp.array((samples - mu[0]) / np.sqrt(var[0]))[:, None], jnp.array(increases.squeeze()))
    return -ei.squeeze()


def negative_ei_mc(args, x, posterior, X, y, y_best, num_samples, rng_key):
    sampler = partial(posterior, X_train=X, y_train=y)
    mu, var = sampler(X_test=x[None, :])
    samples = rng_key.normal(mu.squeeze(), np.sqrt(var.squeeze()), num_samples)
    increases = np.maximum(samples - y_best, 0)
    ei = np.mean(increases)
    return -ei.squeeze()


def negative_ei_closed_form(args, x, posterior, X, y, y_best, num_samples, rng_key):
    # Get the mean (mu) and standard deviation (sigma) from the sampler
    sampler = partial(posterior, X_train=X, y_train=y)
    mu, var = sampler(X_test=x[None, :])

    # Compute the improvement and the standard normal terms
    z = (mu - y_best) / np.sqrt(var)
    ei = (mu - y_best) * scipy.stats.norm.cdf(z) + np.sqrt(var) * scipy.stats.norm.pdf(z)
    return -ei.squeeze()


def negative_ei_look_ahead_mc(args, rng_key, x, posterior, lower_bound, upper_bound, X, y, 
                              y_best, num_samples):
    outer_sampler = partial(posterior, X_train=X, y_train=y)
    mu, var = outer_sampler(X_test=x[None, :])
    outer_samples = rng_key.normal(mu.squeeze(), np.sqrt(var.squeeze()), num_samples)
    dim = X.shape[1]

    inner_expectation = np.zeros((num_samples, 1))
    for s, sample in enumerate(outer_samples):
        init_x = rng_key.uniform(lower_bound, upper_bound, (1, dim))
        negative_utility_fn = lambda x_prime: negative_ei_mc(args, x_prime, posterior, 
                                                             np.vstack([X, x]), 
                                                             np.vstack([y, sample]), 
                                                             y_best, num_samples, rng_key)
        bounds = [(lower_bound[0], upper_bound[0])]  # Repeat bounds for each parameter dimension
        result = scipy.optimize.minimize(negative_utility_fn, init_x.squeeze(), method = 'Nelder-Mead', 
                                         bounds=bounds, options={'maxiter': 5})
        inner_expectation[s,:] = result.fun

    increases = np.maximum(outer_samples - y_best, 0) + inner_expectation
    ei = np.mean(increases)
    return -ei.squeeze()


def negative_ei_look_ahead_kq(args, rng_key, x, posterior, lower_bound, upper_bound, X, y, 
                              y_best, num_samples):
    outer_sampler = partial(posterior, X_train=X, y_train=y)
    mu, var = outer_sampler(X_test=x[None, :])
    outer_samples = rng_key.normal(mu.squeeze(), np.sqrt(var.squeeze()), num_samples)
    dim = X.shape[1]
    
    inner_expectation = np.zeros((num_samples,))
    for s, sample in enumerate(outer_samples):
        init_x = rng_key.uniform(lower_bound, upper_bound, (1, dim))
        negative_utility_fn = lambda x_prime: negative_ei_kq(args, x_prime, posterior, 
                                                             np.vstack([X, x]), 
                                                             np.vstack([y, sample]), 
                                                             y_best, num_samples, rng_key)
        
        bounds = [(lower_bound[0], upper_bound[0])]  # Repeat bounds for each parameter dimension
        result = scipy.optimize.minimize(negative_utility_fn, init_x.squeeze(), method = 'Nelder-Mead', 
                                         bounds=bounds, options={'maxiter': 5})
        inner_expectation[s] = result.fun

    increases = np.maximum(outer_samples - y_best, 0) - inner_expectation # this is minus, because we compute negative EI
    if args.kernel == 'rbf':
        ei = KQ_RBF_Gaussian(jnp.array(outer_samples[:, None]), jnp.array(increases.squeeze()), jnp.array(mu[0]), jnp.array(var))
    elif args.kernel == 'matern':
        ei = KQ_Matern_Gaussian(jnp.array((outer_samples - mu[0]) / np.sqrt(var[0]))[:, None], jnp.array(increases.squeeze()))
    else:
        raise ValueError("Kernel not recognised")
    return -ei.squeeze()


def optimise_sample(args, rng_key, posterior, X, y, lower_bound, upper_bound, y_best, num_samples, num_initial_sample_points):
    dim = X.shape[1]
    initial_conditions = rng_key.uniform(lower_bound, upper_bound, (num_initial_sample_points, dim))

    # We want to maximise the utility function, but the optimiser performs minimisation.
    # Since we're minimising the sample drawn, the sample is actually the negative utility function.
    # Expected Improvement with closed form
    if args.utility == 'EI_closed_form':
        negative_utility_fn = lambda x: negative_ei_closed_form(args, x, posterior, X, y, y_best, num_samples, rng_key)
    # Expected Improvement with Monte Carlo
    elif args.utility == 'EI_mc':
        negative_utility_fn = lambda x: negative_ei_mc(args, x, posterior, X, y, y_best, num_samples, rng_key)
    # Expected Improvement with Kernel Quadrature
    elif args.utility == 'EI_kq':
        negative_utility_fn = lambda x: negative_ei_kq(args, x, posterior, X, y, y_best, num_samples, rng_key)
    elif args.utility == 'EI_look_ahead_mc':
        negative_utility_fn = lambda x: negative_ei_look_ahead_mc(args, rng_key, x, posterior, lower_bound, 
                                                                  upper_bound, X, y, 
                                                                  y_best, num_samples)
    elif args.utility == 'EI_look_ahead_kq':
        negative_utility_fn = lambda x: negative_ei_look_ahead_kq(args, rng_key, x, posterior, lower_bound, 
                                                                  upper_bound, X, y, 
                                                                  y_best, num_samples)
    else:
        raise ValueError("Utility function not recognised")
    
    bounds = [(lower_bound[0], upper_bound[0])] * len(initial_conditions[0])  # Repeat bounds for each parameter dimension

    params_list = []
    fun_vals_list = []

    # Run optimization for each initial condition and store both params and function values
    for init_x in initial_conditions:
        result = scipy.optimize.minimize(negative_utility_fn, init_x, method='Nelder-Mead', 
                                         bounds=bounds, options={'maxiter': 10, 'disp': False})
        params_list.append(result.x)  # Optimized parameters
        fun_vals_list.append(result.fun)  # Function value at the minimum

    x_star = np.array(params_list)[np.argmin(np.array(fun_vals_list))]
    return x_star

def plot_bayes_opt(
    args,
    rng_key,
    get_data_fn,
    posterior,
    X,
    y,
    queried_x,
    lower_bound, 
    upper_bound
) -> None:
    sampler = partial(posterior, X_train=X, y_train=y)
    plt_x = np.linspace(lower_bound[0], upper_bound[0], 1000).reshape(-1, 1)
    mu, var = sampler(X_test=plt_x)
    sample_y = rng_key.multivariate_normal(mu.squeeze(), var, 1).T
    plt_y = get_data_fn(plt_x)

    fig, ax = plt.subplots()
    ax.plot(plt_x, mu.squeeze(), label="Predictive Mean", color=cols[1])
    ax.fill_between(
        plt_x.squeeze(),
        mu.squeeze() - 2 * np.sqrt(np.diag(var)),
        mu.squeeze() + 2 * np.sqrt(np.diag(var)),
        alpha=0.2,
        label="Two sigma",
        color=cols[1],
    )
    ax.plot(plt_x, sample_y, label="Posterior Sample")
    ax.plot(
        plt_x,
        plt_y,
        label="True Function",
        color=cols[0],
        linestyle="--",
        linewidth=2,
    )
    ax.axvline(x=0.0, linestyle=":", color=cols[3], label="True Optimum")
    ax.scatter(X, y, label="Observations", color=cols[2], zorder=2)
    ax.scatter(
        queried_x.squeeze(),
        get_data_fn(queried_x[None,:]).squeeze(),
        label="Qeury Point",
        marker="*",
        color=cols[3],
        zorder=3,
    )
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    plt.tight_layout()
    plt.savefig(args.save_path + f"bo_{len(X)}.png")
    plt.close()
    return

def main(args):
    bo_iters = 100
    initial_sample_num = 5
    rng_key = np.random.default_rng(args.seed)

    # load datasets
    if args.datasets == 'ackley':
        dim = args.dim
        get_data_fn = partial(load_ackley, dim=args.dim)
        lower_bound, upper_bound = np.array([-5.0]), np.array([5.0])
        ground_truth_best_y = 1.4019
    elif args.datasets == 'emulator':
        get_data_fn = emulator
        lower_bound, upper_bound = np.array([-10.0]), np.array([10.0])
        ground_truth_best_y = 0.
        dim = 1
    else:
        raise ValueError("Dataset not recognised")
    X = rng_key.uniform(lower_bound, upper_bound, (initial_sample_num, dim))
    y = get_data_fn(X)
    X_init, y_init = X, y

    # GP prior
    pairwise_distances = np.abs(X[:, None, :] - X[None, :, :])
    lengthscales = np.median(pairwise_distances, axis=(0, 1)) / 10
    # kernel = gpx.kernels.Matern52(n_dims=dim, lengthscale=lengthscales)
    kernel = Kernel(length_scale=lengthscales, variance=1.0, kernel_type='matern_3_2')
    posterior = partial(gp_posterior, kernel=kernel)
    num_samples = args.N

    # BO loop
    nmse_list = []
    for iter in tqdm(range(bo_iters)):
        y_best = np.max(y)

        # Draw a sample from the posterior, and find the minimiser of it
        x_star = optimise_sample(args, rng_key, posterior, X, y, 
                                 lower_bound, upper_bound, y_best, num_samples, num_initial_sample_points=1)

        # if (iter + len(y)) % 10 == 0:
        # plot_bayes_opt(args, rng_key, get_data_fn, posterior, X, y, x_star, lower_bound, upper_bound,)

        # Evaluate the black-box function at the best point observed so far, and add it to the dataset
        y_star = get_data_fn(x_star[None, :])
        X = np.vstack([X, x_star])
        y = np.vstack([y, y_star])

        nmse_list.append((np.max(y) - ground_truth_best_y) ** 2 / (np.max(y_init) - ground_truth_best_y) ** 2)

    plt.figure()
    plt.plot(nmse_list)
    plt.xlabel("Iterations")
    plt.ylabel("NMSE")
    plt.yscale("log")
    plt.savefig(args.save_path + f"bo_nmse.png")
    plt.close()

    np.save(args.save_path + f"bo_nmse.npy", np.array(nmse_list))
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/bo/{args.datasets}/'
    args.save_path += f"{args.utility}__dim_{args.dim}__N_{args.N}__kernel_{args.kernel}__seed_{args.seed}/"
    if os.path.exists(args.save_path) and os.listdir(args.save_path):  # If directory exists and is not empty
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == "__main__":
    args = get_config()
    args = create_dir(args)
    main(args)
    print("========================================")
    print("Finished running")
    print(f"Results saved at {args.save_path}")