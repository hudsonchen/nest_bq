from typing import List,Tuple 
import jax
from jax import config
import jax.numpy as jnp
import scipy
import numpy as np
import jaxopt
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

from utils.kernel_means import *
from datasets.bo_datasets import *
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")

cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


def get_config():
    parser = argparse.ArgumentParser(description='Toy example')
    # Args settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--utility', type=str)
    parser.add_argument('--datasets', type=str)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--dim', type=int, default=2)
    args = parser.parse_args()
    return args


def get_posterior(
    data: gpx.Dataset, prior: gpx.gps.Prior, key: jnp.ndarray
) -> gpx.gps.AbstractPosterior:
    # Our function is noise-free, so we set the observation noise's standard deviation to a very small value
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=data.n, obs_stddev=jnp.array(1e-6)
    )

    posterior = prior * likelihood

    posterior, _ = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=data,
        verbose=False
    )

    return posterior

def negative_ei_kq(args, x, posterior, train_data, y_best, num_samples, rng_key):
    # Sample from the posterior distribution at x
    rng_key, _ = jax.random.split(rng_key)
    sampler = partial(posterior.predict, train_data=train_data)
    dist = sampler(x[None, :])
    mu, var = dist.mean(), dist.variance()
    samples = dist.sample(rng_key, [num_samples])
    increases = jnp.maximum(samples - y_best, 0)
    # ei = jnp.mean(increases)
    ei = KQ_RBF_Gaussian(rng_key, samples, increases.squeeze(), mu, var[:, None])
    return -ei.squeeze()


def negative_ei_mc(args, x, posterior, train_data, y_best, num_samples, rng_key):
    rng_key, _ = jax.random.split(rng_key)
    sampler = partial(posterior.predict, train_data=train_data)
    samples = sampler(x[None, :]).sample(rng_key, [num_samples])
    increases = jnp.maximum(samples - y_best, 0)
    ei = jnp.mean(increases)
    return -ei.squeeze()

def negative_ei_look_ahead_mc(args, rng_key, x, posterior, lower_bound, upper_bound, train_data, 
                              y_best, num_initial_sample_points, num_samples):
    rng_key, _ = jax.random.split(rng_key)
    outer_sampler = partial(posterior.predict, train_data=train_data)
    outer_samples = outer_sampler(x[None, :]).sample(rng_key, [num_samples])
    dim = train_data.X.shape[1]

    inner_expectation = jnp.zeros((num_samples, 1))
    for s, sample in enumerate(outer_samples):
        rng_key, _ = jax.random.split(rng_key)
        init_x = jax.random.uniform(rng_key, shape=(dim, ), minval=lower_bound, maxval=upper_bound)
        negative_utility_fn = lambda x_prime: negative_ei_mc(args, x_prime[None, :], posterior, train_data + gpx.Dataset(X=x[None, :], y=sample[:, None]), 
                                                       y_best, num_samples, rng_key)
        
        bounds = [(lower_bound[0], upper_bound[0])]  # Repeat bounds for each parameter dimension
        params_list = []
        fun_vals_list = []

        # Run optimization for each initial condition and store both params and function values
        result = scipy.optimize.minimize(negative_utility_fn, init_x, method = 'L-BFGS-B', 
                                            bounds=bounds, options={'maxiter': 2})
        params_list.append(result.x)  # Optimized parameters
        fun_vals_list.append(result.fun)  # Function value at the minimum

        inner_expectation.at[s, :].set(np.array(fun_vals_list).max())

    increases = jnp.maximum(outer_samples - y_best, 0) + inner_expectation
    ei = jnp.mean(increases)
    return -ei.squeeze()


def negative_ei_look_ahead_kq(args, rng_key, x, posterior, lower_bound, upper_bound, train_data, 
                              y_best, num_initial_sample_points, num_samples):
    rng_key, _ = jax.random.split(rng_key)
    outer_sampler = partial(posterior.predict, train_data=train_data)
    outer_samples = outer_sampler(x[None, :]).sample(rng_key, [num_samples])
    dim = train_data.X.shape[1]
    
    inner_expectation = jnp.zeros((num_samples, 1))
    for s, sample in enumerate(outer_samples):
        rng_key, _ = jax.random.split(rng_key)
        init_x = jax.random.uniform(rng_key, shape=(dim, ), minval=lower_bound, maxval=upper_bound)
        negative_utility_fn = lambda x_prime: negative_ei_kq(args, x_prime[None, :], posterior, train_data + gpx.Dataset(X=x[None, :], y=sample[:, None]), 
                                                       y_best, num_samples, rng_key)
        
        bounds = [(lower_bound[0], upper_bound[0])] # Repeat bounds for each parameter dimension
        params_list = []
        fun_vals_list = []

        # Run optimization for each initial condition and store both params and function values
        result = scipy.optimize.minimize(negative_utility_fn, init_x, method = 'L-BFGS-B', 
                                            bounds=bounds, options={'maxiter': 2})
        params_list.append(result.x)  # Optimized parameters
        fun_vals_list.append(result.fun)  # Function value at the minimum

        inner_expectation.at[s, :].set(np.array(fun_vals_list).max())

    increases = np.maximum(outer_samples - y_best, 0) + inner_expectation
    mu, var = outer_sampler(x[None, :]).mean(), outer_sampler(x[None, :]).variance()
    if args.kernel == 'rbf':
        ei = KQ_RBF_Gaussian(rng_key, outer_samples, increases.squeeze(), mu, var)
    elif args.kernel == 'matern':
        ei = KQ_Matern_Gaussian(rng_key, (outer_samples - mu) / np.sqrt(var), increases.squeeze())
    else:
        raise ValueError("Kernel not recognised")
    return -ei.squeeze()


def negative_ei_closed_form(args, x, posterior, train_data, y_best, num_samples, rng_key):
    # Get the mean (mu) and standard deviation (sigma) from the sampler
    sampler = partial(posterior.predict, train_data=train_data)
    dist = sampler(x)
    mu = dist.mean()
    sigma = dist.stddev()

    # Compute the improvement and the standard normal terms
    z = (mu - y_best) / sigma
    ei = (mu - y_best) * jax.scipy.stats.norm.cdf(z) + sigma * jax.scipy.stats.norm.pdf(z)
    return -ei.squeeze()

def optimise_sample(args, rng_key, posterior, train_data, lower_bound, upper_bound, y_best, num_initial_sample_points):
    rng_key, _ = jax.random.split(rng_key)
    dim = train_data.X.shape[1]
    initial_conditions = jax.random.uniform(rng_key, shape=(num_initial_sample_points, dim), minval=lower_bound, maxval=upper_bound)
    num_samples = 10

    # We want to maximise the utility function, but the optimiser performs minimisation.
    # Since we're minimising the sample drawn, the sample is actually the negative utility function.
    # Expected Improvement with closed form
    if args.utility == 'EI_closed_form':
        negative_utility_fn = lambda x: negative_ei_closed_form(args, x, posterior, train_data, y_best, num_samples, rng_key)
    # Expected Improvement with Monte Carlo
    elif args.utility == 'EI_mc':
        negative_utility_fn = lambda x: negative_ei_mc(args, x, posterior, train_data, y_best, num_samples, rng_key)
    # Expected Improvement with Kernel Quadrature
    elif args.utility == 'EI_kq':
        negative_utility_fn = lambda x: negative_ei_kq(args, x, posterior, train_data, y_best, num_samples, rng_key)
    elif args.utility == 'EI_look_ahead_mc':
        negative_utility_fn = lambda x: negative_ei_look_ahead_mc(args, rng_key, x, posterior, lower_bound, 
                                                                  upper_bound, train_data, 
                                                                  y_best, num_initial_sample_points, num_samples)
    elif args.utility == 'EI_look_ahead_kq':
        negative_utility_fn = lambda x: negative_ei_look_ahead_kq(args, rng_key, x, posterior, lower_bound, 
                                                                  upper_bound, train_data, 
                                                                  y_best, num_initial_sample_points, num_samples)
    else:
        raise ValueError("Utility function not recognised")
    
    bounds = [(lower_bound[0], upper_bound[0])] * len(initial_conditions[0])  # Repeat bounds for each parameter dimension

    params_list = []
    fun_vals_list = []

    # Run optimization for each initial condition and store both params and function values
    for init_x in initial_conditions:
        result = scipy.optimize.minimize(negative_utility_fn, init_x, method='L-BFGS-B', 
                                         bounds=bounds, options={'maxiter': 2, 'disp': False})
        params_list.append(result.x)  # Optimized parameters
        fun_vals_list.append(result.fun)  # Function value at the minimum

    x_star = jnp.array(params_list)[jnp.argmin(jnp.array(fun_vals_list))]
    return x_star

def plot_bayes_opt(
    args,
    rng_key,
    get_data_fn,
    posterior: gpx.gps.AbstractPosterior,
    dataset: gpx.Dataset,
    queried_x,
    lower_bound, 
    upper_bound
) -> None:
    sampler = partial(posterior.predict, train_data=dataset)
    rng_key, _ = jax.random.split(rng_key)
    plt_x = jnp.linspace(lower_bound[0], upper_bound[0], 1000).reshape(-1, 1)
    y = get_data_fn(plt_x)
    sample_y = sampler(plt_x).sample(rng_key, [1]).squeeze()

    latent_dist = posterior.predict(plt_x, train_data=dataset)
    predictive_dist = posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    fig, ax = plt.subplots()
    ax.plot(plt_x, predictive_mean, label="Predictive Mean", color=cols[1])
    ax.fill_between(
        plt_x.squeeze(),
        predictive_mean - 2 * predictive_std,
        predictive_mean + 2 * predictive_std,
        alpha=0.2,
        label="Two sigma",
        color=cols[1],
    )
    ax.plot(
        plt_x,
        predictive_mean - 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    ax.plot(
        plt_x,
        predictive_mean + 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    ax.plot(plt_x, sample_y, label="Posterior Sample")
    ax.plot(
        plt_x,
        y,
        label="True Function",
        color=cols[0],
        linestyle="--",
        linewidth=2,
    )
    ax.axvline(x=2.0087, linestyle=":", color=cols[3], label="True Optimum")
    ax.scatter(dataset.X, dataset.y, label="Observations", color=cols[2], zorder=2)
    ax.scatter(
        queried_x,
        sampler(queried_x[None,:]).sample(rng_key, [1]).squeeze(),
        label="Qeury Point",
        marker="*",
        color=cols[3],
        zorder=3,
    )
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    plt.tight_layout()
    plt.savefig(args.save_path + f"bo_{len(dataset.X)}.png")
    plt.close()
    return

def main(args):
    bo_iters = 50
    initial_sample_num = 5
    rng_key = jax.random.PRNGKey(args.seed)

    # load datasets
    if args.datasets == 'ackley':
        dim = args.dim
        get_data_fn = partial(load_ackley, dim=args.dim)
        lower_bound, upper_bound = jnp.array([-5.0]), jnp.array([5.0])
        ground_truth_best_y = 1.4019
    elif args.datasets == 'emulator':
        get_data_fn = emulator
        lower_bound, upper_bound = jnp.array([-10.0]), jnp.array([10.0])
        ground_truth_best_y = 0.
        dim = 1
    else:
        raise ValueError("Dataset not recognised")
    initial_x = jax.random.uniform(rng_key, (initial_sample_num, dim), minval=lower_bound, maxval=upper_bound)
    initial_y = get_data_fn(initial_x)
    D = gpx.Dataset(X=initial_x, y=initial_y)
    D_init = D

    # GP prior
    mean = gpx.mean_functions.Zero()
    pairwise_distances = jnp.abs(initial_x[:, None, :] - initial_x[None, :, :])
    lengthscales = jnp.median(pairwise_distances, axis=(0, 1))
    # kernel = gpx.kernels.Matern52(n_dims=dim, lengthscale=lengthscales)
    kernel = gpx.kernels.RBF(n_dims=dim, lengthscale=lengthscales)
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)

    # BO loop
    nmse_list = []
    for iter in tqdm(range(bo_iters)):
        rng_key, _ = jax.random.split(rng_key)
        posterior = get_posterior(D, prior, rng_key)
        y_best = jnp.max(D.y)

        # Draw a sample from the posterior, and find the minimiser of it
        x_star = optimise_sample(args, rng_key, posterior, D, lower_bound, upper_bound, y_best, num_initial_sample_points=5)

        # if (iter + len(D_init.y)) % 10 == 0:
        # plot_bayes_opt(args, rng_key, get_data_fn, posterior, D, x_star, lower_bound, upper_bound,)

        # Evaluate the black-box function at the best point observed so far, and add it to the dataset
        y_star = get_data_fn(x_star[None, :])
        D = D + gpx.Dataset(X=x_star[None, :], y=y_star)

        nmse_list.append((jnp.max(D.y) - ground_truth_best_y) ** 2 / (jnp.max(D_init.y) - ground_truth_best_y) ** 2)

    plt.figure()
    plt.plot(nmse_list)
    plt.xlabel("Iterations")
    plt.ylabel("NMSE")
    plt.yscale("log")
    plt.savefig(args.save_path + f"bo_nmse.png")
    plt.close()

    jnp.save(args.save_path + f"bo_nmse.npy", jnp.array(nmse_list))
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/bo/{args.datasets}/'
    args.save_path += f"{args.utility}__dim_{args.dim}__kernel_{args.kernel}__seed_{args.seed}/"
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