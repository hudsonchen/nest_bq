from typing import List,Tuple 
import jax
from jax import config
import jax.numpy as jnp
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

from utils.kernel_means import *

cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


def get_config():
    parser = argparse.ArgumentParser(description='Toy example')
    # Args settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    args = parser.parse_args()
    return args

def standardised_forrester(x):
    mean = 0.45321
    std = 4.4258
    return ((6 * x - 2) ** 2 * jnp.sin(12 * x - 4) - mean) / std

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
        train_data=data
    )

    return posterior


def negative_expected_decrease_kernel_quadrature(x, sampler, f_best, num_samples, rng_key):
    # Sample from the posterior distribution at x
    dist = sampler(x)
    mu, var = dist.mean(), dist.variance()
    samples = dist.sample(rng_key, [num_samples])
    decreases = jnp.maximum(samples - f_best, 0)
    # ed = jnp.mean(decreases)
    ed = KQ_RBF_Gaussian(rng_key, x, decreases, mu, var)
    return ed.squeeze()


def negative_expected_decrease_monte_carlo(x, sampler, f_best, num_samples, rng_key):
    # Sample from the posterior distribution at x
    dist = sampler(x)
    samples = dist.sample(rng_key, [num_samples])
    decreases = jnp.maximum(samples - f_best, 0)
    ed = jnp.mean(decreases)
    return ed.squeeze()

def negative_expected_decrease_closed_form(x, sampler, f_best, num_samples, rng_key):
    # Get the mean (mu) and standard deviation (sigma) from the sampler
    dist = sampler(x)
    mu = dist.mean()
    sigma = dist.stddev()

    # Compute the improvement and the standard normal terms
    z = (mu - f_best) / sigma
    ed = (mu - f_best) * jax.scipy.stats.norm.cdf(z) + sigma * jax.scipy.stats.norm.pdf(z)
    return ed.squeeze()


def optimise_sample(sampler, rng_key, lower_bound, upper_bound, f_best, num_initial_sample_points):
    initial_sample_points = jax.random.uniform(
        rng_key,
        shape=(num_initial_sample_points, lower_bound.shape[0]),
        dtype=jnp.float64,
        minval=lower_bound,
        maxval=upper_bound,
    )
    rng_key, _ = jax.random.split(rng_key)
    initial_sample_y = sampler(initial_sample_points).sample(rng_key, [1])
    best_x = jnp.array([initial_sample_points[jnp.argmin(initial_sample_y)]])

    num_samples = 100

    dist = sampler(best_x)
    mu, var = dist.mean(), dist.variance()
    samples = dist.sample(rng_key, [num_samples])
    decreases = jnp.maximum(samples - f_best, 0)
    # ed = jnp.mean(decreases)
    ed = KQ_RBF_Gaussian(rng_key, best_x, decreases.squeeze(), mu, var[:,None])

    # We want to maximise the utility function, but the optimiser performs minimisation. Since we're minimising the sample drawn, the sample is actually the negative utility function.
    # Thompson sampling
    # negative_utility_fn = lambda x: sampler(x).sample(rng_key, [1]).squeeze()
    # Expected Improvement with closed form
    # negative_utility_fn = lambda x: negative_expected_decrease_closed_form(x, sampler, f_best, num_samples, rng_key)
    # Expected Improvement with Monte Carlo
    negative_utility_fn = lambda x: negative_expected_decrease_monte_carlo(x, sampler, f_best, num_samples, rng_key)

    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=negative_utility_fn, method="l-bfgs-b")
    bounds = (lower_bound, upper_bound)
    x_star = lbfgsb.run(best_x, bounds=bounds).params
    return x_star

def plot_bayes_opt(
    args,
    rng_key,
    posterior: gpx.gps.AbstractPosterior,
    sampler,
    dataset: gpx.Dataset,
    queried_x,
) -> None:
    rng_key, _ = jax.random.split(rng_key)
    plt_x = jnp.linspace(0, 1, 1000).reshape(-1, 1)
    forrester_y = standardised_forrester(plt_x)
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
        forrester_y,
        label="Forrester Function",
        color=cols[0],
        linestyle="--",
        linewidth=2,
    )
    ax.axvline(x=0.757, linestyle=":", color=cols[3], label="True Optimum")
    ax.scatter(dataset.X, dataset.y, label="Observations", color=cols[2], zorder=2)
    ax.scatter(
        queried_x,
        sampler(queried_x).sample(rng_key, [1]).squeeze(),
        label="Qeury Point",
        marker="*",
        color=cols[3],
        zorder=3,
    )
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    plt.tight_layout()
    plt.savefig(args.save_path + f"bo_{len(dataset.X)}.png")
    plt.show()


def main(args):
    bo_iters = 30
    initial_sample_num = 5
    lower_bound = jnp.array([0.0])
    upper_bound = jnp.array([1.0])
    rng_key = jax.random.PRNGKey(0)

    # Set up initial dataset
    initial_x = jax.random.uniform(rng_key, shape=(initial_sample_num, 1), minval=lower_bound, maxval=upper_bound)
    initial_y = standardised_forrester(initial_x)
    D = gpx.Dataset(X=initial_x, y=initial_y)

    # GP prior
    mean = gpx.mean_functions.Zero()
    kernel = gpx.kernels.Matern52(n_dims=1)
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)

    for _ in range(bo_iters):
        rng_key, _ = jax.random.split(rng_key)
        posterior = get_posterior(D, prior, rng_key)
        f_best = jnp.min(D.y)

        # Draw a sample from the posterior, and find the minimiser of it
        posterior_sampler = partial(posterior.predict, train_data = D)
        x_star = optimise_sample(posterior_sampler, rng_key, lower_bound, upper_bound, f_best, num_initial_sample_points=100)

        plot_bayes_opt(args, rng_key, posterior, posterior_sampler, D, x_star)

        # Evaluate the black-box function at the best point observed so far, and add it to the dataset
        y_star = standardised_forrester(x_star)
        print(f"Queried Point: {x_star}, Black-Box Function Value: {y_star}")
        D = D + gpx.Dataset(X=x_star, y=y_star)



def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/bo/'
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == "__main__":
    args = get_config()
    args = create_dir(args)
    main(args)