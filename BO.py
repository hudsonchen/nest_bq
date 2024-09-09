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

def optimise_sample(sampler, rng_key, lower_bound, upper_bound, num_initial_sample_points):
    initial_sample_points = jax.random.uniform(
        rng_key,
        shape=(num_initial_sample_points, lower_bound.shape[0]),
        dtype=jnp.float64,
        minval=lower_bound,
        maxval=upper_bound,
    )
    rng_key, _ = jax.random.split(rng_key)
    initial_sample_y = sampler(rng_key, initial_sample_points)
    best_x = jnp.array([initial_sample_points[jnp.argmin(initial_sample_y)]])

    # We want to maximise the utility function, but the optimiser performs minimisation. Since we're minimising the sample drawn, the sample is actually the negative utility function.
    negative_utility_fn = lambda x: sampler(x).sample(rng_key, [1])
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=negative_utility_fn, method="l-bfgs-b")
    bounds = (lower_bound, upper_bound)
    x_star = lbfgsb.run(best_x, bounds=bounds).params
    return x_star

def plot_bayes_opt(
    args,
    posterior: gpx.gps.AbstractPosterior,
    sample,
    dataset: gpx.Dataset,
    queried_x,
) -> None:
    plt_x = jnp.linspace(0, 1, 1000).reshape(-1, 1)
    forrester_y = standardised_forrester(plt_x)
    sample_y = sample(plt_x)

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
        sample(queried_x),
        label="Posterior Sample Optimum",
        marker="*",
        color=cols[3],
        zorder=3,
    )
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    plt.savefig(args.save_path + f"bo_{len(dataset.X)}.png")
    plt.show()


def main(args):
    bo_iters = 5
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

        # Draw a sample from the posterior, and find the minimiser of it
        posterior_sampler = partial(posterior.predict, train_data = D)
        x_star = optimise_sample(posterior_sampler, rng_key, lower_bound, upper_bound, num_initial_sample_points=100)

        plot_bayes_opt(args, posterior, posterior_sampler, D, x_star)

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