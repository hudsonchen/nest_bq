import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model, fit_gpytorch_mll
from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel
from gpytorch.kernels import ScaleKernel
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from botorch.test_functions import Ackley, DropWave, Branin, Cosine8
from botorch.utils.datasets import FixedNoiseDataset
from mlmcbo.acquisition_functions import qEIMLMCTwoStep
from mlmcbo.utils import optimize_mlmc
from BO_acqf import *
from utils.kernels import *
import time
import argparse
import warnings
from tqdm import tqdm
import shutil
import pwd
import os
import math
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import warnings
from botorch.exceptions import InputDataWarning, BadInitialCandidatesWarning

# Suppress specific warnings from BoTorch
warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)


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
plt.rcParams['axes.labelsize'] = 26
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')

# plt.rc('font', family='Arial', size=12)
plt.rc('axes', titlesize=26, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=26, frameon=False)
plt.rc('xtick', labelsize=22, direction='in')
plt.rc('ytick', labelsize=22, direction='in')

def get_config():
    parser = argparse.ArgumentParser(description='Bayesian Optimization')
    # Args settings
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--utility', type=str, default='lookahead_EI_kq')
    parser.add_argument('--datasets', type=str, default='ackley')
    parser.add_argument('--kernel', type=str, default='matern')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--reparam', type=str, default='uniform')
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    return args
    
def main(args):
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    # Initialization
    if args.datasets == 'ackley':
        ackley = Ackley(dim=args.dim)
        bounds = ackley.bounds.to(torch.float64)
        bounds /= 3
        train_x = draw_sobol_samples(bounds=bounds, n=1, q=2 * args.dim).squeeze(0)
        load_data = lambda x: -Ackley(dim=args.dim)(x)[:, None]
        train_y = load_data(train_x)
        ground_truth_best_y = 1.4019
        X_init, y_init = train_x, train_y
    elif args.datasets == 'dropwave':
        dropwave = DropWave()
        bounds = dropwave.bounds.to(torch.float64)
        train_x = draw_sobol_samples(bounds=bounds, n=1, q=2 * args.dim).squeeze(0)
        load_data = lambda x: -DropWave()(x)[:, None]
        train_y = load_data(train_x)
        ground_truth_best_y = 1.0
        X_init, y_init = train_x, train_y
    elif args.datasets == 'cosine8':
        cosine8 = Cosine8()
        bounds = cosine8.bounds.to(torch.float64)
        train_x = draw_sobol_samples(bounds=bounds, n=1, q=2 * args.dim).squeeze(0)
        load_data = lambda x: Cosine8()(x)[:, None]
        train_y = load_data(train_x)
        ground_truth_best_y = 0.8
        X_init, y_init = train_x, train_y
        args.dim = 8
    else:
        pass
    
    rewards = [train_y.max().item()]  # Track maximum reward
    num_samples = int(1. / args.delta)

    mc_weights = torch.ones([num_samples]) / num_samples
    jitter = num_samples ** (-1. / 2.)
    if args.kernel == 'matern':
        if args.reparam == 'uniform':
            u_uniform = torch.rand(num_samples)
            kq_weights_1 = kme_Matern_12_Uniform_1d(0., 1., 1., u_uniform.numpy()) 
            kq_weights_2 = np.linalg.inv(my_Matern_12_product(u_uniform.numpy()[:, None], u_uniform.numpy()[:, None], np.ones([1])) + jitter * np.eye(num_samples))
            u_gaussian = torch.distributions.Normal(0, 1).icdf(u_uniform)
        elif args.reparam == 'gaussian':
            u_gaussian = torch.randn(num_samples)
            kq_weights_1 = kme_Matern_12_Gaussian_1d(1., u_gaussian.numpy())
            kq_weights_2 = np.linalg.inv(my_Matern_12_product(u_gaussian.numpy()[:, None], u_gaussian.numpy()[:, None], np.ones([1])) + jitter * np.eye(num_samples))
        else:
            pass
    elif args.kernel == 'rbf':
        if args.reparam == 'uniform':
            u_uniform = torch.rand(num_samples)
            kq_weights_1 = kme_RBF_uniform(0., 1., 1., u_uniform.numpy())
            kq_weights_2 = np.linalg.inv(my_RBF(u_uniform.numpy()[:, None], u_uniform.numpy()[:, None], np.ones([1])) + jitter * np.eye(num_samples))
            u_gaussian = torch.distributions.Normal(0, 1).icdf(u_uniform)
        elif args.reparam == 'gaussian':
            u_gaussian = torch.randn(num_samples)
            kq_weights_1 = kme_RBF_Gaussian(jnp.zeros([1]), jnp.ones([1]), 1.0, u_gaussian.numpy())
            kq_weights_2 = np.linalg.inv(my_RBF(u_gaussian.numpy()[:, None], u_gaussian.numpy()[:, None], np.ones([1])) + jitter * np.eye(num_samples))
        else:
            pass
    kq_weights = torch.from_numpy(np.array(kq_weights_1 @ kq_weights_2))
    # kq_weights = kq_weights / (kq_weights).sum()

    for i in range(args.iterations):
        # Fit GP model
        if 'mlmc' not in args.utility:
            model = SingleTaskGP(train_x, train_y, covar_module=ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1])))
            # model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
        else:
            train_yvar = torch.full_like(train_y, 1e-4)
            model = FixedNoiseGP(train_x,
                                train_y,
                                train_yvar,
                                )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

        # Define acquisition function and optimize
        acqf_kq = my_lookahead_EI(model=model, args=args, num_samples=num_samples, num_fantasies=args.q,
                                   bounds=bounds, u=u_gaussian, weights=mc_weights, y_best=train_y.max())
        acqf_mc = my_lookahead_EI(model=model, args=args, num_samples=num_samples, num_fantasies=args.q,
                                   bounds=bounds, u=u_gaussian, weights=kq_weights, y_best=train_y.max())
        qEI = qEIMLMCTwoStep(
                model=model,
                bounds=bounds,
                num_restarts=10,
                raw_samples=20,
                q=2,
                batch_sizes=[2]
            )
        candidate_mlmc, _, _ = optimize_mlmc(inc_function=qEI,
                                eps=0.1,
                                dl=3,
                                alpha=1,
                                beta=1.5,
                                gamma=1,
                                meanc=1,
                                varc=1,
                                var0=1,
                                match_mode='point',
                                )
        candidate_mlmc = candidate_mlmc[:args.q, :]
        candidate_mlmc = torch.clamp(candidate_mlmc, min=bounds[0, :], max=bounds[1, :])

        candidate_kq, _ = optimize_acqf(
            acq_function=acqf_kq, bounds=bounds, q=args.q * 2, num_restarts=10, raw_samples=20
        )
        candidate_kq = candidate_kq[:args.q, :]
        candidate_kq = torch.clamp(candidate_kq, min=bounds[0, :], max=bounds[1, :])

        candidate_mc, _ = optimize_acqf(
            acq_function=acqf_mc, bounds=bounds, q=args.q * 2, num_restarts=10, raw_samples=20
        )
        candidate_mc = candidate_mc[:args.q, :]
        candidate_mc = torch.clamp(candidate_mc, min=bounds[0, :], max=bounds[1, :])

        # Plot the landscape of utility
        num_samples_truth = 10000
        mc_weights_truth = torch.ones([num_samples_truth]) / num_samples_truth
        u_gaussian_truth = torch.randn(num_samples_truth)
        acqf_truth = my_lookahead_EI(model=model, args=args, num_samples=num_samples_truth, 
                                     num_fantasies=args.q, bounds=bounds, u=u_gaussian_truth, 
                                     weights=mc_weights_truth, y_best=train_y.max())
        x1_min, x2_min = bounds[0, :2] 
        x1_max, x2_max = bounds[1, :2]
        grid_size = 100
        x1_lin = torch.linspace(x1_min, x1_max, grid_size)
        x2_lin = torch.linspace(x2_min, x2_max, grid_size)
        X1, X2 = torch.meshgrid(x1_lin, x2_lin, indexing="xy")
        grid_points = torch.stack((X1.reshape(-1), X2.reshape(-1)), dim=-1)
        if args.dim == 2:
            grid_points_whole = grid_points
        else:
            grid_points_rest = torch.stack([
                            torch.linspace(bounds[0, i], bounds[1, i], steps=grid_points.shape[0])
                            for i in range(2, args.dim)
                        ]).T
            grid_points_whole = torch.cat([grid_points, grid_points_rest], dim=1)
        utility_truth_grid = acqf_truth(grid_points_whole[:, None, :].repeat(1, args.q * 2, 1))
        utility_kq_grid = acqf_kq(grid_points_whole[:, None, :].repeat(1, args.q * 2, 1))
        utility_mc_grid = acqf_mc(grid_points_whole[:, None, :].repeat(1, args.q * 2, 1))

        candidate_kq_optimal = grid_points[utility_kq_grid.reshape(-1).argmax().item(), :][None, :]
        candidate_mc_optimal = grid_points[utility_mc_grid.reshape(-1).argmax().item(), :][None, :]
        candidate_true_optimal = grid_points[utility_truth_grid.reshape(-1).argmax().item(), :][None, :]
        plt.figure(figsize=(28, 6.5))
        plt.subplot(1, 3, 1)
        plt.imshow(utility_truth_grid.reshape(grid_size, grid_size).detach().numpy(),
                   extent=(x1_min, x1_max, x2_min, x2_max), origin="lower", cmap="viridis",
                   norm=matplotlib.colors.LogNorm(vmin=utility_truth_grid.min().item() + 1e-6, 
                                vmax=utility_truth_grid.max().item() + 1e-6))
        plt.title("Utility Landscape (Ground Truth)")
        plt.xlabel(f"Best candidate location=[{candidate_true_optimal[0, 0].item():.4f}, {candidate_true_optimal[0, 1].item():.4f}]")
        plt.colorbar()
        plt.scatter(candidate_kq_optimal[0, 0], candidate_kq_optimal[0, 1], color='red', marker='x',
            s=200, linewidths=3, label='NKQ')
        plt.scatter(candidate_mc_optimal[0, 0], candidate_mc_optimal[0, 1], color='blue', marker='o',
                    s=200, linewidths=3, label='NMC')
        plt.scatter(candidate_mlmc[0, 0], candidate_mlmc[0, 1], color='green', marker='^',
                    s=200, linewidths=3, label='MLMC')
        plt.legend(ncol=3, bbox_to_anchor=(1.4, 1.05), loc='lower left')
        plt.subplot(1, 3, 2)
        plt.imshow(utility_kq_grid.reshape(grid_size, grid_size).detach().numpy(),
                   extent=(x1_min, x1_max, x2_min, x2_max), origin="lower", cmap="viridis",
                   norm=matplotlib.colors.LogNorm(vmin=utility_kq_grid.min().item() + 1e-6, 
                                vmax=utility_kq_grid.max().item() + 1e-6))
        plt.title("Utility Landscape (NKQ)")
        plt.xlabel(f"NKQ candidate location=[{candidate_kq_optimal[0, 0].item():.4f}, {candidate_kq_optimal[0, 1].item():.4f}]")
        plt.colorbar()
        plt.scatter(candidate_kq_optimal[0, 0], candidate_kq_optimal[0, 1], color='red', marker='x',
                    s=200, linewidths=3, label='NKQ')
    
        plt.subplot(1, 3, 3)
        plt.imshow(utility_mc_grid.reshape(grid_size, grid_size).detach().numpy(),
                     extent=(x1_min, x1_max, x2_min, x2_max), origin="lower", cmap="viridis",
                     norm=matplotlib.colors.LogNorm(vmin=utility_mc_grid.min().item() + 1e-6, 
                                  vmax=utility_mc_grid.max().item()+ 1e-6))
        plt.colorbar()
        plt.title("Utility Landscape (NMC)")
        plt.xlabel(f"NMC candidate location=[{candidate_mc_optimal[0, 0].item():.4f}, {candidate_mc_optimal[0, 1].item():.4f}]")
        plt.scatter(candidate_mc_optimal[0, 0], candidate_mc_optimal[0, 1], color='blue', marker='o',
                    s=200, linewidths=3, label='NMC')
        plt.savefig(f"{args.save_path}/utility_landscape_{i}.png", bbox_inches='tight')
            
        new_y = load_data(candidate_kq)
        rewards.append(max(rewards[-1], new_y.max().item()))   # Update max reward tracking
        # Update training data and re-fit
        train_x = torch.cat([train_x, candidate_kq])
        train_y = torch.cat([train_y, new_y])
        
        print(f"Iteration {i+1} finished")
    
    return

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/bo_plot_landscape/{args.datasets}/'
    args.save_path += f"{args.utility}__dim_{args.dim}__delta_{args.delta}__q_{args.q}__kernel_{args.kernel}__reparam_{args.reparam}__iter_{args.iterations}__seed_{args.seed}"
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
