import torch
import numpy as np
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


def get_config():
    parser = argparse.ArgumentParser(description='Toy example')
    # Args settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--utility', type=str)
    parser.add_argument('--datasets', type=str)
    parser.add_argument('--kernel', type=str, default='matern')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--reparam', type=str, default='uniform')
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    return args
    
def main(args):
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    jitter = 1e-10
    # jitter = 0.0

    # Initialization
    if args.datasets == 'ackley':
        ackley = Ackley(dim=args.dim)
        bounds = ackley.bounds.to(torch.float64)
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
    else:
        pass
    
    rewards = [train_y.max().item()]  # Track maximum reward
    nmse_list = [1.0]  # Track NMSE
    cost_list = [0.0]  # Track cost

    if 'mc' in args.utility:
        num_samples = int(args.delta ** (-2))
    elif 'kq' in args.utility:
        num_samples = int(args.delta ** (-1))
    else:
        pass

    mc_weights = torch.ones([num_samples]) / num_samples
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
        if args.utility == 'EI_mc': 
            acqf = my_EI(model=model, args=args, num_samples=num_samples, u=u_gaussian, weights=mc_weights, y_best=train_y.max())
        elif args.utility == 'EI_kq':
            acqf = my_EI(model=model, args=args, num_samples=num_samples, u=u_gaussian, weights=kq_weights, y_best=train_y.max())
        elif args.utility == 'EI':
            acqf = my_closed_form_EI(model=model, y_best=train_y.max())
        elif args.utility == 'lookahead_EI_mc':
            acqf = my_lookahead_EI(model=model, args=args, num_samples=num_samples, num_fantasies=args.q,
                                   bounds=bounds, u=u_gaussian, weights=mc_weights, y_best=train_y.max())
        elif args.utility == 'lookahead_EI_kq':
            acqf = my_lookahead_EI(model=model, args=args, num_samples=num_samples, num_fantasies=args.q,
                                   bounds=bounds, u=u_gaussian, weights=kq_weights, y_best=train_y.max())
        elif args.utility == 'lookahead_EI_mlmc':
            qEI = qEIMLMCTwoStep(
                    model=model,
                    bounds=bounds,
                    num_restarts=10,
                    raw_samples=20,
                    q=2,
                    batch_sizes=[2]
                )
        else:
            raise ValueError("Utility function not recognized")
        
        start_time = time.time()
        if args.utility == 'lookahead_EI_mlmc':
            candidate, _, _ = optimize_mlmc(inc_function=qEI,
                                    eps=args.delta,
                                    dl=3,
                                    alpha=1,
                                    beta=1.5,
                                    gamma=1,
                                    meanc=1,
                                    varc=1,
                                    var0=1,
                                    match_mode='point',
                                    )
        else:
            candidate, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=args.q * 2, num_restarts=10, raw_samples=20
            )
            candidate = candidate[:args.q, :]
        new_y = load_data(candidate)
        rewards.append(max(rewards[-1], new_y.max().item()))   # Update max reward tracking
        # Update training data and re-fit
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        
        nmse_list.append((torch.max(train_y).item() - ground_truth_best_y) ** 2 / (torch.max(y_init).item() - ground_truth_best_y) ** 2)
        cost = time.time() - start_time
        cost_list.append(cost)
        print(f"Iteration {i+1} finished, {args.utility} time {cost}")

        # Plot reward progression
        plt.figure(figsize=(12, 5))
        
        # Reward Plot
        plt.subplot(1, 3, 1)
        plt.plot(rewards, marker='o', color='b')
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Reward Progression")

        plt.subplot(1, 3, 2)
        plt.plot(nmse_list, marker='o', color='r')
        plt.xlabel("Iteration")
        plt.ylabel("NMSE")
        plt.title("NMSE Progression")

        # GP Posterior Plot
        plt.subplot(1, 3, 3)
        if args.dim == 1:
            x = torch.linspace(-5, 5, 100).unsqueeze(-1)
            model.eval()
            with torch.no_grad(), torch.enable_grad():
                posterior = model(x)
                mean = posterior.mean.detach().numpy()
                lower, upper = posterior.confidence_region()

            plt.plot(x.numpy(), mean, 'b-', label="Mean")
            plt.plot(x.numpy(), load_data(x).numpy(), 'k--', label="Objective")
            plt.fill_between(x.numpy().squeeze(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.2, color="b")
            plt.scatter(train_x.numpy(), train_y.numpy(), color="r", label="Samples")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("GP Posterior")
            plt.legend()
        plt.savefig(f"{args.save_path}/gp_visualization_{i}.png")
        plt.close()
    
    rewards = np.array(rewards)
    np.save(f"{args.save_path}/rewards.npy", rewards)
    nmse = np.array(nmse_list)
    np.save(f"{args.save_path}/nmse.npy", nmse)
    costs = np.cumsum(np.array(cost_list))
    np.save(f"{args.save_path}/costs.npy", costs)
    return

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/bo/{args.datasets}/'
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
    print(f"\nChanging save path from\n\n{args.save_path}\n\nto\n\n{args.save_path}__complete\n")
    import shutil
    if os.path.exists(f"{args.save_path}__complete"):
        shutil.rmtree(f"{args.save_path}__complete")
    os.rename(args.save_path, f"{args.save_path}__complete")
    print(f"Results saved at {args.save_path}")