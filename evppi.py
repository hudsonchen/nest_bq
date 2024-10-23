import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import argparse
import os
import pwd
import pickle
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--multi_level', action='store_true', default=False)
    parser.add_argument('--eps', type=float, default=0.01)
    args = parser.parse_args()
    return args


def f1(theta, x):
    """
    Compute f1.

    Args:
        theta: (T, )
        x: (T, N, 9)

    Returns:
        f(x, theta): (T, N)
    """
    lamba_ = 1e4
    # lamba_ = 1.0
    # lambda * (Theta_5 * Theta_6 * Theta_7 + Theta_8 * Theta_9 * Theta_10) - (Theta_1 + Theta_2 * Theta_3 * Theta_4)
    # Theta_5 is theta
    # Theta_1 is x[0], Theta_2 is x[1], Theta_3 is x[2], Theta_4 is x[3], Theta_6 is x[4], Theta_7 is x[5], Theta_8 is x[6],
    # Theta_9 is x[7], Theta_10 is x[8],
    return lamba_ * (theta * x[:, :, 4] * x[:, :, 5] + x[:, :, 6] * x[:, :, 7] * x[:, :, 8]) - \
           (x[:, :, 0] + x[:, :, 1] * x[:, :, 2] * x[:, :, 3])


def f2(theta, x):
    """
    Compute f2.

    Args:
        theta: (T, )
        x: (T, N, 9)

    Returns:
        f(x, theta): (T, N)
    """
    lamba_ = 1e4
    # lamba_ = 1.0
    # lambda * (Theta_5 * Theta_6 * Theta_7 + Theta_8 * Theta_9 * Theta_10) - (Theta_1 + Theta_2 * Theta_3 * Theta_4)
    # Theta_14 is theta
    # Theta_4 is x[0], Theta_11 is x[1], Theta_12 is x[2], Theta_13 is x[3], Theta_15 is x[4], Theta_16 is x[5],
    # Theta_17 is x[6], Theta_18 is x[7], Theta_19 is x[8]
    return lamba_ * (theta * x[:, :, 4] * x[:, :, 5] + x[:, :, 6] * x[:, :, 7] * x[:, :, 8]) - \
           (x[:, :, 1] + x[:, :, 2] * x[:, :, 3] * x[:, :, 0])


def conditional_distribution(joint_mean, joint_covariance, theta, dimensions_x, dimensions_theta):
    """
    Compute conditional distribution p(x | theta).

    Args:
        joint_mean: (19,)
        joint_covariance: (19, 19)
        theta: (N, len(dimensions_theta))
        dimensions_x: list
        dimensions_theta: list

    Returns:
        mean_x_given_theta: shape (N, len(dimensions_x))
        cov_x_given_theta: shape (N, len(dimensions_x), len(dimensions_x))
    """
    dimensions_x = jnp.array(dimensions_x)
    dimensions_theta = jnp.array(dimensions_theta)

    mean_theta = jnp.take(joint_mean, dimensions_theta)[:, None]
    mean_x = jnp.take(joint_mean, dimensions_x)[:, None]

    # Create a grid of indices from A and B using meshgrid
    cov_ThetaTheta = joint_covariance[jnp.ix_(dimensions_theta, dimensions_theta)]
    cov_XX = joint_covariance[jnp.ix_(dimensions_x, dimensions_x)]
    cov_XTheta = joint_covariance[jnp.ix_(dimensions_x, dimensions_theta)]
    cov_ThetaX = joint_covariance[jnp.ix_(dimensions_theta, dimensions_x)]

    mean_x_given_theta = mean_x + cov_XTheta @ jnp.linalg.inv(cov_ThetaTheta) @ (theta.T - mean_theta)
    cov_x_given_theta = cov_XX - cov_XTheta @ jnp.linalg.inv(cov_ThetaTheta) @ cov_ThetaX
    return mean_x_given_theta.T, cov_x_given_theta



def sample_theta(T, rng_key):
    T = int(T)
    rng_key, _ = jax.random.split(rng_key)
    Theta_mean = jnp.array([0.7, 0.8])
    Theta_sigma = jnp.array([[0.01, 0.01 * 0.6], [0.01 * 0.6, 0.01]])
    u = jax.random.multivariate_normal(rng_key,
                                       mean=jnp.zeros_like(Theta_mean),
                                       cov=jnp.eye(Theta_sigma.shape[0]),
                                       shape=(T,))
    Theta = u @ jnp.linalg.cholesky(Theta_sigma) + Theta_mean
    Theta1 = Theta[:, 0][:, None]
    Theta2 = Theta[:, 1][:, None]
    return Theta1, Theta2, u


def sample_x_theta(N, Theta1, Theta2, rng_key):
    rng_key, _ = jax.random.split(rng_key)

    ThetaX_mean = jnp.array([1000., 0.1, 5.2, 400., 0.7,
                         0.3, 3.0, 0.25, -0.1, 0.5,
                         1500, 0.08, 6.1, 0.8, 0.3,
                         3.0, 0.2, -0.1, 0.5])
    ThetaX_sigma = jnp.array([1.0, 0.02, 1.0, 200, 0.1,
                          0.1, 0.5, 0.1, 0.02, 0.2,
                          1.0, 0.02, 1.0, 0.1, 0.05,
                          1.0, 0.05, 0.02, 0.2])
    ThetaX_sigma = jnp.diag(ThetaX_sigma ** 2)
    ThetaX_sigma = ThetaX_sigma.at[4, 6].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[6, 6]))
    ThetaX_sigma = ThetaX_sigma.at[6, 4].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[6, 6]))
    ThetaX_sigma = ThetaX_sigma.at[4, 13].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[13, 13]))
    ThetaX_sigma = ThetaX_sigma.at[13, 4].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[13, 13]))
    ThetaX_sigma = ThetaX_sigma.at[4, 15].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[15, 4].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[6, 13].set(0.6 * jnp.sqrt(ThetaX_sigma[6, 6]) * jnp.sqrt(ThetaX_sigma[13, 13]))
    ThetaX_sigma = ThetaX_sigma.at[13, 6].set(0.6 * jnp.sqrt(ThetaX_sigma[6, 6]) * jnp.sqrt(ThetaX_sigma[13, 13]))
    ThetaX_sigma = ThetaX_sigma.at[6, 15].set(0.6 * jnp.sqrt(ThetaX_sigma[6, 6]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[15, 6].set(0.6 * jnp.sqrt(ThetaX_sigma[6, 6]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[13, 15].set(0.6 * jnp.sqrt(ThetaX_sigma[13, 13]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[15, 13].set(0.6 * jnp.sqrt(ThetaX_sigma[13, 13]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    
    f1_cond_dist_fn = partial(conditional_distribution, joint_mean=ThetaX_mean, joint_covariance=ThetaX_sigma,
                              dimensions_theta=[4], dimensions_x=[0, 1, 2, 3, 5, 6, 7, 8, 9])

    mean_1, sigma_1 = f1_cond_dist_fn(theta=Theta1)
    T = Theta1.shape[0]
    u1 = jnp.zeros([T, N, 9]) + 0.0
    x1 = jnp.zeros([T, N, 9]) + 0.0
    for i in range(T):
        rng_key, _ = jax.random.split(rng_key)
        u1_temp = jax.random.multivariate_normal(rng_key,
                                                mean=jnp.zeros_like(mean_1[i, :]),
                                                cov=jnp.eye(sigma_1.shape[0]),
                                                shape=(N,))
        L1 = jnp.linalg.cholesky(sigma_1)
        u1 = u1.at[i, :, :].set(u1_temp)
        x1 = x1.at[i, :, :].set(u1_temp @ L1 + mean_1[i, :])

    f2_cond_dist_fn = partial(conditional_distribution, joint_mean=ThetaX_mean, joint_covariance=ThetaX_sigma,
                              dimensions_theta=[13], dimensions_x=[3, 10, 11, 12, 14, 15, 16, 17, 18])
    mean_2, sigma_2 = f2_cond_dist_fn(theta=Theta2)
    u2 = jnp.zeros([T, N, 9]) + 0.0
    x2 = jnp.zeros([T, N, 9]) + 0.0
    for i in range(T):
        rng_key, _ = jax.random.split(rng_key)
        u2_temp = jax.random.multivariate_normal(rng_key,
                                                mean=jnp.zeros_like(mean_2[i, :]),
                                                cov=jnp.eye(sigma_2.shape[0]),
                                                shape=(N,))
        L2 = jnp.linalg.cholesky(sigma_2)
        u2 = u2.at[i, :, :].set(u2_temp)
        x2 = x2.at[i, :, :].set(u2_temp @ L2 + mean_2[i, :])
    return u1, x1, u2, x2


def sample(T, N, rng_key):
    rng_key, _ = jax.random.split(rng_key)
    Theta1, Theta2, u = sample_theta(T, rng_key)
    u1, x1, u2, x2 = sample_x_theta(N, Theta1, Theta2, rng_key) 
    return Theta1, Theta2, u, u1, x1, u2, x2


def nested_monte_carlo(Theta1, Theta2, u, u1, x1, u2, x2):
    f1_val, f2_val = f1(Theta1, x1), f2(Theta2, x2)
    f1_val_mean, f2_val_mean = jnp.mean(f1_val, axis=1), jnp.mean(f2_val, axis=1)
    I = np.mean(jnp.maximum(f1_val_mean, f2_val_mean))
    return I


def nested_kernel_quadrature(Theta1, Theta2, u, u1, x1, u2, x2):
    # Debug code
    # rng_key, _ = jax.random.split(rng_key)
    # Theta1_debug, Theta2_debug, u_debug = sample_theta(100, rng_key)
    # u1_debug, x1_debug, u2_debug, x2_debug = sample_x_theta(100, Theta1_debug, Theta2_debug, rng_key)
    # f1_val_debug, f2_val_debug = f1(Theta1_debug, x1_debug), f2(Theta2_debug, x2_debug)
    # f1_val_mean_debug, f2_val_mean_debug = jnp.mean(f1_val_debug, axis=1), jnp.mean(f2_val_debug, axis=1)
    # I_debug = np.mean(jnp.maximum(f1_val_mean_debug, f2_val_mean_debug))
    #
    f1_val, f2_val = f1(Theta1, x1), f2(Theta2, x2)
    scale_1, shift_1, scale_2, shift_2 = f1_val.std(), f1_val.mean(), f2_val.std(), f2_val.mean()
    f1_val_normalized = (f1_val - shift_1) / scale_1
    f2_val_normalized = (f2_val - shift_2) / scale_2
    f1_val_kq = KQ_Matern_12_Gaussian_Vectorized(u1, f1_val_normalized) 
    f2_val_kq = KQ_Matern_12_Gaussian_Vectorized(u2, f2_val_normalized)
    f1_val_kq_, f2_val_kq_ = f1_val_kq * scale_1 + shift_1, f2_val_kq * scale_2 + shift_2
    f_max = jnp.maximum(f1_val_kq_, f2_val_kq_)
    scale, shift = f_max.std(), f_max.mean()
    f_max_normalized = (f_max - shift) / scale
    I = KQ_Matern_12_Gaussian(u, f_max_normalized)
    I = I * scale + shift
    return I


def nested_kernel_quadrature_multi_level(Theta1, Theta2, u, u1_prev, x1_prev, u2_prev, x2_prev,
                                         u1, x1, u2, x2):
    # Debug code
    # rng_key, _ = jax.random.split(rng_key)
    # Theta1_debug, Theta2_debug, u_debug = sample_theta(100, rng_key)
    # u1_debug, x1_debug, u2_debug, x2_debug = sample_x_theta(100, Theta1_debug, Theta2_debug, rng_key)
    # f1_val_debug, f2_val_debug = f1(Theta1_debug, x1_debug), f2(Theta2_debug, x2_debug)
    # f1_val_mean_debug, f2_val_mean_debug = jnp.mean(f1_val_debug, axis=1), jnp.mean(f2_val_debug, axis=1)
    # I_debug = np.mean(jnp.maximum(f1_val_mean_debug, f2_val_mean_debug))
    #
    f1_val, f2_val = f1(Theta1, x1), f2(Theta2, x2)
    scale_1, shift_1, scale_2, shift_2 = f1_val.std(), f1_val.mean(), f2_val.std(), f2_val.mean()
    f1_val_normalized = (f1_val - shift_1) / scale_1
    f2_val_normalized = (f2_val - shift_2) / scale_2
    f1_val_kq = KQ_Matern_12_Gaussian_Vectorized(u1, f1_val_normalized) 
    f2_val_kq = KQ_Matern_12_Gaussian_Vectorized(u2, f2_val_normalized)
    f1_val_kq_, f2_val_kq_ = f1_val_kq * scale_1 + shift_1, f2_val_kq * scale_2 + shift_2
    f_max = jnp.maximum(f1_val_kq_, f2_val_kq_)

    f1_val_prev, f2_val_prev = f1(Theta1, x1_prev), f2(Theta2, x2_prev)
    scale_1_prev, shift_1_prev, scale_2_prev, shift_2_prev = f1_val_prev.std(), f1_val_prev.mean(), f2_val_prev.std(), f2_val_prev.mean()
    f1_val_normalized_prev = (f1_val_prev - shift_1_prev) / scale_1_prev
    f2_val_normalized_prev = (f2_val_prev - shift_2_prev) / scale_2_prev
    f1_val_kq_prev = KQ_Matern_12_Gaussian_Vectorized(u1_prev, f1_val_normalized_prev)
    f2_val_kq_prev = KQ_Matern_12_Gaussian_Vectorized(u2_prev, f2_val_normalized_prev)
    f1_val_kq_prev_, f2_val_kq_prev_ = f1_val_kq_prev * scale_1_prev + shift_1_prev, f2_val_kq_prev * scale_2_prev + shift_2_prev
    f_max_prev = jnp.maximum(f1_val_kq_prev_, f2_val_kq_prev_)

    f_difference = f_max - f_max_prev
    scale, shift = f_difference.std(), f_difference.mean()
    f_difference_normalized = (f_difference - shift) / scale
    I = KQ_Matern_12_Gaussian(u, f_difference_normalized)
    I = I * scale + shift
    return I


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
            Theta1, Theta2, u, u1, x1, u2, x2 = sample(T, N, rng_key)

            if use_kq:
                rng_key, _ = jax.random.split(rng_key)
                Y = nested_kernel_quadrature(Theta1, Theta2, u, u1, x1, u2, x2)
                Yl = Yl.at[l].set(Y)
            else:
                rng_key, _ = jax.random.split(rng_key)
                Y = nested_monte_carlo(Theta1, Theta2, u, u1, x1, u2, x2)
                Yl = Yl.at[l].set(Y)
            Cl = Cl.at[l].set(N * T)
        else:
            N, N_prev, T = int(Nl[l]), int(Nl[l-1]), int(Tl[l]) + 1
            Theta1, Theta2, u, u1, x1, u2, x2 = sample(T, N, rng_key)
            u1_prev, x1_prev, u2_prev, x2_prev = u1[:, :N_prev, :], x1[:, :N_prev, :], u2[:, :N_prev, :], x2[:, :N_prev, :]

            if use_kq:
                rng_key, _ = jax.random.split(rng_key)
                Y_diff = nested_kernel_quadrature_multi_level(Theta1, Theta2, u, u1_prev, x1_prev, u2_prev, x2_prev, u1, x1, u2, x2)
                Yl = Yl.at[l].set(Y_diff)
            else:
                rng_key, _ = jax.random.split(rng_key)
                Y_prev = nested_monte_carlo(Theta1, Theta2, u, u1_prev, x1_prev, u2_prev, x2_prev)
                Y = nested_monte_carlo(Theta1, Theta2, u, u1, x1, u2, x2)
                Yl = Yl.at[l].set(Y - Y_prev)
            Cl = Cl.at[l].set(N * T)

    # Final estimation
    P = Yl.sum()
    C = Cl.sum()
    return P, C


def run(args):
    rng_key = jax.random.PRNGKey(args.seed)
    # T_large = N_large = 1000
    # rng_key, _ = jax.random.split(rng_key)
    # Theta_large1, Theta_large2, _ = sample_theta(T_large, rng_key)
    # u1_large, x1_large, u2_large, x2_large = sample_x_theta(N_large, Theta_large1, Theta_large2, rng_key)
    # f1_X_large = f1(Theta_large1, x1_large)
    # f2_X_large = f2(Theta_large2, x2_large)
    # ground_truth_1 = f1_X_large.mean(1)
    # ground_truth_2 = f2_X_large.mean(1)
    # true_value = jnp.maximum(ground_truth_1, ground_truth_2).mean()
    # True value is around 6035.58
    true_value = 6035.58
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

        use_kq = True
        I_MLMC_nkq, cost = mlmc(args.eps, N0, L, use_kq, rng_key)
        print(f"MLMC KQ: {I_MLMC_nkq} with cost {cost}")
        I_kq_err_dict[f'cost_{cost}'] = jnp.abs(I_MLMC_nkq - true_value)
    else:
        N0 = 1
        N_total = int((1. / args.eps) * N0)
        T_total = int((1. / args.eps) * N0)
        I_nmc = nested_monte_carlo(T_total, N_total, rng_key)
        cost = N_total * T_total
        print(f"NMC: {I_nmc} with cost {cost}")
        I_mc_err_dict[f'cost_{cost}'] = jnp.abs(I_nmc - true_value)

        N_total = int((1. / args.eps) * N0)
        T_total = int((1. / args.eps) * N0)
        I_nkq = nested_kernel_quadrature(T_total, N_total, rng_key)
        print(f"NKQ: {I_nkq} with cost {cost}")
        I_kq_err_dict[f'cost_{cost}'] = jnp.abs(I_nkq - true_value)

    
    with open(f"{args.save_path}/seed_{args.seed}_MC", 'wb') as file:
        pickle.dump(I_mc_err_dict, file)
    with open(f"{args.save_path}/seed_{args.seed}_KQ", 'wb') as file:
        pickle.dump(I_kq_err_dict, file)
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/evppi/'
    args.save_path += f"multi_level_{args.multi_level}__eps_{args.eps}__seed_{args.seed}"
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