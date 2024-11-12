import jax.numpy as jnp
import jax
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
    os.chdir("/home/zongchen/nest_bq")
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


# def g(x, theta):
#     return jnp.sqrt(2 / jnp.pi) * jnp.exp(-2 * (x - theta) ** 2)

# def f(x):
#     return jnp.log(x)

def f(x):
    return x ** 2

# def g(x, theta):
#     return (1 + jnp.sqrt(3) * jnp.abs(x - theta)) * jnp.exp(- jnp.sqrt(3) * jnp.abs(x - theta))

def g(x, theta):
    return (x ** 2.5).sum(-1) + (theta[:, None, :] ** 2.5).sum(-1)

def simulate_theta(T, d, rng_key):
    rng_key, _ = jax.random.split(rng_key)
    Theta = jax.random.uniform(rng_key, shape=(T, d), minval=0., maxval=1.)
    return Theta


def simulate_x_theta(N, d, Theta, rng_key):
    def simulate_x_per_theta(N, theta, rng_key):
        x = jax.random.uniform(rng_key, shape=(N, d), minval=0., maxval=1.)
        return x
    vmap_func = jax.vmap(simulate_x_per_theta, in_axes=(None, 0, 0))
    X = vmap_func(N, Theta, jax.random.split(rng_key, len(Theta)))
    return X


def nested_monte_carlo(X, Theta):
    g_X = g(X, Theta)
    I_theta_MC = g_X.mean(1)
    I_NMC = f(I_theta_MC).mean(0)
    return I_NMC


def nested_kernel_quadrature(x, theta):
    g_X = g(x, theta)
    a, b = 0, 1
    scale = 1.0
    T, d = x.shape[0], x.shape[2]
    I_theta_KQ = KQ_Matern_32_Uniform_Vectorized(x, g_X, a * jnp.ones([T, d]), b * jnp.ones([T, d]), scale) 
    f_I_theta_KQ = f(I_theta_KQ)
    I_NKQ = KQ_Matern_32_Uniform(theta, f_I_theta_KQ, a * jnp.ones([d]), b * jnp.ones([d]), scale)
    return I_NKQ


def nested_kernel_quadrature_multi_level(x, x_prev, theta):
    a, b = 0, 1
    scale = 1.0
    T, d = x.shape[0], x.shape[2]

    g_X = g(x, theta)
    I_theta_KQ = KQ_Matern_32_Uniform_Vectorized(x, g_X, a * jnp.ones([T, d]), b * jnp.ones([T, d]), scale)
    f_I_theta_KQ = f(I_theta_KQ)
    g_X_prev = g(x_prev, theta)
    I_theta_KQ_prev = KQ_Matern_32_Uniform_Vectorized(x_prev, g_X_prev, a * jnp.ones([T, d]), b * jnp.ones([T, d]), scale)
    f_I_theta_KQ_prev = f(I_theta_KQ_prev)

    f_difference = f_I_theta_KQ - f_I_theta_KQ_prev
    I_NKQ = KQ_Matern_32_Uniform(theta, f_difference, a * jnp.ones([d]), b * jnp.ones([d]), scale)
    return I_NKQ


def mlmc(eps, N0, L, use_kq, rng_key):
    d = 1
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
            Theta = simulate_theta(T, d, rng_key)
            X = simulate_x_theta(N, d, Theta, rng_key)

            if use_kq:
                rng_key, _ = jax.random.split(rng_key)
                Y = nested_kernel_quadrature(X, Theta)
                Yl = Yl.at[l].set(Y)
            else:
                rng_key, _ = jax.random.split(rng_key)
                Y = nested_monte_carlo(X, Theta)
                Yl = Yl.at[l].set(Y)
            Cl = Cl.at[l].set(N * T)
        else:
            N, N_prev, T = int(Nl[l]), int(Nl[l-1]), int(Tl[l]) + 1
            Theta = simulate_theta(T, d, rng_key)
            X = simulate_x_theta(N, d, Theta, rng_key)
            X_prev = X[:, :N_prev, :]

            if use_kq:
                rng_key, _ = jax.random.split(rng_key)
                Y_diff = nested_kernel_quadrature_multi_level(X, X_prev, Theta)
                Yl = Yl.at[l].set(Y_diff)
            else:
                rng_key, _ = jax.random.split(rng_key)
                Y_prev = nested_monte_carlo(X_prev, Theta)
                Y = nested_monte_carlo(X, Theta)
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
    true_value = (12. / 49.) + (1. / 6.)
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
        Theta = simulate_theta(T_total, 1, rng_key)
        X = simulate_x_theta(N_total, 1, Theta, rng_key)
        I_nmc = nested_monte_carlo(X, Theta)
        cost = N_total * T_total
        print(f"NMC: {I_nmc} with cost {cost}")
        I_mc_err_dict[f'cost_{cost}'] = jnp.abs(I_nmc - true_value)

        I_nkq = nested_kernel_quadrature(X, Theta)
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
    args.save_path += f'results/toy_mlmc/'
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