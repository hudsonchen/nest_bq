import torch
import jax
import jax.numpy as jnp
import numpy as np
from botorch.test_functions import Ackley, DropWave
from botorch.utils.datasets import FixedNoiseDataset


def emulator(x):
    mean = 0.45321
    std = 4.4258
    # return ((6 * x - 2) ** 2 * jnp.sin(12 * x - 4) - mean) / std
    return jnp.exp(-(x - 2)**2) + jnp.exp(-((x - 6)**2) / 10) + 1 / (x**2 + 1)


def load_ackley(x, dim):
    """
    Generates a dataset using the Ackley function and converts the dataset into JAX NumPy arrays.
    """
    x_torch = torch.tensor(np.array(x)) 
    ackley = Ackley(dim=dim) 
    y_torch = -ackley(x_torch) # Take the negative of the Ackley function to maximize it
    y = jnp.array(y_torch.numpy())
    return y[:, None]


def load_dropwave(x, dim):
    """
    Generates a dataset using the dropwave function and converts the dataset into JAX NumPy arrays.
    """
    x_torch = torch.tensor(np.array(x)) 
    dropwave = DropWave(dim=dim) 
    y_torch = -dropwave(x_torch) # Take the negative of the Ackley function to maximize it
    y = jnp.array(y_torch.numpy())
    return y[:, None]

