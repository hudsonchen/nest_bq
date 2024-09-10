import torch
import jax
import jax.numpy as jnp
import numpy as np
from botorch.test_functions import Ackley
from botorch.utils.datasets import FixedNoiseDataset


def emulator(x):
    mean = 0.45321
    std = 4.4258
    # return ((6 * x - 2) ** 2 * jnp.sin(12 * x - 4) - mean) / std
    return jnp.exp(-(x - 2)**2) + jnp.exp(-((x - 6)**2) / 10) + 1 / (x**2 + 1)


def load_ackley(x: jnp.Array, n_samples: int):
    """
    Generates a dataset using the Ackley function and converts the dataset into JAX NumPy arrays.
    """
    x_torch = torch.tensor(np.array(x)) 
    ackley = Ackley(dim=2)  # Ackley function with specified dimensionality
    y_torch = ackley(x_torch)  # Maximize the function
    y = jnp.array(y_torch.numpy())
    return x, y

