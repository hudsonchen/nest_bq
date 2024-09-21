import torch
import jax
import numpy as np
from botorch.test_functions import Ackley, DropWave, Branin, Cosine8
from botorch.utils.datasets import FixedNoiseDataset


def emulator(x):
    mean = 0.45321
    std = 4.4258
    # return ((6 * x - 2) ** 2 * np.sin(12 * x - 4) - mean) / std
    return np.exp(-(x - 2)**2) + np.exp(-((x - 6)**2) / 10) + 1 / (x**2 + 1)


def load_ackley(x, dim):
    """
    Generates a dataset using the Ackley function and converts the dataset into JAX NumPy arrays.
    """
    x_torch = torch.tensor(np.array(x)) 
    ackley = Ackley(dim=dim) 
    y_torch = -ackley(x_torch) # Take the negative of the Ackley function to maximize it
    y = np.array(y_torch.numpy())
    return y[:, None]


def load_dropwave(x):
    """
    Generates a dataset using the dropwave function and converts the dataset into JAX NumPy arrays.
    """
    x_torch = torch.tensor(np.array(x)) 
    dropwave = DropWave() 
    y_torch = -dropwave(x_torch) # Take the negative of the Ackley function to maximize it
    y = np.array(y_torch.numpy())
    return y[:, None]


def load_branin(x):
    """
    Generates a dataset using the Branin function and converts the dataset into JAX NumPy arrays.
    """
    x_torch = torch.tensor(np.array(x)) 
    branin = Branin() 
    y_torch = -branin(x_torch) # Take the negative of the Branin function to maximize it
    y = np.array(y_torch.numpy())
    return y[:, None]

def load_cosine8(x):
    """
    Generates a dataset using the Cosine8 function and converts the dataset into JAX NumPy arrays.
    """
    x_torch = torch.tensor(np.array(x)) 
    cosine8 = Cosine8() 
    y_torch = cosine8(x_torch) # Take the negative of the Cosine8 function to maximize it
    y = np.array(y_torch.numpy())
    return y[:, None]