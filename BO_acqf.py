import torch
from botorch import settings
from botorch.acquisition import AcquisitionFunction
from botorch.sampling import SobolQMCNormalSampler, IIDNormalSampler
import matplotlib.pyplot as plt
from utils.kernels import my_Matern_12_product
import numpy as np
from botorch.optim import optimize_acqf

COUNT = 0

class my_EI(AcquisitionFunction):
    def __init__(self, model, args, num_samples, u, weights, y_best):
        super().__init__(model)
        self.model = model  # Store the model for access in forward
        self.args = args
        self.num_samples = num_samples
        self.u = u
        self.weights = weights
        self.y_best = y_best
        self.print = True

    def forward(self, X):
        """
        X is a tensor of shape (batch_size, q, d), where d is the input dimensionality.
        This method should return a tensor of shape (batch_size,), representing the
        acquisition value for each point in X.
        """
        posterior = self.model(X)  # Get the GP posterior distribution at X
        mu, variance = posterior.mean, posterior.variance 
        if len(mu.shape) == 3:
            mu = mu.squeeze(0)
            variance = variance.squeeze(0)
        samples = (torch.sqrt(variance)[:,:, None] * self.u[None, None, :] + mu[:,:, None]) # (batch_size, q, num_samples)
        if len(self.y_best.shape) == 2:
            self.y_best = self.y_best.T[:,:, None]
        increases = torch.maximum(samples - self.y_best, torch.tensor(0)) 
        max_increases = increases.max(dim=1).values # shape: (batch_size, num_samples)
        ei = (max_increases * self.weights[None, :]).sum(dim=1)   # shape: (batch_size,)

        # Debug code
        # if self.print:
        #     global COUNT
        #     COUNT += 1
        #     u_all = np.linspace(-3, 3, 200)
        #     plt.figure()
        #     plt.scatter(self.u, max_increases[0, :].detach().numpy())
        #     K_inv = np.linalg.inv(my_Matern_12_product(self.u.numpy()[:, None], self.u.numpy()[:, None], np.ones([1])) + 1e-6 * np.eye(self.num_samples))
        #     kernel_sol = my_Matern_12_product(u_all[:, None], self.u.numpy()[:, None], np.ones([1])) @ K_inv @ max_increases[0, :].detach().numpy()
        #     plt.plot(u_all, kernel_sol)
        #     plt.savefig(f"{self.args.save_path}/kernel_debug_{COUNT}.pdf")
        #     self.print = False
        # Debug code for q=1
        # std = torch.sqrt(variance)
        # z = (mu - self.y_best) / std
        # ei_closed_form = std * (z * torch.distributions.Normal(0, 1).cdf(z) + torch.distributions.Normal(0, 1).log_prob(z).exp())
        # ei_closed_form = ei_closed_form.squeeze()
        # ei_monte_carlo = max_increases.mean(dim=1)
        # print(f'MC error {torch.abs(ei_monte_carlo - ei_closed_form).mean().item()}')
        # print(f'KQ error {torch.abs(ei - ei_closed_form).mean().item()}')
        # # # Debug code
        return ei 
    

class my_lookahead_EI(AcquisitionFunction):
    def __init__(self, model, args, num_samples, num_fantasies, bounds, u, weights, y_best):
        super().__init__(model)
        self.model = model  # Store the model for access in forward
        self.args = args
        self.num_samples = num_samples
        self.bounds = bounds
        self.u = u
        self.weights = weights
        self.y_best = y_best
        self.print = True
        self.sampler = IIDNormalSampler(sample_shape=torch.Size([1]), resample=False)
        self.num_fantasies = num_fantasies

    def forward(self, X):
        """
        X is a tensor of shape (batch_size, q + num_fatansy, d), where d is the input dimensionality.
        This method should return a tensor of shape (batch_size,), representing the
        acquisition value for each point in X.
        """
        X_actual, X_fantasies = _split_fantasy_points(X, self.num_fantasies)
        posterior = self.model(X_actual)  # Get the GP posterior distribution at X
        mu, variance = posterior.mean, posterior.variance 
        samples = (torch.sqrt(variance)[:,:, None] * self.u[None, None, :] + mu[:,:, None]) # (batch_size, q, num_samples)
        increases = torch.maximum(samples - self.y_best, torch.tensor(0)) 
        max_increases = increases.max(dim=1).values # shape: (batch_size, num_samples)
        ei = (max_increases * self.weights[None, :]).sum(dim=1)   # shape: (batch_size,)
            
        # Step 2: Calculate the one-step lookahead EI using fantasized observations
        fantasy_model = self.model.fantasize(X_actual, sampler=self.sampler, observation_noise=False)  # shape: (batch_size, q, num_fantasies)
        
        # Determine best value under fantasy model
        best_f = fantasy_model.train_targets.max(dim=-1)[0]

        # Define one-step EI acquisition function under the fantasized model
        one_step_ei = my_EI(model=fantasy_model, args=self.args, num_samples=self.num_samples, 
                            u=self.u, weights=self.weights, y_best=best_f)  # shape: (batch_size, q, num_fantasies)

        # Calculate the one-step lookahead EI for each point in X
        with settings.propagate_grads(True):
            values = one_step_ei(X_fantasies)
            
        # Average the one-step EI values across fantasies
        one_step_ei_avg = values.mean(dim=0)  # Averaging over the fantasy samples

        # Step 3: Return combined EI (current EI + lookahead EI)
        return ei + one_step_ei_avg


class my_closed_form_EI(AcquisitionFunction):
    def __init__(self, model, y_best):
        super().__init__(model)
        self.model = model
        self.y_best = y_best

    def forward(self, X):
        """
        X is a tensor of shape (batch_size, q, d), where d is the input dimensionality.
        This method should return a tensor of shape (batch_size,), representing the
        acquisition value for each point in X.
        """
        posterior = self.model(X)  # Get the GP posterior distribution at X
        mu, variance = posterior.mean, posterior.variance
        std = torch.sqrt(variance)
        z = (mu - self.y_best) / std
        ei = std * (z * torch.distributions.Normal(0, 1).cdf(z) + torch.distributions.Normal(0, 1).log_prob(z).exp())
        ei = ei.squeeze()
        return ei
    
def _split_fantasy_points(X, n_f: int):
    r"""Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `batch_shape x n_f x d`-dim tensor `X_fantasies` of fantasy points.
    """
    if n_f > X.size(-2):
        raise ValueError(
            f"n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})"
        )
    split_sizes = [X.size(-2) - n_f, n_f]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    return X_actual, X_fantasies