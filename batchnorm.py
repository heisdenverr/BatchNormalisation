
import torch
from torch import nn

class BatchNorm(nn.Module):

  def __init__(self, features, epsilon=1e-5):
    super().__init__()

    self.epsilon = epsilon

    self.gamma = nn.Parameter(torch.ones(features))
    self.beta = nn.Parameter(torch.zeros(features))
  
  def forward(self, x):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + self.epsilon)
    out = self.gamma*x_norm + self.beta
    return out


# Define model
bn =BatchNorm(features=5)

# Create input tensor (batch_size=3, features=5)
x = torch.randn(3, 5, requires_grad=True)

# Forward pass
y = bn(x)

# Compute loss (sum of all outputs)
loss = y.sum()

# Backpropagation
loss.backward()

# Print computed gradients
print("Gradient of γ (dL/dγ):", bn.gamma.grad)  # Should be sum of normalized inputs
print("Gradient of β (dL/dβ):", bn.beta.grad)  # Should be sum of ones
print("Gradient of x (dL/dx):", x.grad)        # Should flow through BN layer correctly

# PyTorch built-in BatchNorm
bn_torch = nn.BatchNorm1d(5, affine=True)
x_torch = x.clone().detach().requires_grad_(True)  # Copy input

# Forward pass with PyTorch's BN
y_torch = bn_torch(x_torch)

# Backpropagation
loss_torch = y_torch.sum()
loss_torch.backward()

# Compare gradients
print("PyTorch γ grad:", bn_torch.weight.grad)
print("PyTorch β grad:", bn_torch.bias.grad)
print("PyTorch x grad:", x_torch.grad)
