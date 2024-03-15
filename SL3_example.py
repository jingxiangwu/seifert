import torch
from torch import optim
import math

def objective_prodx(u, v, a1, a2, a3):
    x1 = torch.diag(a1)
    x2 = u @ torch.diag(a2) @ torch.linalg.inv(u)
    x3 = v @ torch.diag(a3) @ torch.linalg.inv(v)
    mm = x1 @ x2 @ x3 - torch.eye(3, dtype=torch.complex64)
    return torch.sum(torch.abs(mm) ** 2)

def spin_to_eigen(spins):
    return torch.exp(2 * math.pi * 1j * torch.tensor(spins))

# Updated values for a1, a2, a3 with complex exponential components
a1 = spin_to_eigen([1/15, -1/3, 4/15])
a2 = spin_to_eigen([-1/3, -1/12, 5/12])
a3 = spin_to_eigen([-1/6, -1/6, 1/3])

# Initialize u and v as complex matrices
u = torch.randn(3, 3, dtype=torch.complex64, requires_grad=True)
v = torch.randn(3, 3, dtype=torch.complex64, requires_grad=True)

# Optimizer
optimizer = optim.Adam([u, v], lr=0.01)

# Optimization loop
for step in range(1000):  # Number of optimization steps
    optimizer.zero_grad()  # Zero the gradients
    loss = objective_prodx(u, v, a1, a2, a3)
    loss.backward()  # Compute the gradient
    optimizer.step()  # Update u and v
    
    if step % 100 == 0:
        print(f'Step {step}, Loss: {loss.item()}')

# Check the optimized values of u and v
print("Optimized u:", u)
print("Optimized v:", v)