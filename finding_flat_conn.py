import torch
from torch import optim
import math

def spin_to_eigen(spins):
    """Converts spins to eigenvalues. """
    return torch.exp(2 * math.pi * 1j * torch.tensor(spins, dtype=torch.complex64))

def objective_prodx(U, a):
    """Calculates the objective function based on the product of matrices and their difference from the identity matrix."""
    Nc = len(a[0])
    product = torch.eye(Nc, dtype=torch.complex64)
    for i, ui in enumerate(U):
        xi = ui @ torch.diag(a[i]) @ torch.linalg.inv(ui)
        product = product @ xi
    mm = product - torch.eye(Nc, dtype=torch.complex64)
    return torch.sum(torch.abs(mm) ** 2)

def AUV_to_X(U, a):
    """Generates matrices based on the transformation defined by U and a."""
    return [ui @ torch.diag(ai) @ torch.linalg.inv(ui) for ui, ai in zip(U, a)]

def optimize_matrices(n, Nc, a, lr=0.005, steps=5001, log_interval=500):
    """Optimizes matrices U to minimize the objective function."""
    U = [torch.randn(Nc, Nc, dtype=torch.complex64, requires_grad=True) for _ in range(n)]
    optimizer = optim.Adam(U, lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        loss = objective_prodx(U, a)
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            print(f'Step {step}, Loss: {loss.item()}')

    return U

# Main execution
n, Nc = 3, 3
a = [
    spin_to_eigen([1/15, -1/3, 4/15]),
    spin_to_eigen([-1/3, -1/12, 5/12]),
    spin_to_eigen([-1/6, -1/6, 1/3])
]

optimized_U = optimize_matrices(n, Nc, a)
X_matrices = AUV_to_X(optimized_U, a)
print("Optimized Matrices:", X_matrices)
