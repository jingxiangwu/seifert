#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:00:32 2024

@author: jwu
"""

from itertools import product
import numpy as np
import torch
from torch import optim
import math
from prettytable import PrettyTable
import pandas as pd
import time
from sympy import symbols, Matrix, simplify

from google.colab import drive
import os
# Dynamically create a filename based on data attributes
def create_filename(data_info):
    Nc = data_info['Nc']
    p_values = '_'.join(map(str, data_info['p_list']))
    q_values = '_'.join(map(str, data_info['q_list']))
    return f'/content/drive/MyDrive/Nc{Nc}_p{p_values}_q{q_values}.pkl'
# Check if Google Drive is mounted; if not, mount it
if not os.path.isdir('/content/drive/My Drive'):
    drive.mount('/content/drive')
else:
    print("Google Drive is already mounted.")


# Determine if CUDA is available and choose the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eigen_degree(eigens_list):
    # count the number of equalities that are satisfied by this eigenvalue list
    def count_equalities_within_tolerance(elements, epsilon=1e-6):
      assert len(elements.shape) == 1
      count = 0
      N = len(elements)

      for i in range(N):
          next_i = (i + 1) % N
          if np.isclose(elements[i], elements[next_i], atol=epsilon).all():
              count += 1

      return count

    return [count_equalities_within_tolerance(np.array(eigens)) for eigens in eigens_list]

def objective_prodx(U, a):
    """
    Calculates the objective function \prod x_i = 1
    """
    Nc = len(a[0])
    product = torch.eye(Nc, dtype=torch.complex128, device=device)  # Initialize on the correct device
    for i, ui in enumerate(U):
        xi = ui @ torch.diag(a[i].to(device)) @ torch.linalg.inv(ui)  # Ensure a[i] is moved to device
        product = product @ xi
    mm = product - torch.eye(Nc, dtype=torch.complex128, device=device)
    return torch.sum(torch.abs(mm) ** 2)

def AUV_to_X(U, a):
    """Generates matrices based on transformation. X_i = U_i A_i U_i^-1 """
    return [ui @ torch.diag(ai.to(device)) @ torch.linalg.inv(ui) for ui, ai in zip(U, a)]  # Ensure ai is moved to device

def optimize_matrices(n, Nc, a, lr=0.005, steps=5001, log_interval=1000, loss_threshold=1e-14):
    """Optimizes matrices U."""
    U = [torch.randn(Nc, Nc, dtype=torch.complex128, requires_grad=True, device=device) for _ in range(n)]  # Initialize on device
    optimizer = optim.Adam(U, lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        loss = objective_prodx(U, a)
        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or loss.item() < loss_threshold:
            print(f'Step {step}, Loss: {loss.item():.4e}')

        if loss.item() < loss_threshold:
            print("Early stopping triggered.")
            break

    return U, loss.item()


def prepare_check_irreducible(n, Nc):
    """
    Pre-compute the symbolic expressions for checking irreducibility.
    Returns:
    - A function that takes a numpy matrix and performs the substitution
      to check irreducibility.
    """
    # Define the symbols for the r matrix.
    r_symbols = symbols(f'r_1:{Nc+1}_1:{Nc+1}')
    r_matrix = Matrix(Nc, Nc, r_symbols)

    # Generate symbolic X matrices.
    X_symbols = [[symbols(f'X{j}_{i+1}{k+1}') for i in range(Nc) for k in range(Nc)] for j in range(1, n+1)]
    X_matrices = [Matrix(Nc, Nc, X_symbols[j]) for j in range(n)]

    def commutator(x, rr):
        return x * rr - rr * x

    # Compute and flatten the commutators.
    commutators = [commutator(X, r_matrix) for X in X_matrices]
    commutators_flatten = [elem for comm in commutators for elem in comm]

    # Extract coefficients with respect to the r symbols.
    coefficients_matrix = Matrix(Nc**2 * n, Nc**2, lambda i, j: commutators_flatten[i].coeff(r_symbols[j]))

    # Function that performs the substitution
    def substitution_func(numpy_X):
        """
        Performs the substitution using the pre-computed symbolic expressions
        and the input numpy matrix to check for irreducibility.
        """
        subs_dict = {X_matrices[j][i, k]: numpy_X[j, i, k] for j in range(n) for i in range(Nc) for k in range(Nc)}
        substituted_matrix = coefficients_matrix.subs(subs_dict)
        return substituted_matrix

    # return the substitution function
    return substitution_func


# Main execution
data_all_fibres = {'Nc':3, 'p_list':[4,5,5], 'q_list':[1,1,1]}
Nc = data_all_fibres['Nc']
n  = len(data_all_fibres['p_list'])
filename = create_filename(data_all_fibres)
df_big = pd.read_pickle(filename)


conn_candidates = df_big['eigenvalues']
conn_candidates_tensor = [[torch.from_numpy(elem).to(device) for elem in a] for a in conn_candidates]

loss_threshold = 1e-12

conn_found = []
conn_not_found = []

print(f"Total number of candidate connections {len(conn_candidates_tensor)}")
tic = time.time()

for i in range(len(conn_candidates_tensor)):
  print(f"candidate {i}")
  a_tensor = conn_candidates_tensor[i]
  optimized_U, final_loss = optimize_matrices(n, Nc, a_tensor, lr=0.01,steps=20001, loss_threshold=loss_threshold)
  if final_loss < loss_threshold:
      X_matrices = AUV_to_X(optimized_U, a_tensor)
      # Store the necessary information for CPU processing
      conn_found.append((i, X_matrices, final_loss))
  else:
    conn_not_found.append((i, final_loss))


torch.cuda.synchronize(device)  # Ensure all GPU operations are complete
toc = time.time()
print(f"GPU operations completed in {toc - tic} seconds")

def process_results(X_matrices, final_loss, check_irreducible):
    numpy_matrices = [matrix.cpu().detach().numpy() for matrix in X_matrices]
    numpy_X = np.stack(numpy_matrices)
    coefficients_matrix_substituted = check_irreducible(numpy_X)
    coefficients_matrix_np = np.array(coefficients_matrix_substituted.tolist(), dtype=np.complex128)
    coefficients_matrix_rank = np.linalg.matrix_rank(coefficients_matrix_np, tol=1e-10)
    return coefficients_matrix_rank, final_loss

print("Processing CPU-bound tasks...")
check_irreducible = prepare_check_irreducible(n, Nc)

tic = time.time()

for i, X_matrices, final_loss in conn_found:
    rank, loss = process_results(X_matrices, final_loss, check_irreducible)
    print(f"candidate {i}")
    dyn = [arr.tolist() for arr in df_big.loc[i, "Dynkin label"]]
    eigen = conn_candidates[i]

    print('degrees are ', eigen_degree(eigen))
    print(f"Dynkin label is {dyn}")
    if rank == Nc**2 - 1:
        print('\033[92mIrreducible\033[0m since rank is Nc^2 - 1')
    else:
        print('Reducible')
    print(f'Final loss is {loss}')
    print('==================')

for i, final_loss in conn_not_found:
    print(f"candidate {i}")
    dyn = [arr.tolist() for arr in df_big.loc[i, "Dynkin label"]]
    eigen = conn_candidates[i]
    print('degrees are ', eigen_degree(eigen))
    print(f"Dynkin label is {dyn}")
    print(f'Final loss is {loss}')
    print('==================')

toc = time.time()
print(f"CPU operations completed in {toc - tic} seconds")