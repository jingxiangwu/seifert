from itertools import product
import numpy as np
import torch
from torch import optim
import math

def spin_to_eigen(spins):
    return torch.exp(2 * math.pi * 1j * torch.tensor(spins))
    
def pq_to_eigen(p, q, l, Nc):
    def rules(eigens):
        return np.all(eigens[:-1] < eigens[1:]) and (eigens[-1] < eigens[0] + p)

    cartesian_products = product(range(-2 * p, 2 * p + 1), repeat=Nc-1)

    eigens_list = []
    for eigen_tuple in cartesian_products:
        eigen_adjusted = np.array([(ei - (q % Nc) * l / Nc) for ei in eigen_tuple])
        # Calculate the Nc-th eigenvalue
        eigen_final = np.append(eigen_adjusted, -np.sum(eigen_adjusted))
        # Sort the array before adding to the list
        eigens_list.append(np.sort(eigen_final))

    # Remove duplicates: convert arrays to tuples for set operation, then back to list of arrays
    eigens_list_unique = np.unique(np.array([tuple(eigen) for eigen in eigens_list]), axis=0)

    # Filtering based on the `rules`, converting each tuple back to list
    eigens_filtered = [spin_to_eigen(eigen/p) for eigen in eigens_list_unique if rules(eigen)]

    
    return eigens_filtered

def seifert_to_eigen(pqlist, Nc):
    # for pq in pqlist:
        # print(len(pq_to_eigen(pq[0],pq[1], 0, Nc)))
        # print(pq_to_eigen(pq[0],pq[1], 0, Nc))
    return list(product(*[pq_to_eigen(pq[0],pq[1], 0, Nc) for pq in pqlist]))

def count_equalities_within_tolerance(elements, epsilon=1e-6):
    assert len(elements.shape) == 1
    count = 0
    N = len(elements)
    
    # Iterate through the list, comparing each element with the next
    for i in range(N):
        # Use modulo to wrap around to the first element after the last
        next_i = (i + 1) % N
        # torch.isclose returns a tensor of Booleans, so we use torch.all to check if all elements satisfy the condition
        if torch.all(torch.isclose(elements[i], elements[next_i], atol=epsilon)):
            count += 1
            
    return count
def eigen_degree(eigens_list):
    return [count_equalities_within_tolerance(eigens) for eigens in eigens_list]

def spin_to_eigen(spins):
    """Converts spins to eigenvalues. """
    return torch.exp(2 * math.pi * 1j * torch.tensor(spins, dtype=torch.complex128))

def objective_prodx(U, a):
    """Calculates the objective function based on the product of matrices and their difference from the identity matrix."""
    Nc = len(a[0])
    product = torch.eye(Nc, dtype=torch.complex128)
    for i, ui in enumerate(U):
        xi = ui @ torch.diag(a[i]) @ torch.linalg.inv(ui)
        product = product @ xi
    mm = product - torch.eye(Nc, dtype=torch.complex128)
    return torch.sum(torch.abs(mm) ** 2)

def AUV_to_X(U, a):
    """Generates matrices based on the transformation defined by U and a."""
    return [ui @ torch.diag(ai) @ torch.linalg.inv(ui) for ui, ai in zip(U, a)]

def optimize_matrices(n, Nc, a, lr=0.005, steps=5001, log_interval=1000, loss_threshold=1e-14):
    """Optimizes matrices U to minimize the objective function."""
    U = [torch.randn(Nc, Nc, dtype=torch.complex128, requires_grad=True) for _ in range(n)]
    optimizer = optim.Adam(U, lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        loss = objective_prodx(U, a)
        loss.backward()
        optimizer.step()

        #if step % log_interval == 0 or loss.item() < loss_threshold:
            #print(f'Step {step}, Loss: {loss.item()}')
        
        if loss.item() < loss_threshold:
            #print("Early stopping triggered.")
            break

    return U, loss.item()

def chop(num, tol = 1e-14):
    tol = 1e-15
    num[np.abs(num) < tol] = 0
    return num

from sympy import symbols, Matrix, simplify

def check_irreducible(numpy_X):
    """
    Given a NumPy matrix [X1, X2, ..., Xn], this function computes the coefficients matrix
    for the commutators [Xi,r], then substitutes
    the symbolic 'X' with values from the NumPy matrix.
    """
    n, Nc, _ = numpy_X.shape  # Shape of numpy_X to determine the number of matrices and their sizes.

    # Define the symbols for the r matrix.
    r_symbols = symbols(f'r_1:{Nc+1}_1:{Nc+1}')
    r_matrix = Matrix(Nc, Nc, r_symbols)

    # Generate symbolic X matrices.
    X_symbols = [[symbols(f'X{j}_{i+1}{k+1}') for i in range(Nc) for k in range(Nc)] for j in range(1, n+1)]
    X_matrices = [Matrix(Nc, Nc, X_symbols[j]) for j in range(n)]

    # Commutator function.
    def commutator(x, rr):
        return x * rr - rr * x

    # Compute and flatten the commutators.
    commutators = [commutator(X, r_matrix) for X in X_matrices]
    commutators_flatten = [elem for comm in commutators for elem in comm]

    # Extract coefficients with respect to the r symbols.
    coefficients_matrix = Matrix(Nc**2 * n, Nc**2, lambda i, j: commutators_flatten[i].coeff(r_symbols[j]))

    # Perform the substitution.
    subs_dict = {X_matrices[j][i, k]: numpy_X[j, i, k] for j in range(n) for i in range(Nc) for k in range(Nc)}
    substituted_matrix = coefficients_matrix.subs(subs_dict)

    return substituted_matrix

# Main execution
import time
n, Nc = 3, 4
loss_threshold = 1e-14

tic = time.time()
for a in seifert_to_eigen([[5,1],[5,1],[7,1]], Nc):
    #print('degrees are ', eigen_degree(a))
    optimized_U, final_loss = optimize_matrices(n, Nc, a, steps=20001, loss_threshold=loss_threshold)
    #print(f'final loss is {final_loss}')
    if final_loss < loss_threshold:
        #print('\033[92m' + 'valid flat connection' + '\033[0m')
    
        X_matrices = AUV_to_X(optimized_U, a)
        numpy_matrices = [matrix.detach().numpy() for matrix in X_matrices]
        numpy_X = np.stack(numpy_matrices)
        coefficients_matrix_substituted = check_irreducible(numpy_X)
        coefficients_matrix_np = np.array(coefficients_matrix_substituted.tolist(), dtype=np.complex128)
        coefficients_matrix_rank = np.linalg.matrix_rank(coefficients_matrix_np,tol=1e-10)
        # print('rank of commutators are ', coefficients_matrix_rank)
        # if coefficients_matrix_rank == Nc**2-1:
        #     print('\033[92m' + 'irreducible ' + '\033[0m')
        # else:
        #     print('reducible')
    else:
        break
        #print('invalid')
    #print('==================')

toc = time.time()
print(toc-tic)