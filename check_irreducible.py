from sympy import symbols, Matrix, simplify
import numpy as np

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

# Example usage
numpy_X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Example 3D NumPy matrix
coefficients_matrix_substituted = check_irreducible(numpy_X)
display(coefficients_matrix_substituted)
